#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from backend_api import BackendAPIError, BackendClient

IDK_ANSWER = "I don't know based on the provided documents."
CITATION_PATTERN = re.compile(r"\[source=.*?page=.*?chunk=.*?\]")


@dataclass
class EvalRow:
    sample_id: str
    question: str
    eval_session_id: str | None
    rewritten_question: str | None
    model: str
    latency_ms: float
    cosine_similarity: float
    citation_coverage: float
    source_precision: float
    source_f1: float
    answer_has_citation: float
    required_term_coverage: float | None
    rewrite_term_coverage: float | None
    idk_correct: float | None
    notes: str


class LangfuseEvalLogger:
    def __init__(
        self,
        enabled: bool,
        run_name: str,
        host: str | None,
        public_key: str | None,
        secret_key: str | None,
    ) -> None:
        self.enabled = enabled
        self.run_name = run_name
        self.client: Any | None = None

        if not enabled:
            return

        if not public_key or not secret_key:
            raise ValueError(
                "Langfuse logging requested but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are missing"
            )

        try:
            from langfuse import Langfuse
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Langfuse integration requires the `langfuse` package. "
                "Install it in your eval environment before using --langfuse."
            ) from exc

        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

    def log_sample(
        self,
        row: EvalRow,
        sample: dict[str, Any],
        response_payload: dict[str, Any],
        backend_config: dict[str, Any],
    ) -> None:
        if not self.enabled or self.client is None:
            return

        trace = self.client.trace(
            name="rag-eval-sample",
            session_id=row.eval_session_id,
            input={
                "question": row.question,
                "expected_answer": sample.get("expected_answer", ""),
                "expected_sources": sample.get("expected_sources", []),
            },
            output={
                "answer": response_payload.get("answer", ""),
                "rewritten_question": row.rewritten_question,
                "sources": response_payload.get("sources", []),
            },
            metadata={
                "eval_run": self.run_name,
                "sample_id": row.sample_id,
                "conversation_id": sample.get("conversation_id"),
                "backend_config": backend_config,
                "notes": row.notes,
            },
            tags=["evaluation", "rag-local"],
        )

        metric_values: list[tuple[str, float | None]] = [
            ("cosine_similarity", row.cosine_similarity),
            ("citation_coverage", row.citation_coverage),
            ("source_precision", row.source_precision),
            ("source_f1", row.source_f1),
            ("answer_has_citation", row.answer_has_citation),
            ("required_term_coverage", row.required_term_coverage),
            ("rewrite_term_coverage", row.rewrite_term_coverage),
            ("idk_correct", row.idk_correct),
            ("latency_ms", row.latency_ms),
        ]
        for metric_name, metric_value in metric_values:
            if metric_value is None:
                continue
            trace.score(name=metric_name, value=float(metric_value))

    def flush(self) -> None:
        if not self.enabled or self.client is None:
            return
        self.client.flush()
        self.client.shutdown()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _normalize_source_name(source: str) -> str:
    return Path(source).name.lower()


def _safe_mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _optional_ratio(items: list[str], text: str) -> float | None:
    if not items:
        return None
    lowered_text = text.lower()
    matches = sum(1 for item in items if item.lower() in lowered_text)
    return matches / len(items)


def cosine_similarity(model: SentenceTransformer, expected: str, generated: str) -> float:
    vectors = model.encode([expected, generated], normalize_embeddings=True)
    score = float(np.dot(vectors[0], vectors[1]))
    return max(0.0, min(1.0, score))


def source_recall_precision_f1(
    expected_sources: list[dict[str, Any]],
    predicted_sources: list[dict[str, Any]],
) -> tuple[float, float, float]:
    expected = {
        (_normalize_source_name(str(item["source"])), int(item["page"]))
        for item in expected_sources
        if "source" in item and "page" in item
    }
    predicted = {
        (_normalize_source_name(str(item["source"])), int(item["page"]))
        for item in predicted_sources
        if "source" in item and "page" in item
    }

    if not expected and not predicted:
        return 1.0, 1.0, 1.0

    overlap = expected.intersection(predicted)
    recall = len(overlap) / len(expected) if expected else 1.0
    precision = len(overlap) / len(predicted) if predicted else 0.0
    if recall + precision == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * recall * precision / (recall + precision)
    return recall, precision, f1


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Dataset at {path} must be a JSON array")
    return payload


def write_csv(rows: list[EvalRow], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_id",
                "question",
                "eval_session_id",
                "rewritten_question",
                "model",
                "latency_ms",
                "cosine_similarity",
                "citation_coverage",
                "source_precision",
                "source_f1",
                "answer_has_citation",
                "required_term_coverage",
                "rewrite_term_coverage",
                "idk_correct",
                "notes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.sample_id,
                    row.question,
                    row.eval_session_id or "",
                    row.rewritten_question or "",
                    row.model,
                    f"{row.latency_ms:.2f}",
                    f"{row.cosine_similarity:.4f}",
                    f"{row.citation_coverage:.4f}",
                    f"{row.source_precision:.4f}",
                    f"{row.source_f1:.4f}",
                    f"{row.answer_has_citation:.4f}",
                    "" if row.required_term_coverage is None else f"{row.required_term_coverage:.4f}",
                    "" if row.rewrite_term_coverage is None else f"{row.rewrite_term_coverage:.4f}",
                    "" if row.idk_correct is None else f"{row.idk_correct:.4f}",
                    row.notes,
                ]
            )


def write_predictions_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def write_markdown(
    rows: list[EvalRow],
    output_path: Path,
    backend_config: dict[str, Any],
    embedding_model: str,
    run_name: str,
    dataset_path: Path,
) -> None:
    if not rows:
        raise ValueError("No evaluation rows to save")

    mean_cosine = _safe_mean([row.cosine_similarity for row in rows])
    mean_coverage = _safe_mean([row.citation_coverage for row in rows])
    mean_precision = _safe_mean([row.source_precision for row in rows])
    mean_f1 = _safe_mean([row.source_f1 for row in rows])
    mean_citation = _safe_mean([row.answer_has_citation for row in rows])
    mean_required = _safe_mean([row.required_term_coverage for row in rows])
    mean_rewrite = _safe_mean([row.rewrite_term_coverage for row in rows])
    mean_idk = _safe_mean([row.idk_correct for row in rows])
    mean_latency = _safe_mean([row.latency_ms for row in rows])

    lines = [
        "# Evaluation Results",
        "",
        f"- Run: `{run_name}`",
        f"- Timestamp (UTC): `{datetime.now(UTC).isoformat()}`",
        f"- Dataset: `{dataset_path}`",
        f"- Embedding model (metric): `{embedding_model}`",
        "",
        "## Backend Config Snapshot",
        "",
        f"- `ollama_model`: `{backend_config.get('ollama_model', 'unknown')}`",
        f"- `chunk_size`: `{backend_config.get('chunk_size', 'unknown')}`",
        f"- `chunk_overlap`: `{backend_config.get('chunk_overlap', 'unknown')}`",
        f"- `top_k`: `{backend_config.get('top_k', 'unknown')}`",
        f"- `use_mmr`: `{backend_config.get('use_mmr', 'unknown')}`",
        f"- `query_rewrite_enabled`: `{backend_config.get('query_rewrite_enabled', 'unknown')}`",
        f"- `memory_enabled`: `{backend_config.get('memory_enabled', 'unknown')}`",
        "",
        "## Per-Sample Table",
        "",
        "| id | cosine | citation_coverage | source_f1 | answer_has_citation | required_term_coverage | rewrite_term_coverage | idk_correct | latency_ms | model | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f"| {row.sample_id} | {row.cosine_similarity:.3f} | {row.citation_coverage:.3f} | "
            f"{row.source_f1:.3f} | {row.answer_has_citation:.3f} | "
            f"{_fmt_optional(row.required_term_coverage)} | {_fmt_optional(row.rewrite_term_coverage)} | "
            f"{_fmt_optional(row.idk_correct)} | {row.latency_ms:.1f} | {row.model} | {row.notes} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Mean cosine similarity: **{_fmt_optional(mean_cosine)}**",
            f"- Mean citation coverage (source-page recall): **{_fmt_optional(mean_coverage)}**",
            f"- Mean source precision: **{_fmt_optional(mean_precision)}**",
            f"- Mean source F1: **{_fmt_optional(mean_f1)}**",
            f"- Mean answer citation presence: **{_fmt_optional(mean_citation)}**",
            f"- Mean required-term coverage: **{_fmt_optional(mean_required)}**",
            f"- Mean rewrite-term coverage: **{_fmt_optional(mean_rewrite)}**",
            f"- Mean IDK correctness: **{_fmt_optional(mean_idk)}**",
            f"- Mean latency (ms): **{_fmt_optional(mean_latency)}**",
            "",
            "## Commentary",
            "",
            "- `citation_coverage` and `source_f1` reflect grounding quality, not only text similarity.",
            "- `rewrite_term_coverage` is reported only for follow-up cases that define expected rewrite hints.",
            "- `idk_correct` is reported only for samples that explicitly expect abstention.",
            "- Use this report for regression tracking after retrieval/prompt/model changes.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_session_id(
    sample: dict[str, Any],
    default_session_id: str | None,
    session_prefix: str,
    conversation_sessions: dict[str, str],
) -> str | None:
    sample_session_id = sample.get("session_id")
    if isinstance(sample_session_id, str) and sample_session_id.strip():
        return sample_session_id.strip()

    conversation_id = sample.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id.strip():
        key = conversation_id.strip()
        if key not in conversation_sessions:
            conversation_sessions[key] = f"{session_prefix}-{key}"
        return conversation_sessions[key]

    return default_session_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RAG evaluation against FastAPI backend")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="scripts/eval_dataset.json")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model for cosine metric. Default: backend /config embedding_model.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--auto-ingest", action="store_true", help="Automatically call /ingest when index is empty")
    parser.add_argument("--force-ingest", action="store_true", help="Force rebuild index before evaluation")
    parser.add_argument(
        "--default-session-id",
        default=None,
        help="Optional default session id for all samples without session_id/conversation_id",
    )
    parser.add_argument("--session-prefix", default="eval", help="Session prefix for conversation_id-derived sessions")
    parser.add_argument("--run-name", default=None, help="Optional run name for artifacts and Langfuse")
    parser.add_argument("--langfuse", action="store_true", help="Log per-sample eval traces/scores to Langfuse")
    parser.add_argument("--langfuse-host", default=os.getenv("LANGFUSE_HOST", "http://localhost:3000"))
    parser.add_argument("--langfuse-public-key", default=os.getenv("LANGFUSE_PUBLIC_KEY"))
    parser.add_argument("--langfuse-secret-key", default=os.getenv("LANGFUSE_SECRET_KEY"))
    args = parser.parse_args()

    run_name = args.run_name or f"eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = BackendClient(args.backend_url, timeout_seconds=args.timeout_seconds)

    try:
        health = client.health()
        config_payload = client.config()
    except BackendAPIError as exc:
        raise SystemExit(f"Backend unavailable: {exc}") from exc

    backend_config = config_payload.get("config", {}) if isinstance(config_payload, dict) else {}
    if args.force_ingest:
        ingest_summary = client.ingest(force=True)
        print(f"[ingest] {json.dumps(ingest_summary)}")
    elif args.auto_ingest and int(health.get("indexed_chunks", 0)) == 0:
        ingest_summary = client.ingest(force=False)
        print(f"[ingest] {json.dumps(ingest_summary)}")

    embedding_model_name = args.embedding_model or str(
        backend_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    print(f"[eval] using embedding model for cosine metric: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    logger = LangfuseEvalLogger(
        enabled=args.langfuse,
        run_name=run_name,
        host=args.langfuse_host,
        public_key=args.langfuse_public_key,
        secret_key=args.langfuse_secret_key,
    )

    samples = load_dataset(dataset_path)
    rows: list[EvalRow] = []
    records: list[dict[str, Any]] = []
    conversation_sessions: dict[str, str] = {}

    for sample in samples:
        sample_id = str(sample["id"])
        question = str(sample["question"]).strip()
        expected_answer = str(sample["expected_answer"])
        expected_sources = sample.get("expected_sources", [])
        required_terms = sample.get("required_terms", [])
        rewrite_terms = sample.get("expected_rewrite_contains", [])
        expect_idk = sample.get("expect_idk")

        eval_session_id = _resolve_session_id(
            sample=sample,
            default_session_id=args.default_session_id,
            session_prefix=args.session_prefix,
            conversation_sessions=conversation_sessions,
        )

        started = time.perf_counter()
        payload = client.query(question=question, session_id=eval_session_id)
        latency_ms = (time.perf_counter() - started) * 1000.0

        generated_answer = str(payload.get("answer", ""))
        model_name = str(payload.get("model", "unknown"))
        rewritten_question = payload.get("rewritten_question")
        rewritten_question = str(rewritten_question) if rewritten_question is not None else None
        predicted_sources = payload.get("sources", [])

        cos = cosine_similarity(model, expected_answer, generated_answer)
        recall, precision, f1 = source_recall_precision_f1(expected_sources, predicted_sources)
        has_citation = 1.0 if CITATION_PATTERN.search(generated_answer) else 0.0
        required_coverage = _optional_ratio(required_terms, generated_answer)
        rewrite_coverage = _optional_ratio(rewrite_terms, rewritten_question or question)

        idk_correct: float | None = None
        if isinstance(expect_idk, bool):
            normalized_answer = _normalize_text(generated_answer)
            predicted_idk = normalized_answer == _normalize_text(IDK_ANSWER)
            idk_correct = 1.0 if predicted_idk == expect_idk else 0.0

        notes: list[str] = []
        if recall < 1.0:
            notes.append("partial_expected_source_coverage")
        if has_citation == 0.0:
            notes.append("missing_answer_citation")
        if required_coverage is not None and required_coverage < 1.0:
            notes.append("missing_required_terms")
        if rewrite_coverage is not None and rewrite_coverage < 1.0:
            notes.append("rewrite_terms_not_fully_matched")
        if idk_correct is not None and idk_correct < 1.0:
            notes.append("idk_mismatch")
        note_text = "ok" if not notes else ";".join(notes)

        row = EvalRow(
            sample_id=sample_id,
            question=question,
            eval_session_id=eval_session_id,
            rewritten_question=rewritten_question,
            model=model_name,
            latency_ms=latency_ms,
            cosine_similarity=cos,
            citation_coverage=recall,
            source_precision=precision,
            source_f1=f1,
            answer_has_citation=has_citation,
            required_term_coverage=required_coverage,
            rewrite_term_coverage=rewrite_coverage,
            idk_correct=idk_correct,
            notes=note_text,
        )
        rows.append(row)

        record = {
            "sample": sample,
            "response": payload,
            "metrics": {
                "cosine_similarity": cos,
                "citation_coverage": recall,
                "source_precision": precision,
                "source_f1": f1,
                "answer_has_citation": has_citation,
                "required_term_coverage": required_coverage,
                "rewrite_term_coverage": rewrite_coverage,
                "idk_correct": idk_correct,
                "latency_ms": latency_ms,
            },
            "note": note_text,
        }
        records.append(record)

        logger.log_sample(
            row=row,
            sample=sample,
            response_payload=payload,
            backend_config=backend_config,
        )

    logger.flush()

    csv_path = output_dir / "evaluation_results.csv"
    md_path = output_dir / "evaluation_results.md"
    predictions_path = output_dir / "evaluation_predictions.jsonl"

    write_csv(rows, csv_path)
    write_markdown(
        rows=rows,
        output_path=md_path,
        backend_config=backend_config,
        embedding_model=embedding_model_name,
        run_name=run_name,
        dataset_path=dataset_path,
    )
    write_predictions_jsonl(records, predictions_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote predictions JSONL: {predictions_path}")


if __name__ == "__main__":
    main()
