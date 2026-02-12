from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.rag_service import RagService
from app.schemas import EvalRunRequest

logger = logging.getLogger(__name__)

IDK_ANSWER = "I don't know based on the provided documents."
CITATION_PATTERN = re.compile(r"\[source=.*?page=.*?chunk=.*?\]")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _cosine_similarity(model: SentenceTransformer, expected: str, generated: str) -> float:
    vectors = model.encode([expected, generated], normalize_embeddings=True)
    score = float(np.dot(vectors[0], vectors[1]))
    return max(0.0, min(1.0, score))


def _source_recall_precision_f1(
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


class EvaluationService:
    def __init__(self, settings: Settings, rag_service: RagService) -> None:
        self.settings = settings
        self.rag_service = rag_service
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None

    def start_job(self, payload: EvalRunRequest) -> dict[str, Any]:
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active and active.get("status") in {"queued", "running"}:
                    raise RuntimeError(f"Evaluation job already running: {self._active_job_id}")

            job_id = uuid4().hex
            job = {
                "job_id": job_id,
                "status": "queued",
                "created_at": _now_iso(),
                "started_at": None,
                "finished_at": None,
                "processed": 0,
                "total": 0,
                "current_sample_id": None,
                "message": "Queued",
                "error": None,
                "summary": {},
                "artifacts": {},
                "rows": [],
                "options": payload.model_dump(),
                "artifact_files": {},
            }
            self._jobs[job_id] = job
            self._active_job_id = job_id

        worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        worker.start()
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Evaluation job queued",
        }

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            status = {
                "job_id": job["job_id"],
                "status": job["status"],
                "created_at": job["created_at"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "processed": int(job["processed"]),
                "total": int(job["total"]),
                "progress": (float(job["processed"]) / float(job["total"])) if job["total"] > 0 else 0.0,
                "current_sample_id": job["current_sample_id"],
                "message": job["message"],
                "error": job["error"],
                "summary": dict(job.get("summary", {})),
                "artifacts": dict(job.get("artifacts", {})),
            }
            return status

    def get_results(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "summary": dict(job.get("summary", {})),
                "artifacts": dict(job.get("artifacts", {})),
                "rows": list(job.get("rows", [])),
            }

    def get_artifact_file(self, job_id: str, artifact_name: str) -> Path | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            files = job.get("artifact_files", {})
            value = files.get(artifact_name)
            if not value:
                return None
            return Path(value)

    def _update_job(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(kwargs)

    def _set_active_job(self, job_id: str | None) -> None:
        with self._lock:
            self._active_job_id = job_id

    def _resolve_session_id(
        self,
        sample: dict[str, Any],
        default_session_id: str | None,
        session_prefix: str,
        job_id: str,
        conversation_sessions: dict[str, str],
    ) -> str | None:
        sample_session_id = sample.get("session_id")
        if isinstance(sample_session_id, str) and sample_session_id.strip():
            return sample_session_id.strip()

        conversation_id = sample.get("conversation_id")
        if isinstance(conversation_id, str) and conversation_id.strip():
            key = conversation_id.strip()
            if key not in conversation_sessions:
                conversation_sessions[key] = f"{session_prefix}-{job_id[:8]}-{key}"
            return conversation_sessions[key]

        return default_session_id

    def _load_dataset(self, dataset_path: Path) -> list[dict[str, Any]]:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Dataset at {dataset_path} must be a JSON array")
        return payload

    def _write_csv(self, rows: list[dict[str, Any]], output_path: Path) -> None:
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
                        row["sample_id"],
                        row["question"],
                        row.get("eval_session_id", ""),
                        row.get("rewritten_question", ""),
                        row["model"],
                        f"{row['latency_ms']:.2f}",
                        f"{row['cosine_similarity']:.4f}",
                        f"{row['citation_coverage']:.4f}",
                        f"{row['source_precision']:.4f}",
                        f"{row['source_f1']:.4f}",
                        f"{row['answer_has_citation']:.4f}",
                        ""
                        if row["required_term_coverage"] is None
                        else f"{row['required_term_coverage']:.4f}",
                        ""
                        if row["rewrite_term_coverage"] is None
                        else f"{row['rewrite_term_coverage']:.4f}",
                        "" if row["idk_correct"] is None else f"{row['idk_correct']:.4f}",
                        row["notes"],
                    ]
                )

    def _write_predictions_jsonl(self, records: list[dict[str, Any]], output_path: Path) -> None:
        lines = [json.dumps(record, ensure_ascii=True) for record in records]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_markdown(
        self,
        rows: list[dict[str, Any]],
        output_path: Path,
        summary: dict[str, float | None],
        run_name: str,
        dataset_path: Path,
        embedding_model: str,
    ) -> None:
        lines = [
            "# Evaluation Results",
            "",
            f"- Run: `{run_name}`",
            f"- Timestamp (UTC): `{_now_iso()}`",
            f"- Dataset: `{dataset_path}`",
            f"- Embedding model (metric): `{embedding_model}`",
            "",
            "## Per-Sample Table",
            "",
            "| id | cosine | citation_coverage | source_f1 | answer_has_citation | required_term_coverage | rewrite_term_coverage | idk_correct | latency_ms | model | notes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]

        def _fmt_optional(value: float | None) -> str:
            return "-" if value is None else f"{value:.3f}"

        for row in rows:
            lines.append(
                f"| {row['sample_id']} | {row['cosine_similarity']:.3f} | {row['citation_coverage']:.3f} | "
                f"{row['source_f1']:.3f} | {row['answer_has_citation']:.3f} | "
                f"{_fmt_optional(row['required_term_coverage'])} | {_fmt_optional(row['rewrite_term_coverage'])} | "
                f"{_fmt_optional(row['idk_correct'])} | {row['latency_ms']:.1f} | {row['model']} | {row['notes']} |"
            )

        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- Mean cosine similarity: **{_fmt_optional(summary.get('mean_cosine_similarity'))}**",
                f"- Mean citation coverage: **{_fmt_optional(summary.get('mean_citation_coverage'))}**",
                f"- Mean source precision: **{_fmt_optional(summary.get('mean_source_precision'))}**",
                f"- Mean source F1: **{_fmt_optional(summary.get('mean_source_f1'))}**",
                f"- Mean answer citation presence: **{_fmt_optional(summary.get('mean_answer_has_citation'))}**",
                f"- Mean required-term coverage: **{_fmt_optional(summary.get('mean_required_term_coverage'))}**",
                f"- Mean rewrite-term coverage: **{_fmt_optional(summary.get('mean_rewrite_term_coverage'))}**",
                f"- Mean IDK correctness: **{_fmt_optional(summary.get('mean_idk_correct'))}**",
                f"- Mean latency (ms): **{_fmt_optional(summary.get('mean_latency_ms'))}**",
            ]
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _build_summary(self, rows: list[dict[str, Any]]) -> dict[str, float | None]:
        return {
            "mean_cosine_similarity": _safe_mean([float(row["cosine_similarity"]) for row in rows]),
            "mean_citation_coverage": _safe_mean([float(row["citation_coverage"]) for row in rows]),
            "mean_source_precision": _safe_mean([float(row["source_precision"]) for row in rows]),
            "mean_source_f1": _safe_mean([float(row["source_f1"]) for row in rows]),
            "mean_answer_has_citation": _safe_mean([float(row["answer_has_citation"]) for row in rows]),
            "mean_required_term_coverage": _safe_mean([row.get("required_term_coverage") for row in rows]),
            "mean_rewrite_term_coverage": _safe_mean([row.get("rewrite_term_coverage") for row in rows]),
            "mean_idk_correct": _safe_mean([row.get("idk_correct") for row in rows]),
            "mean_latency_ms": _safe_mean([float(row["latency_ms"]) for row in rows]),
        }

    def _run_job(self, job_id: str) -> None:
        try:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                options = dict(job["options"])

            run_name = f"backend-eval-{job_id[:8]}"
            dataset_path = Path(options.get("dataset_path") or self.settings.eval_dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

            force_ingest = bool(options.get("force_ingest", False))
            auto_ingest = bool(options.get("auto_ingest", True))
            default_session_id = options.get("default_session_id")
            session_prefix = str(options.get("session_prefix") or "eval")
            embedding_model_name = str(options.get("embedding_model") or self.settings.embedding_model)

            self._update_job(
                job_id,
                status="running",
                started_at=_now_iso(),
                message="Preparing evaluation",
            )

            if force_ingest:
                self.rag_service.ingest(force=True)
            elif auto_ingest and self.rag_service.indexed_chunk_count() == 0:
                self.rag_service.ingest(force=False)

            if self.rag_service.indexed_chunk_count() == 0:
                raise RuntimeError("Vector index is empty. Ingest corpus before evaluation.")

            samples = self._load_dataset(dataset_path)
            total = len(samples)
            self._update_job(job_id, total=total)

            model = SentenceTransformer(embedding_model_name)
            conversation_sessions: dict[str, str] = {}
            rows: list[dict[str, Any]] = []
            predictions: list[dict[str, Any]] = []

            for index, sample in enumerate(samples, start=1):
                sample_id = str(sample["id"])
                question = str(sample["question"]).strip()
                expected_answer = str(sample.get("expected_answer", ""))
                expected_sources = sample.get("expected_sources", [])
                required_terms = sample.get("required_terms", [])
                rewrite_terms = sample.get("expected_rewrite_contains", [])
                expect_idk = sample.get("expect_idk")

                eval_session_id = self._resolve_session_id(
                    sample=sample,
                    default_session_id=default_session_id,
                    session_prefix=session_prefix,
                    job_id=job_id,
                    conversation_sessions=conversation_sessions,
                )

                self._update_job(
                    job_id,
                    message=f"Running sample {index}/{total}",
                    current_sample_id=sample_id,
                    processed=index - 1,
                )

                started = time.perf_counter()
                query_result = self.rag_service.query(question=question, session_id=eval_session_id)
                latency_ms = (time.perf_counter() - started) * 1000.0

                generated_answer = query_result.answer
                rewritten_question = query_result.rewritten_question or question
                predicted_sources = [
                    {
                        "source": source.source,
                        "page": source.page,
                        "chunk_id": source.chunk_id,
                        "score": source.score,
                    }
                    for source in query_result.sources
                ]

                cosine = _cosine_similarity(model, expected_answer, generated_answer)
                recall, precision, source_f1 = _source_recall_precision_f1(expected_sources, predicted_sources)
                answer_has_citation = 1.0 if CITATION_PATTERN.search(generated_answer) else 0.0
                required_term_coverage = _optional_ratio(required_terms, generated_answer)
                rewrite_term_coverage = _optional_ratio(rewrite_terms, rewritten_question or question)

                idk_correct: float | None = None
                if isinstance(expect_idk, bool):
                    normalized_answer = _normalize_text(generated_answer)
                    predicted_idk = normalized_answer == _normalize_text(IDK_ANSWER)
                    idk_correct = 1.0 if predicted_idk == expect_idk else 0.0

                notes: list[str] = []
                if recall < 1.0:
                    notes.append("partial_expected_source_coverage")
                if answer_has_citation == 0.0:
                    notes.append("missing_answer_citation")
                if required_term_coverage is not None and required_term_coverage < 1.0:
                    notes.append("missing_required_terms")
                if rewrite_term_coverage is not None and rewrite_term_coverage < 1.0:
                    notes.append("rewrite_terms_not_fully_matched")
                if idk_correct is not None and idk_correct < 1.0:
                    notes.append("idk_mismatch")
                note_text = "ok" if not notes else ";".join(notes)

                row = {
                    "sample_id": sample_id,
                    "question": question,
                    "eval_session_id": eval_session_id,
                    "rewritten_question": rewritten_question,
                    "model": query_result.model,
                    "latency_ms": latency_ms,
                    "cosine_similarity": cosine,
                    "citation_coverage": recall,
                    "source_precision": precision,
                    "source_f1": source_f1,
                    "answer_has_citation": answer_has_citation,
                    "required_term_coverage": required_term_coverage,
                    "rewrite_term_coverage": rewrite_term_coverage,
                    "idk_correct": idk_correct,
                    "notes": note_text,
                }
                rows.append(row)
                predictions.append(
                    {
                        "sample": sample,
                        "response": {
                            "answer": generated_answer,
                            "model": query_result.model,
                            "session_id": query_result.session_id,
                            "rewritten_question": query_result.rewritten_question,
                            "sources": predicted_sources,
                        },
                        "metrics": {
                            "cosine_similarity": cosine,
                            "citation_coverage": recall,
                            "source_precision": precision,
                            "source_f1": source_f1,
                            "answer_has_citation": answer_has_citation,
                            "required_term_coverage": required_term_coverage,
                            "rewrite_term_coverage": rewrite_term_coverage,
                            "idk_correct": idk_correct,
                            "latency_ms": latency_ms,
                        },
                        "note": note_text,
                    }
                )
                self._update_job(job_id, processed=index, current_sample_id=sample_id)

            summary = self._build_summary(rows)
            artifact_dir = self.settings.eval_output_dir / job_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            csv_path = artifact_dir / "evaluation_results.csv"
            md_path = artifact_dir / "evaluation_results.md"
            jsonl_path = artifact_dir / "evaluation_predictions.jsonl"

            self._write_csv(rows, csv_path)
            self._write_markdown(
                rows=rows,
                output_path=md_path,
                summary=summary,
                run_name=run_name,
                dataset_path=dataset_path,
                embedding_model=embedding_model_name,
            )
            self._write_predictions_jsonl(predictions, jsonl_path)

            artifact_urls = {
                "csv": f"/eval/artifact/{job_id}/csv",
                "markdown": f"/eval/artifact/{job_id}/markdown",
                "jsonl": f"/eval/artifact/{job_id}/jsonl",
            }

            self._update_job(
                job_id,
                status="completed",
                finished_at=_now_iso(),
                message="Evaluation completed",
                summary=summary,
                rows=rows,
                artifacts=artifact_urls,
                artifact_files={
                    "csv": str(csv_path),
                    "markdown": str(md_path),
                    "jsonl": str(jsonl_path),
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Evaluation job failed")
            self._update_job(
                job_id,
                status="failed",
                finished_at=_now_iso(),
                message="Evaluation failed",
                error=str(exc),
            )
        finally:
            self._set_active_job(None)
