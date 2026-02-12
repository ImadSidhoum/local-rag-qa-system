#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


@dataclass
class EvalRow:
    sample_id: str
    question: str
    cosine_similarity: float
    citation_coverage: float
    answer_has_citation: float
    model: str
    notes: str


def cosine_similarity(model: SentenceTransformer, expected: str, generated: str) -> float:
    vectors = model.encode([expected, generated], normalize_embeddings=True)
    score = float(np.dot(vectors[0], vectors[1]))
    return max(0.0, min(1.0, score))


def citation_coverage(expected_sources: list[dict[str, Any]], predicted_sources: list[dict[str, Any]]) -> float:
    if not expected_sources:
        return 1.0

    expected = {(item["source"], int(item["page"])) for item in expected_sources}
    predicted = {(item["source"], int(item["page"])) for item in predicted_sources}
    matches = len(expected.intersection(predicted))
    return matches / len(expected)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def query_backend(backend_url: str, question: str) -> dict[str, Any]:
    response = requests.post(
        f"{backend_url.rstrip('/')}/query",
        json={"question": question},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def save_csv(rows: list[EvalRow], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_id",
                "question",
                "cosine_similarity",
                "citation_coverage",
                "answer_has_citation",
                "model",
                "notes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.sample_id,
                    row.question,
                    f"{row.cosine_similarity:.4f}",
                    f"{row.citation_coverage:.4f}",
                    f"{row.answer_has_citation:.4f}",
                    row.model,
                    row.notes,
                ]
            )


def save_markdown(rows: list[EvalRow], output_path: Path) -> None:
    if not rows:
        raise ValueError("No evaluation rows to save")

    avg_cosine = sum(row.cosine_similarity for row in rows) / len(rows)
    avg_coverage = sum(row.citation_coverage for row in rows) / len(rows)
    avg_answer_citation = sum(row.answer_has_citation for row in rows) / len(rows)

    lines = [
        "# Evaluation Results",
        "",
        "> Metrics:",
        "> - `cosine_similarity`: semantic similarity between expected and generated answer embeddings.",
        "> - `citation_coverage`: overlap between expected `(source,page)` and returned sources.",
        "> - `answer_has_citation`: whether the generated answer includes explicit citation format.",
        "",
        "| id | cosine_similarity | citation_coverage | answer_has_citation | model | notes |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f"| {row.sample_id} | {row.cosine_similarity:.3f} | {row.citation_coverage:.3f} | "
            f"{row.answer_has_citation:.3f} | {row.model} | {row.notes} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Mean cosine similarity: **{avg_cosine:.3f}**",
            f"- Mean citation coverage: **{avg_coverage:.3f}**",
            f"- Mean answer citation presence: **{avg_answer_citation:.3f}**",
            "",
            "## Commentary",
            "",
            "- Higher cosine with lower citation coverage indicates answer quality without perfect retrieval grounding.",
            "- Low `answer_has_citation` suggests prompt/citation formatting needs tightening.",
            "- Use this table to compare model/chunk/retrieval configuration changes over time.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RAG evaluation")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="scripts/eval_dataset.json")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="docs")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    model = SentenceTransformer(args.embedding_model)

    citation_pattern = re.compile(r"\[source=.*?page=.*?chunk=.*?\]")
    rows: list[EvalRow] = []

    for sample in samples:
        payload = query_backend(args.backend_url, sample["question"])
        generated_answer = payload.get("answer", "")
        model_name = payload.get("model", "unknown")
        predicted_sources = payload.get("sources", [])

        cos = cosine_similarity(model, sample["expected_answer"], generated_answer)
        coverage = citation_coverage(sample.get("expected_sources", []), predicted_sources)
        has_citation = 1.0 if citation_pattern.search(generated_answer) else 0.0

        note = "ok"
        if coverage < 1.0:
            note = "partial source match"
        if has_citation == 0.0:
            note = "missing citation in answer"

        rows.append(
            EvalRow(
                sample_id=sample["id"],
                question=sample["question"],
                cosine_similarity=cos,
                citation_coverage=coverage,
                answer_has_citation=has_citation,
                model=model_name,
                notes=note,
            )
        )

    csv_path = output_dir / "evaluation_results.csv"
    md_path = output_dir / "evaluation_results.md"

    save_csv(rows, csv_path)
    save_markdown(rows, md_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
