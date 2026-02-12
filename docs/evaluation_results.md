# Evaluation Results

- Run: `eval-20260212-173158`
- Timestamp (UTC): `2026-02-12T17:38:15.306188+00:00`
- Dataset: `scripts/eval_dataset.json`
- Embedding model (metric): `sentence-transformers/all-MiniLM-L6-v2`

## Backend Config Snapshot

- `ollama_model`: `llama3.2:1b`
- `chunk_size`: `900`
- `chunk_overlap`: `150`
- `top_k`: `4`
- `use_mmr`: `False`
- `query_rewrite_enabled`: `True`
- `memory_enabled`: `True`

## Per-Sample Table

| id | cosine | citation_coverage | source_f1 | answer_has_citation | required_term_coverage | rewrite_term_coverage | idk_correct | latency_ms | model | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| q1 | 0.492 | 0.000 | 0.000 | 1.000 | 0.500 | - | - | 15258.0 | llama3.2:1b | partial_expected_source_coverage;missing_required_terms |
| q2 | 0.585 | 1.000 | 1.000 | 1.000 | 0.000 | - | - | 12195.8 | llama3.2:1b | missing_required_terms |
| q3 | 0.831 | 1.000 | 0.667 | 1.000 | 0.000 | - | - | 34675.6 | llama3.2:1b | missing_required_terms |
| q4 | 0.721 | 1.000 | 0.500 | 1.000 | - | - | - | 18255.6 | llama3.2:1b | ok |
| q5 | 0.158 | 1.000 | 0.667 | 0.000 | - | - | - | 16147.5 | llama3.2:1b | missing_answer_citation |
| q6 | 0.787 | 1.000 | 0.667 | 1.000 | - | - | - | 30608.2 | llama3.2:1b | ok |
| q7 | 0.816 | 1.000 | 1.000 | 1.000 | 0.667 | - | - | 118246.9 | llama3.2:1b | missing_required_terms |
| q8 | 0.663 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | - | 72793.9 | llama3.2:1b | missing_required_terms |
| q9 | 0.242 | 1.000 | 0.000 | 1.000 | - | - | 0.000 | 52357.3 | llama3.2:1b | idk_mismatch |

## Aggregate

- Mean cosine similarity: **0.588**
- Mean citation coverage (source-page recall): **0.889**
- Mean source precision: **0.537**
- Mean source F1: **0.611**
- Mean answer citation presence: **0.889**
- Mean required-term coverage: **0.333**
- Mean rewrite-term coverage: **1.000**
- Mean IDK correctness: **0.000**
- Mean latency (ms): **41170.982**

## Commentary

- `citation_coverage` and `source_f1` reflect grounding quality, not only text similarity.
- `rewrite_term_coverage` is reported only for follow-up cases that define expected rewrite hints.
- `idk_correct` is reported only for samples that explicitly expect abstention.
- Use this report for regression tracking after retrieval/prompt/model changes.