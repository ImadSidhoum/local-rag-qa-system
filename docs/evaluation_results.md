# Evaluation Results

- Run: `backend-eval-01daf70d`
- Timestamp (UTC): `2026-02-13T08:07:23.527364+00:00`
- Dataset: `/app/scripts/eval_dataset.json`
- Embedding model (metric): `google/embeddinggemma-300m`

## Per-Sample Table

| id | cosine | citation_coverage | source_f1 | answer_has_citation | required_term_coverage | rewrite_term_coverage | idk_correct | latency_ms | model | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| q1 | 0.591 | 1.000 | 0.667 | 1.000 | 0.500 | - | - | 15716.0 | llama3.2:1b | missing_required_terms |
| q2 | 0.635 | 1.000 | 1.000 | 1.000 | 0.000 | - | - | 9284.2 | llama3.2:1b | missing_required_terms |
| q3 | 0.823 | 1.000 | 0.500 | 1.000 | 0.000 | - | - | 13513.4 | llama3.2:1b | missing_required_terms |
| q4 | 0.588 | 1.000 | 0.500 | 1.000 | - | - | - | 12573.5 | llama3.2:1b | ok |
| q5 | 0.345 | 1.000 | 0.500 | 0.000 | - | - | - | 7229.1 | llama3.2:1b | missing_answer_citation |
| q6 | 0.719 | 1.000 | 0.667 | 1.000 | - | - | - | 12503.5 | llama3.2:1b | ok |
| q7 | 0.634 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | 13443.5 | llama3.2:1b | ok |
| q8 | 0.199 | 1.000 | 0.667 | 1.000 | 0.000 | 0.000 | - | 18634.1 | llama3.2:1b | missing_required_terms;rewrite_terms_not_fully_matched |
| q9 | 1.000 | 1.000 | 0.000 | 0.000 | - | - | 1.000 | 8036.7 | llama3.2:1b | missing_answer_citation |

## Aggregate

- Mean cosine similarity: **0.615**
- Mean citation coverage: **1.000**
- Mean source precision: **0.500**
- Mean source F1: **0.611**
- Mean answer citation presence: **0.778**
- Mean required-term coverage: **0.300**
- Mean rewrite-term coverage: **0.000**
- Mean IDK correctness: **1.000**
- Mean latency (ms): **12325.979**