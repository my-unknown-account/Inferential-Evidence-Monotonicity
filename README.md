<div align="center">

# *Evidence Monotonicity in Retrieval Scoring*

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)
[![Dataset: QUIT](https://img.shields.io/badge/Dataset-QUIT-orange)](#-experimental-setup)
[![Task: Inferential QA](https://img.shields.io/badge/Task-Inferential%20QA-purple)](#-overview)
[![Submitted: ECIR 2027](https://img.shields.io/badge/Submitted-ECIR%202027-C0392B)](https://www.ecir2027.co.uk/)

**A diagnostic framework for studying whether retrieval scores remain stable or increase when answer-supporting evidence is added while existing evidence is retained.**

</div>

## 🧭 Overview

This repository contains the code and reproducibility scripts for the paper
**"Evidence Monotonicity in Retrieval Scoring"**.

The paper studies a diagnostic question for inferential question answering:

> Do retrieval scores reflect the incremental value of useful evidence?

Inferential QA differs from standard answer-containing retrieval: the answer may
not appear explicitly in the passage, and a reader must infer it by combining
indirect evidence. This repository tests whether retrieval scores behave
consistently as such evidence accumulates. Evidence monotonicity is a diagnostic
of scoring behavior rather than a universal requirement for retrieval functions.
The analysis compares controlled passage scores; it does not measure changes in
full-corpus ranking.

Across **129,600 controlled one-hint evidence additions**, adding valid evidence
lowers the retrieval score in roughly **30-67%** of cases, depending on the
retriever. Many of those score decreases still improve downstream reader
accuracy.

## ✨ Contributions

- Defines **evidence monotonicity** as a score-level diagnostic for inferential
  retrieval.
- Measures monotonicity violations across six retrievers spanning five retrieval
  paradigms using controlled evidence additions from QUIT.
- Studies when violations occur by evidence-accumulation stage and added-hint
  convergence.
- Quantifies whether retrieval-score changes align with downstream reader
  accuracy changes.
- Provides reproducible scripts for generating intermediate JSONL files, result
  tables, logs, and qualitative examples.

## 🧪 Experimental Setup

The experiment uses the QUIT test split:

| Quantity | Value |
| --- | ---: |
| Questions | 1,728 |
| Valid hints per question | 5 |
| Ordered passages per question | 325 |
| Unique evidence sets per question | 31 |
| One-hint transitions per question | 75 |
| Total controlled evidence additions | 129,600 |

Each QUIT question has five answer-supporting evidence items, called hints. A
hint supports inference without simply revealing the answer. Every transition
retains all existing evidence and adds exactly one item, while the question and
target answer remain fixed. For each evidence set, retriever scores are averaged
over all hint permutations to control for changes caused solely by evidence order.

The main metric is **Evidence Monotonicity Violation Rate (EMVR)**:

```text
mean_score(q, S) = average score over all permutations of evidence set S
delta_score = mean_score(q, S + added_item) - mean_score(q, S)
EMVR = # score-decreasing valid evidence additions / # valid evidence additions
```

Ties are not counted as violations. Confidence intervals are computed by
bootstrap resampling questions rather than individual transitions. RQ2 groups
additions by evidence-accumulation stage (`1->2`, `2->3`, `3->4`, `4->5`) and by
added-evidence convergence in five equal-width bins over `[0, 1]`. Convergence
measures how effectively an evidence item narrows the space of plausible answers.

For downstream utility, the analysis uses reader outputs from six generative
readers in QUIT:

- Gemma-3-1B
- Gemma-3-4B
- LLaMA-3.2-1B
- LLaMA-3.1-8B
- Qwen-3-4B
- Qwen-3-8B

Reader utility is the average exact-match correctness across readers and hint
permutations for an evidence set. QUIT excludes questions answerable without
context to limit parametric answerability. RQ3 compares the probability of reader
improvement after score decreases with the probability after score increases,
excluding score ties.

## 🔎 Retrievers

| Retriever | Paradigm | Run script | Default backend/model |
| --- | --- | --- | --- |
| BM25 | Lexical | `scripts/run_bm25.sh` | `rank-bm25` |
| SPLADE | Learned sparse | `scripts/run_splade.sh` | `naver/splade-cocondenser-ensembledistil` |
| DPR | Dense | `scripts/run_dpr.sh` | `facebook/dpr-*-single-nq-base` |
| ColBERT | Late interaction | `scripts/run_colbert.sh` | `colbert-ir/colbertv2.0` through `pylate` |
| BGE | Dense | `scripts/run_bge.sh` | `BAAI/bge-base-en-v1.5` |
| ReasonIR | Reasoning-oriented | `scripts/run_reasonir.sh` | `reasonir/ReasonIR-8B` |

Raw score magnitudes are analyzed within each retriever only; they are not
compared across retrievers because scoring functions use different scales.

## 📊 Main Findings

### 📉 RQ1: Violations Are Widespread

Evidence monotonicity violations occur across every retriever tested.

| Retriever | Paradigm | EMVR (%) | 95% CI |
| --- | --- | ---: | --- |
| BM25 | Lexical | 66.68 | [65.89, 67.48] |
| SPLADE | Learned sparse | 35.04 | [34.40, 35.67] |
| DPR | Dense | 37.64 | [37.00, 38.35] |
| ColBERT | Late interaction | 32.60 | [32.04, 33.19] |
| BGE | Dense | 30.17 | [29.62, 30.69] |
| ReasonIR | Reasoning-oriented | 55.72 | [55.04, 56.39] |

Even BGE, the lowest-violation retriever in this study, assigns a lower score to
nearly one in three valid evidence additions.

### 🧩 RQ2: Later and Informative Evidence Can Still Be Penalized

For neural retrievers, violations generally become more frequent as evidence
accumulates.

| Retriever | EMVR at `1->2` | EMVR at `4->5` |
| --- | ---: | ---: |
| BGE | 14.24 | 39.54 |
| ColBERT | 24.61 | 46.77 |

BM25 behaves differently, decreasing from 75.22% at `1->2` to 56.25% at
`4->5`. ReasonIR rises from 43.43% at `1->2` to above 61% in the final two
stages.

In the highest-convergence bin (`0.8-1.0`), EMVR ranges from 30.81% for BGE to
57.74% for ReasonIR among neural retrievers, and reaches 66.97% for BM25. This
bin has the highest violation rate for each neural retriever under the paper's
binning. Highly informative evidence is therefore not necessarily rewarded by
retrieval scores.

### 🎯 RQ3: Retrieval Scores Only Weakly Track Reader Utility

Reader accuracy often improves even when retrieval score decreases.

| Retriever | P(reader up \| score down) | P(reader up \| score up) | Difference, pp (95% CI) |
| --- | ---: | ---: | --- |
| BM25 | 50.14 | 49.15 | -0.99 [-2.05, 0.20] |
| SPLADE | 49.07 | 50.17 | +1.11 [0.03, 2.20] |
| DPR | 49.19 | 50.14 | +0.95 [-0.09, 1.94] |
| ColBERT | 47.71 | 50.78 | +3.07 [2.04, 4.16] |
| BGE | 47.53 | 50.75 | +3.22 [2.15, 4.31] |
| ReasonIR | 47.98 | 52.04 | +4.06 [3.07, 5.14] |

Probabilities are percentages. The difference is `P(reader up | score up)` minus
`P(reader up | score down)`, in percentage points. The confidence intervals
include zero for BM25 and DPR; the differences are positive for SPLADE, ColBERT,
BGE, and ReasonIR.

For BGE, 18,582 transitions, or 14.34% of all evidence additions, decrease the
retrieval score while increasing reader accuracy. For BM25, this score-down /
reader-up contradiction reaches 33.43% of all additions.

## 🥃 Qualitative Example

One paper example asks:

```text
What global brand withdrew ("rested") its old red-jacketed striding man
104 years after its first appearance in 1908?
```

The answer is **Johnnie Walker**. The initial passage contains two hints:

```text
The brand originated in the Scottish burgh of Kilmarnock in East Ayrshire.
This brand was first established by a grocer in Scotland.
```

None of the six readers answers correctly. After adding a third hint:

```text
It is the world's highest selling Scotch whisky.
```

all six readers answer correctly, but SPLADE, DPR, BGE, and ColBERT assign
lower retrieval scores to the expanded passage.

| Retriever | Initial passage score | Expanded passage score |
| --- | ---: | ---: |
| SPLADE | 8.0483 | 5.5701 |
| DPR | 57.3569 | 54.1458 |
| BGE | 0.4556 | 0.4229 |
| ColBERT | 10.2374 | 9.4045 |

These are the passage scores shown in Figure 1 for QUIT question
`triviahg_train_8512`; the aggregate analyses use permutation-averaged scores.

This illustrates the central mismatch: extra valid evidence can make inference
easier for readers while making the passage look less relevant to retrievers.

## 📁 Repository Layout

```text
outputs/
  figures/     Paper-ready figures.
  tables/      Paper-ready Markdown tables.

scripts/       Shell wrappers for common runs.
src/           Passage construction, scoring, and analysis code.
requirements.txt
run_pipeline.sh
```

The `data/` and `results/` directories are generated by the pipeline and are not
committed to the repository.

## ⚙️ Installation

Create a fresh Python 3.11.9 environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some neural retrievers are large and are best run on a GPU node. ReasonIR
requires a compatible `transformers>=4.47,<5` environment.

## 🚀 Reproduce the Full Pipeline

Run all retrievers and analyses:

```bash
sh run_pipeline.sh
```

The pipeline downloads the QUIT test split, builds passages, scores all
configured retrievers, computes RQ1/RQ2/RQ3 analyses, selects qualitative
examples, and exports paper-ready tables and figures.

Key outputs:

```text
results/run_logs/all_retriever_outputs.md
outputs/tables/table_1_rq1_emvr.md
outputs/tables/table_2_rq3_alignment.md
outputs/figures/figure_2_emvr.png
results/examples/
```

## 🛠️ Run Individual Stages

Download the QUIT test split:

```bash
sh scripts/download_quit_test.sh
```

Build ordered evidence passages:

```bash
python src/build_passages.py \
  --input data/raw/quit_test.json \
  --output data/processed/passages.jsonl
```

Score passages with one retriever:

```bash
python src/score_retriever.py \
  --input data/processed/passages.jsonl \
  --retriever bm25 \
  --output data/scores/bm25.jsonl
```

Average scores over permutations of the same evidence set:

```bash
python src/aggregate_scores.py \
  --input data/scores/bm25.jsonl \
  --output data/processed/bm25_subset_scores.jsonl
```

Compute evidence monotonicity transitions:

```bash
python src/analysis/rq1_monotonicity_violations.py \
  --input data/processed/bm25_subset_scores.jsonl \
  --output results/rq1/bm25/evidence_transitions.jsonl
```

Analyze violation conditions:

```bash
python src/analysis/rq2_violation_conditions.py \
  --transitions results/rq1/bm25/evidence_transitions.jsonl \
  --raw-data data/raw/quit_test.json \
  --output-dir results/rq2/bm25
```

Analyze retriever-reader mismatch:

```bash
python src/analysis/rq3_reader_mismatch.py \
  --transitions results/rq1/bm25/evidence_transitions.jsonl \
  --reader-data data/raw/quit_test.json \
  --output results/rq3/bm25/retriever_reader_transitions.jsonl \
  --mismatches-output results/rq3/bm25/retriever_down_reader_up.jsonl
```

Regenerate exported paper tables and figures:

```bash
python src/export_paper_outputs.py \
  --results-dir results \
  --output-dir outputs
```

## 📦 Data Contracts

`data/processed/passages.jsonl` contains one row per ordered non-empty evidence
subset:

```json
{
  "qid": "triviahg_test_123",
  "question": "Question text",
  "subset": "01110",
  "order": "132",
  "num_hints": 3,
  "hint_ids": [1, 3, 2],
  "passage": "Hint 1 ... Hint 3 ... Hint 2 ...",
  "reader_outputs": {}
}
```

`data/scores/{retriever}.jsonl` stores minimal score rows:

```json
{
  "qid": "triviahg_test_123",
  "subset": "01110",
  "order": "132",
  "retriever": "bm25",
  "score": 12.34
}
```

`data/processed/{retriever}_subset_scores.jsonl` collapses permutations into
unique evidence sets:

```json
{
  "qid": "triviahg_test_123",
  "subset": "01110",
  "num_hints": 3,
  "hint_ids": [1, 2, 3],
  "retriever": "bm25",
  "mean_score": 0.6814,
  "score_std": 0.0132,
  "num_permutations": 6
}
```

`results/rq1/{retriever}/evidence_transitions.jsonl` is the central transition
table:

```json
{
  "qid": "triviahg_test_123",
  "retriever": "bm25",
  "subset_before": "01100",
  "subset_after": "01110",
  "added_hint": 3,
  "num_hints_before": 2,
  "num_hints_after": 3,
  "score_before": 0.6421,
  "score_after": 0.6814,
  "delta": 0.0393,
  "outcome": "increase",
  "violation": false
}
```

## 📝 Citation

If this repository supports your work, please cite the accompanying paper,
**"Evidence Monotonicity in Retrieval Scoring"**, when it becomes available.

## 📄 License

Released under the [MIT License](LICENSE).
