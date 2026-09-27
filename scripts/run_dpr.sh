#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

stage() {
  printf '\033[1;36m==> [%s/8] %s\033[0m\n' "$1" "$2"
}

done_stage() {
  printf '\033[1;32m    done: %s\033[0m\n\n' "$1"
}

stage 1 "Build ordered passages"
python "${PROJECT_ROOT}/src/build_passages.py" \
  --input "${PROJECT_ROOT}/data/raw/quit_test.json" \
  --output "${PROJECT_ROOT}/data/processed/passages.jsonl"
done_stage "data/processed/passages.jsonl"

stage 2 "Score passages with DPR"
python "${PROJECT_ROOT}/src/score_retriever.py" \
  --input "${PROJECT_ROOT}/data/processed/passages.jsonl" \
  --retriever dpr \
  --output "${PROJECT_ROOT}/data/scores/dpr.jsonl"
done_stage "data/scores/dpr.jsonl"

stage 3 "Aggregate permutation scores"
python "${PROJECT_ROOT}/src/aggregate_scores.py" \
  --input "${PROJECT_ROOT}/data/scores/dpr.jsonl" \
  --output "${PROJECT_ROOT}/data/processed/dpr_subset_scores.jsonl"
done_stage "data/processed/dpr_subset_scores.jsonl"

stage 4 "RQ1 evidence transitions"
python "${PROJECT_ROOT}/src/analysis/rq1_monotonicity_violations.py" \
  --input "${PROJECT_ROOT}/data/processed/dpr_subset_scores.jsonl" \
  --output "${PROJECT_ROOT}/results/rq1/dpr/evidence_transitions.jsonl"
done_stage "results/rq1/dpr/evidence_transitions.jsonl"

stage 5 "RQ2 violation conditions"
python "${PROJECT_ROOT}/src/analysis/rq2_violation_conditions.py" \
  --transitions "${PROJECT_ROOT}/results/rq1/dpr/evidence_transitions.jsonl" \
  --raw-data "${PROJECT_ROOT}/data/raw/quit_test.json" \
  --output-dir "${PROJECT_ROOT}/results/rq2/dpr"
done_stage "results/rq2/dpr"

stage 6 "RQ3 retriever-reader mismatch"
python "${PROJECT_ROOT}/src/analysis/rq3_reader_mismatch.py" \
  --transitions "${PROJECT_ROOT}/results/rq1/dpr/evidence_transitions.jsonl" \
  --reader-data "${PROJECT_ROOT}/data/raw/quit_test.json" \
  --output "${PROJECT_ROOT}/results/rq3/dpr/retriever_reader_transitions.jsonl" \
  --mismatches-output "${PROJECT_ROOT}/results/rq3/dpr/retriever_down_reader_up.jsonl"
done_stage "results/rq3/dpr"

stage 7 "Bootstrap confidence intervals"
python "${PROJECT_ROOT}/src/analysis/bootstrap_cis.py" \
  --rq1-transitions "${PROJECT_ROOT}/results/rq1/dpr/evidence_transitions.jsonl" \
  --rq3-transitions "${PROJECT_ROOT}/results/rq3/dpr/retriever_reader_transitions.jsonl" \
  --output "${PROJECT_ROOT}/results/bootstrap/dpr/bootstrap_cis.csv"
done_stage "results/bootstrap/dpr/bootstrap_cis.csv"

stage 8 "RQ3 per-reader robustness"
python "${PROJECT_ROOT}/src/analysis/rq3_reader_robustness.py" \
  --transitions "${PROJECT_ROOT}/results/rq1/dpr/evidence_transitions.jsonl" \
  --reader-data "${PROJECT_ROOT}/data/raw/quit_test.json" \
  --output "${PROJECT_ROOT}/results/rq3/dpr/per_reader_robustness.csv"
done_stage "results/rq3/dpr/per_reader_robustness.csv"

printf '\033[1;32mDPR pipeline complete.\033[0m\n'
