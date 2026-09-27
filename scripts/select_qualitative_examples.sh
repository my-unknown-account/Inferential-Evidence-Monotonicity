#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
RAW_DATA="${PROJECT_ROOT}/data/raw/quit_test.json"
OUTPUT_DIR="${PROJECT_ROOT}/results/examples"
TOP_K="${TOP_K:-10}"

mkdir -p "${OUTPUT_DIR}"

run_one() {
  retriever="$1"
  transitions="${PROJECT_ROOT}/results/rq3/${retriever}/retriever_reader_transitions.jsonl"

  if [ ! -f "${transitions}" ]; then
    printf 'Skipping %s: missing %s\n' "${retriever}" "${transitions}" >&2
    return
  fi

  printf '\033[1;36m==> Selecting qualitative examples for %s\033[0m\n' "${retriever}"
  python "${PROJECT_ROOT}/src/analysis/select_qualitative_example.py" \
    --rq3-transitions "${transitions}" \
    --raw-data "${RAW_DATA}" \
    --output-json "${OUTPUT_DIR}/${retriever}_qualitative_candidates.json" \
    --output-md "${OUTPUT_DIR}/${retriever}_qualitative_candidates.md" \
    --top-k "${TOP_K}"
  printf '\033[1;32m    done: results/examples/%s_qualitative_candidates.md\033[0m\n\n' "${retriever}"
}

run_one bm25
run_one splade
run_one dpr
run_one colbert
run_one bge
run_one reasonir

printf '\033[1;32mQualitative example selection complete.\033[0m\n'
