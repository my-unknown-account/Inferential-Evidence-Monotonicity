#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_PATH="${PROJECT_ROOT}/data/raw/quit_test.json"

mkdir -p "${PROJECT_ROOT}/data/raw"

curl -L \
  "https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/test.json?download=true" \
  -o "${OUTPUT_PATH}"

python - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
count = len(data) if isinstance(data, list) else "unknown"
print(f"Downloaded {path} with {count} records.")
PY
