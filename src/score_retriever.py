"""Score ordered passages with a simple retriever interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from retrievers import get_retriever
except ImportError:
    from src.retrievers import get_retriever


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def score_rows(rows: list[dict], retriever: str) -> list[dict]:
    score_fn = get_retriever(retriever)
    scores = score_fn(rows)
    if len(scores) != len(rows):
        raise ValueError(f"{retriever} returned {len(scores)} scores for {len(rows)} rows.")
    return [
        {
            "qid": row["qid"],
            "subset": row["subset"],
            "order": row["order"],
            "retriever": retriever,
            "score": score,
        }
        for row, score in zip(rows, scores)
    ]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    start_time = time.time()
    print(f"Loading passages: {args.input}", file=sys.stderr, flush=True)
    rows = read_jsonl(args.input)
    print(f"Loaded {len(rows):,} passages", file=sys.stderr, flush=True)
    print(f"Scoring with retriever: {args.retriever}", file=sys.stderr, flush=True)
    scored = score_rows(rows, args.retriever)
    print(f"Writing scores: {args.output}", file=sys.stderr, flush=True)
    write_jsonl(scored, args.output)
    elapsed = time.time() - start_time
    print(f"Wrote {len(scored):,} {args.retriever} scores to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
