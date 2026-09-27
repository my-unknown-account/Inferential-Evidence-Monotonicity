"""Average retriever scores across permutations of the same hint set."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def hint_ids_from_subset(subset: str) -> list[int]:
    width = len(subset)
    return sorted(width - 1 - index for index, value in enumerate(subset) if value == "1")


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    metadata = {}

    for row in rows:
        hint_ids = tuple(hint_ids_from_subset(row["subset"]))
        qid = row.get("qid") or row.get("question_id")
        key = (qid, hint_ids)
        groups[key].append(float(row["score"]))
        metadata[key] = {
            "qid": qid,
            "subset": row["subset"],
            "num_hints": len(hint_ids),
            "hint_ids": list(hint_ids),
            "retriever": row.get("retriever", "unknown"),
        }

    output = []
    for key, scores in sorted(groups.items()):
        row = dict(metadata[key])
        expected_num_permutations = math.factorial(row["num_hints"])
        if len(scores) != expected_num_permutations:
            raise ValueError(
                "Expected "
                f"{expected_num_permutations} permutations for qid={row['qid']} "
                f"subset={row['subset']}, found {len(scores)}."
            )
        row["mean_score"] = sum(scores) / len(scores)
        row["score_std"] = statistics.pstdev(scores)
        row["num_permutations"] = len(scores)
        row["expected_num_permutations"] = expected_num_permutations
        output.append(row)
    return output


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = aggregate(read_jsonl(args.input))
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} subset scores to {args.output}")


if __name__ == "__main__":
    main()
