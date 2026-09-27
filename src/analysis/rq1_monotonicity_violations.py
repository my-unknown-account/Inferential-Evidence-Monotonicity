"""RQ1: How often do retrievers violate evidence monotonicity?"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def subset_from_hint_ids(hint_ids: tuple[int, ...], width: int = 5) -> str:
    selected = set(hint_ids)
    return "".join("1" if (width - 1 - index) in selected else "0" for index in range(width))


def delta_outcome(delta: float) -> str:
    if delta < 0:
        return "decrease"
    if delta > 0:
        return "increase"
    return "tie"


def validate_one_hint_addition(
    qid: str,
    hint_set_before: tuple[int, ...],
    hint_set_after: tuple[int, ...],
    added_hint: int,
) -> None:
    before = set(hint_set_before)
    after = set(hint_set_after)
    actually_added = after - before
    removed = before - after
    if removed or actually_added != {added_hint} or len(hint_set_after) != len(hint_set_before) + 1:
        raise ValueError(
            "Invalid transition for "
            f"qid={qid}: before={hint_set_before}, after={hint_set_after}, "
            f"added_hint={added_hint}."
        )


def compute_transitions(rows: list[dict]) -> list[dict]:
    by_question = defaultdict(dict)
    retrievers = {}

    for row in rows:
        qid = row.get("qid") or row.get("question_id")
        hint_ids = tuple(sorted(row["hint_ids"]))
        by_question[qid][hint_ids] = float(row["mean_score"])
        retrievers[qid] = row.get("retriever", "unknown")

    transitions = []
    for qid, scores in sorted(by_question.items()):
        all_hint_ids = sorted({hint_id for hint_set in scores for hint_id in hint_set})
        subset_width = max(all_hint_ids) + 1 if all_hint_ids else 0
        for hint_set_before, score_before in sorted(scores.items()):
            for hint_id in all_hint_ids:
                if hint_id in hint_set_before:
                    continue
                hint_set_after = tuple(sorted((*hint_set_before, hint_id)))
                if hint_set_after not in scores:
                    continue
                validate_one_hint_addition(qid, hint_set_before, hint_set_after, hint_id)
                score_after = scores[hint_set_after]
                delta = score_after - score_before
                outcome = delta_outcome(delta)
                transitions.append(
                    {
                        "qid": qid,
                        "retriever": retrievers[qid],
                        "subset_before": subset_from_hint_ids(hint_set_before, subset_width),
                        "subset_after": subset_from_hint_ids(hint_set_after, subset_width),
                        "added_hint": hint_id,
                        "num_hints_before": len(hint_set_before),
                        "num_hints_after": len(hint_set_after),
                        "score_before": score_before,
                        "score_after": score_after,
                        "delta": delta,
                        "outcome": outcome,
                        "decrease": outcome == "decrease",
                        "tie": outcome == "tie",
                        "increase": outcome == "increase",
                        "violation": outcome == "decrease",
                    }
                )
    return transitions


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qid",
        "retriever",
        "subset_before",
        "subset_after",
        "added_hint",
        "num_hints_before",
        "num_hints_after",
        "score_before",
        "score_after",
        "delta",
        "outcome",
        "decrease",
        "tie",
        "increase",
        "violation",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit("Writing parquet requires pandas and pyarrow/fastparquet.") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_transitions(rows: list[dict], path: Path) -> None:
    if path.suffix == ".jsonl":
        write_jsonl(rows, path)
    elif path.suffix == ".csv":
        write_csv(rows, path)
    elif path.suffix == ".parquet":
        write_parquet(rows, path)
    else:
        raise ValueError("Output path must end with .jsonl, .csv, or .parquet.")


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    decreases = sum(row["decrease"] for row in rows)
    ties = sum(row["tie"] for row in rows)
    increases = sum(row["increase"] for row in rows)
    emvr = decreases / total if total else 0.0
    decrease_rate = decreases / total if total else 0.0
    tie_rate = ties / total if total else 0.0
    increase_rate = increases / total if total else 0.0
    print()
    print("RQ1 summary")
    print("-----------")
    print(f"Transitions:      {total:,}")
    print(f"Score decreases:  {decreases:,} ({decrease_rate:.2%})")
    print(f"Score unchanged:  {ties:,} ({tie_rate:.2%})")
    print(f"Score increases:  {increases:,} ({increase_rate:.2%})")
    print(f"EMVR:             {emvr:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    transitions = compute_transitions(read_jsonl(args.input))
    write_transitions(transitions, args.output)
    print_summary(transitions)
    print(f"Wrote RQ1 evidence transitions to {args.output}")


if __name__ == "__main__":
    main()
