"""Optional RQ3 robustness check by individual reader model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit("rq3_reader_robustness.py requires tqdm. Install it with: pip install tqdm") from error

try:
    from rq3_reader_mismatch import delta_outcome, read_json_records, read_transitions
except ImportError:
    from src.analysis.rq3_reader_mismatch import delta_outcome, read_json_records, read_transitions


def get_qid(record: dict, fallback: int) -> str:
    return str(record.get("qid") or record.get("question_id") or record.get("id") or fallback)


def is_correct(prediction: object, answers: list[object]) -> bool:
    return prediction in answers


def subset_reader_accuracy_by_model(subset_outputs: dict, answers: list[object]) -> dict[str, float]:
    totals = defaultdict(int)
    correct = defaultdict(int)
    for reader_outputs in subset_outputs.values():
        if not isinstance(reader_outputs, dict):
            continue
        for reader, prediction in reader_outputs.items():
            totals[reader] += 1
            correct[reader] += int(is_correct(prediction, answers))
    return {reader: correct[reader] / totals[reader] for reader in totals}


def build_lookup(records: list[dict]) -> dict[tuple[str, str, str], float]:
    lookup = {}
    for index, record in enumerate(tqdm(records, desc="Indexing reader outputs", unit="question")):
        qid = get_qid(record, index)
        answers = record.get("answers") or []
        subsets = record.get("subsets") or {}
        for subset, subset_outputs in subsets.items():
            if not isinstance(subset_outputs, dict):
                continue
            for reader, accuracy in subset_reader_accuracy_by_model(subset_outputs, answers).items():
                lookup[(qid, subset, reader)] = accuracy
    return lookup


def summarize(transitions: list[dict], lookup: dict[tuple[str, str, str], float]) -> list[dict]:
    readers = sorted({reader for _, _, reader in lookup})
    rows = []
    for reader in tqdm(readers, desc="Summarizing readers", unit="reader"):
        total = 0
        score_down = 0
        score_up = 0
        reader_up_given_down = 0
        reader_up_given_up = 0
        mismatch = 0
        for transition in transitions:
            qid = str(transition["qid"])
            before = lookup.get((qid, str(transition["subset_before"]), reader))
            after = lookup.get((qid, str(transition["subset_after"]), reader))
            if before is None or after is None:
                continue
            total += 1
            retriever_delta = float(transition["delta"])
            reader_delta = after - before
            reader_up = reader_delta > 0
            if retriever_delta < 0:
                score_down += 1
                reader_up_given_down += int(reader_up)
                mismatch += int(reader_up)
            elif retriever_delta > 0:
                score_up += 1
                reader_up_given_up += int(reader_up)
        p_down = reader_up_given_down / score_down if score_down else 0.0
        p_up = reader_up_given_up / score_up if score_up else 0.0
        rows.append(
            {
                "reader": reader,
                "transitions": total,
                "score_down": score_down,
                "score_up": score_up,
                "retriever_down_reader_up": mismatch,
                "p_reader_up_given_score_down": p_down,
                "p_reader_up_given_score_up": p_up,
                "alignment_difference": p_up - p_down,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print()
    print("RQ3 per-reader robustness summary")
    print("---------------------------------")
    print(f"{'reader':<16} {'P(up|down)':>12} {'P(up|up)':>10} {'diff':>10}")
    for row in rows:
        print(
            f"{row['reader']:<16} "
            f"{row['p_reader_up_given_score_down']:>12.2%} "
            f"{row['p_reader_up_given_score_up']:>10.2%} "
            f"{row['alignment_difference']:>10.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", required=True, type=Path)
    parser.add_argument("--reader-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = summarize(read_transitions(args.transitions), build_lookup(read_json_records(args.reader_data)))
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"Wrote per-reader robustness summary to {args.output}")


if __name__ == "__main__":
    main()
