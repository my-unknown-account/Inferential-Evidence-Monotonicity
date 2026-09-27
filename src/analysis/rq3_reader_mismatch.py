"""RQ3: Do retriever-score drops conflict with reader-answer gains?"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_json_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "questions"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Expected a JSON list, or a JSON object with data/examples/questions.")


def read_transitions(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as input_file:
            return [json.loads(line) for line in input_file if line.strip()]
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as error:
            raise SystemExit("Reading parquet requires pandas and pyarrow/fastparquet.") from error
        return pd.read_parquet(path).to_dict(orient="records")
    with path.open(encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def is_correct(prediction: object, answers: list[object]) -> bool:
    return prediction in answers


def get_qid(record: dict, fallback: int) -> str:
    return str(record.get("qid") or record.get("question_id") or record.get("id") or fallback)


def subset_reader_accuracy(subset_outputs: dict, answers: list[object]) -> float | None:
    """Average exact-match correctness across all orders and reader models."""
    total = 0
    correct = 0
    for reader_outputs in subset_outputs.values():
        if not isinstance(reader_outputs, dict):
            continue
        for prediction in reader_outputs.values():
            total += 1
            correct += int(is_correct(prediction, answers))
    if total == 0:
        return None
    return correct / total


def build_reader_accuracy_lookup(records: list[dict]) -> dict[tuple[str, str], float | None]:
    lookup = {}
    for index, record in enumerate(records):
        qid = get_qid(record, index)
        answers = record.get("answers") or []
        subsets = record.get("subsets") or {}
        for subset, subset_outputs in subsets.items():
            if isinstance(subset_outputs, dict):
                lookup[(qid, subset)] = subset_reader_accuracy(subset_outputs, answers)
    return lookup


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).lower() in {"true", "1", "yes"}


def delta_outcome(delta: float | None) -> str | None:
    if delta is None:
        return None
    if delta < 0:
        return "decrease"
    if delta > 0:
        return "increase"
    return "tie"


def enrich_with_reader_deltas(
    transitions: list[dict],
    reader_accuracy: dict[tuple[str, str], float | None],
) -> list[dict]:
    enriched = []
    for row in transitions:
        output = dict(row)
        qid = str(output["qid"])
        subset_before = str(output["subset_before"])
        subset_after = str(output["subset_after"])
        before_accuracy = reader_accuracy.get((qid, subset_before))
        after_accuracy = reader_accuracy.get((qid, subset_after))

        output["retriever_delta"] = float(output["delta"])
        output["retriever_outcome"] = delta_outcome(output["retriever_delta"])
        output["retriever_violation"] = to_bool(output["violation"])
        output["reader_accuracy_before"] = before_accuracy
        output["reader_accuracy_after"] = after_accuracy
        output["reader_delta"] = (
            after_accuracy - before_accuracy
            if before_accuracy is not None and after_accuracy is not None
            else None
        )
        output["reader_outcome"] = delta_outcome(output["reader_delta"])
        output["reader_down"] = output["reader_outcome"] == "decrease"
        output["reader_tie"] = output["reader_outcome"] == "tie"
        output["reader_up"] = output["reader_outcome"] == "increase"
        output["retriever_down_reader_up"] = (
            output["retriever_delta"] < 0
            and output["reader_delta"] is not None
            and output["reader_delta"] > 0
        )
        assert output["retriever_down_reader_up"] == (
            output["retriever_delta"] < 0 and output["reader_up"]
        )
        enriched.append(output)
    return enriched


def reader_scored_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["reader_delta"] is not None]


def build_outcome_matrix(rows: list[dict]) -> list[dict]:
    labels = ["decrease", "tie", "increase"]
    counts = {
        retriever_outcome: {reader_outcome: 0 for reader_outcome in labels}
        for retriever_outcome in labels
    }
    for row in reader_scored_rows(rows):
        counts[row["retriever_outcome"]][row["reader_outcome"]] += 1

    matrix = []
    for retriever_outcome in labels:
        row_counts = counts[retriever_outcome]
        total = sum(row_counts.values())
        matrix.append(
            {
                "retriever_outcome": retriever_outcome,
                "reader_down": row_counts["decrease"],
                "reader_tie": row_counts["tie"],
                "reader_up": row_counts["increase"],
                "total": total,
                "p_reader_up": row_counts["increase"] / total if total else 0.0,
            }
        )
    return matrix


def build_reader_delta_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in reader_scored_rows(rows):
        groups[row["retriever_outcome"]].append(float(row["reader_delta"]))

    output = []
    for retriever_outcome in ["decrease", "tie", "increase"]:
        values = groups[retriever_outcome]
        reader_up = sum(value > 0 for value in values)
        output.append(
            {
                "retriever_outcome": retriever_outcome,
                "n": len(values),
                "reader_up": reader_up,
                "p_reader_up": reader_up / len(values) if values else 0.0,
                "mean_reader_delta": statistics.fmean(values) if values else 0.0,
                "median_reader_delta": statistics.median(values) if values else 0.0,
            }
        )
    return output


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_rate(value: float) -> str:
    return f"{100 * value:6.2f}%"


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    matrix = build_outcome_matrix(rows)
    reader_delta_summary = build_reader_delta_summary(rows)
    mismatches = sum(row["retriever_down_reader_up"] for row in rows)
    rate = mismatches / total if total else 0.0
    reader_available = sum(row["reader_delta"] is not None for row in rows)
    retriever_down = sum(row["retriever_delta"] < 0 for row in rows)
    reader_up = sum(row["reader_up"] for row in rows)
    retriever_down_rate = retriever_down / total if total else 0.0
    reader_up_rate = reader_up / total if total else 0.0
    print()
    print("RQ3 summary")
    print("-----------")
    print(f"Transitions:                    {total:,}")
    print(f"Transitions with reader scores: {reader_available:,}")
    print(f"Retriever-down cases:           {retriever_down:,} ({retriever_down_rate:.2%})")
    print(f"Reader-up cases:                {reader_up:,} ({reader_up_rate:.2%})")
    print(f"Retriever-down / reader-up:     {mismatches:,} ({rate:.2%})")
    print()
    print("Retriever x reader outcome")
    print(f"{'':<14} {'Reader down':>12} {'Reader tie':>12} {'Reader up':>12}")
    for row in matrix:
        print(
            f"{'Retriever ' + row['retriever_outcome']:<14} "
            f"{row['reader_down']:>12,d} "
            f"{row['reader_tie']:>12,d} "
            f"{row['reader_up']:>12,d}"
        )
    print()
    print("Reader-up probability and reader-delta by retriever outcome")
    print(f"{'Retriever':<12} {'N':>9} {'P(reader up)':>14} {'mean delta':>12} {'median delta':>14}")
    for row in reader_delta_summary:
        print(
            f"{row['retriever_outcome']:<12} "
            f"{row['n']:>9,d} "
            f"{format_rate(row['p_reader_up']):>14} "
            f"{row['mean_reader_delta']:>12.4f} "
            f"{row['median_reader_delta']:>14.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", required=True, type=Path)
    parser.add_argument("--reader-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mismatches-output", required=True, type=Path)
    args = parser.parse_args()

    transitions = read_transitions(args.transitions)
    reader_accuracy = build_reader_accuracy_lookup(read_json_records(args.reader_data))
    enriched = enrich_with_reader_deltas(transitions, reader_accuracy)
    mismatches = [row for row in enriched if row["retriever_down_reader_up"]]
    matrix = build_outcome_matrix(enriched)
    reader_delta_summary = build_reader_delta_summary(enriched)
    output_dir = args.output.parent

    write_jsonl(enriched, args.output)
    write_jsonl(mismatches, args.mismatches_output)
    write_csv(matrix, output_dir / "retriever_reader_outcome_matrix.csv")
    write_csv(reader_delta_summary, output_dir / "reader_delta_by_retriever_outcome.csv")
    print_summary(enriched)
    print(f"Wrote RQ3 enriched transitions to {args.output}")
    print(f"Wrote strongest mismatch cases to {args.mismatches_output}")
    print(f"Wrote retriever-reader outcome matrix to {output_dir / 'retriever_reader_outcome_matrix.csv'}")
    print(f"Wrote reader-delta summary to {output_dir / 'reader_delta_by_retriever_outcome.csv'}")


if __name__ == "__main__":
    main()
