"""RQ2: When do evidence monotonicity violations occur?

RQ2 reads the RQ1 evidence-transition table and enriches each edge with
metadata about the newly added hint.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def get_qid(record: dict, fallback: int) -> str:
    return str(record.get("qid") or record.get("question_id") or record.get("id") or fallback)


def build_convergence_lookup(records: list[dict]) -> dict[tuple[str, int], float | None]:
    lookup = {}
    for index, record in enumerate(records):
        qid = get_qid(record, index)
        hints = record.get("hints") or []
        for hint_id, hint in enumerate(hints):
            convergence = hint.get("convergence") if isinstance(hint, dict) else None
            lookup[(qid, hint_id)] = float(convergence) if convergence is not None else None
    return lookup


def is_violation(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).lower() in {"true", "1", "yes"}


def outcome_from_row(row: dict) -> str:
    outcome = row.get("outcome")
    if outcome in {"decrease", "tie", "increase"}:
        return outcome
    delta = float(row["delta"])
    if delta < 0:
        return "decrease"
    if delta > 0:
        return "increase"
    return "tie"


def transition_label(row: dict) -> str:
    return f"{row['num_hints_before']}->{row['num_hints_after']}"


def enrich_transitions(rows: list[dict], convergence_lookup: dict[tuple[str, int], float | None]) -> list[dict]:
    enriched = []
    for row in rows:
        output = dict(row)
        qid = str(output["qid"])
        added_hint = int(output["added_hint"])
        output["added_hint"] = added_hint
        output["added_hint_convergence"] = convergence_lookup.get((qid, added_hint))
        output["transition"] = transition_label(output)
        output["violation"] = is_violation(output["violation"])
        output["outcome"] = outcome_from_row(output)
        output["decrease"] = output["outcome"] == "decrease"
        output["tie"] = output["outcome"] == "tie"
        output["increase"] = output["outcome"] == "increase"
        assert output["violation"] == output["decrease"]
        enriched.append(output)
    return enriched


def empty_counts() -> dict:
    return {"comparisons": 0, "decreases": 0, "ties": 0, "increases": 0}


def add_counts(counts: dict, row: dict) -> None:
    counts["comparisons"] += 1
    counts["decreases"] += int(row["decrease"])
    counts["ties"] += int(row["tie"])
    counts["increases"] += int(row["increase"])


def counts_to_row(label_name: str, label_value: str, counts: dict) -> dict:
    comparisons = counts["comparisons"]
    decreases = counts["decreases"]
    ties = counts["ties"]
    increases = counts["increases"]
    assert decreases + ties + increases == comparisons
    return {
        label_name: label_value,
        "comparisons": comparisons,
        "decreases": decreases,
        "ties": ties,
        "increases": increases,
        "violations": decreases,
        "p_delta_lt_0": decreases / comparisons if comparisons else 0.0,
        "p_delta_eq_0": ties / comparisons if comparisons else 0.0,
        "p_delta_gt_0": increases / comparisons if comparisons else 0.0,
    }


def summarize_overall(rows: list[dict]) -> list[dict]:
    counts = empty_counts()
    for row in rows:
        add_counts(counts, row)
    return [counts_to_row("group", "overall", counts)]


def summarize_by_transition(rows: list[dict]) -> list[dict]:
    groups = defaultdict(empty_counts)
    for row in rows:
        add_counts(groups[row["transition"]], row)

    output = []
    for transition, counts in sorted(groups.items()):
        output.append(counts_to_row("transition", transition, counts))
    return output


def convergence_bin(value: float | None, num_bins: int) -> str:
    if value is None:
        return "missing"
    if value >= 1:
        return f"{(num_bins - 1) / num_bins:.1f}-1.0"
    lower_index = int(value * num_bins)
    lower = lower_index / num_bins
    upper = (lower_index + 1) / num_bins
    return f"{lower:.1f}-{upper:.1f}"


def summarize_by_convergence(rows: list[dict], num_bins: int) -> list[dict]:
    groups = defaultdict(empty_counts)
    for row in rows:
        bucket = convergence_bin(row["added_hint_convergence"], num_bins)
        add_counts(groups[bucket], row)

    output = []
    for bucket, counts in sorted(groups.items()):
        output.append(counts_to_row("added_hint_convergence_bin", bucket, counts))
    return output


def validate_rq1_rq2_consistency(transitions: list[dict], enriched: list[dict]) -> None:
    rq1_total = len(transitions)
    rq2_total = len(enriched)
    rq1_violations = sum(is_violation(row["violation"]) for row in transitions)
    rq2_violations = sum(row["violation"] for row in enriched)
    rq2_decreases = sum(row["decrease"] for row in enriched)
    rq2_ties = sum(row["tie"] for row in enriched)
    rq2_increases = sum(row["increase"] for row in enriched)

    assert rq2_total == rq1_total, f"RQ2 total {rq2_total} != RQ1 total {rq1_total}"
    assert rq2_violations == rq1_violations, (
        f"RQ2 violations {rq2_violations} != RQ1 violations {rq1_violations}"
    )
    assert rq2_violations == rq2_decreases
    assert rq2_decreases + rq2_ties + rq2_increases == rq2_total


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


def print_summary_table(title: str, label_name: str, rows: list[dict]) -> None:
    print(title)
    print(f"{label_name:<18} {'N':>9} {'down':>9} {'tie':>9} {'up':>9}")
    for row in rows:
        print(
            f"{str(row[label_name]):<18} "
            f"{row['comparisons']:>9,d} "
            f"{format_rate(row['p_delta_lt_0']):>9} "
            f"{format_rate(row['p_delta_eq_0']):>9} "
            f"{format_rate(row['p_delta_gt_0']):>9}"
        )


def print_rq2_summary(overall: list[dict], by_transition: list[dict], by_convergence: list[dict]) -> None:
    print()
    print("RQ2 summary")
    print("-----------")
    print_summary_table("Overall outcomes", "group", overall)
    print()
    print_summary_table("By evidence transition", "transition", by_transition)
    print()
    print_summary_table("By added-hint convergence", "added_hint_convergence_bin", by_convergence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", required=True, type=Path)
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-convergence-bins", type=int, default=5)
    args = parser.parse_args()

    transitions = read_transitions(args.transitions)
    convergence_lookup = build_convergence_lookup(read_json_records(args.raw_data))
    enriched = enrich_transitions(transitions, convergence_lookup)
    validate_rq1_rq2_consistency(transitions, enriched)
    overall = summarize_overall(enriched)
    by_transition = summarize_by_transition(enriched)
    by_convergence = summarize_by_convergence(enriched, args.num_convergence_bins)

    write_jsonl(enriched, args.output_dir / "evidence_transitions_with_hint_metadata.jsonl")
    write_csv(overall, args.output_dir / "outcome_summary.csv")
    write_csv(by_transition, args.output_dir / "p_delta_lt_0_by_transition.csv")
    write_csv(by_convergence, args.output_dir / "p_delta_lt_0_by_added_hint_convergence.csv")

    print_rq2_summary(overall, by_transition, by_convergence)
    print(f"Wrote RQ2 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
