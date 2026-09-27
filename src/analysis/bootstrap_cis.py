"""Question-level bootstrap confidence intervals for RQ1/RQ2/RQ3 metrics."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit("bootstrap_cis.py requires tqdm. Install it with: pip install tqdm") from error


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).lower() in {"true", "1", "yes"}


def transition_label(row: dict) -> str:
    return f"{row['num_hints_before']}->{row['num_hints_after']}"


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def group_by_qid(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["qid"])].append(row)
    return dict(groups)


def rq_metrics(rows: list[dict]) -> dict[str, float]:
    metrics = {}
    metrics["rq1_emvr_overall"] = rate(sum(to_bool(row["violation"]) for row in rows), len(rows))

    transition_groups = defaultdict(list)
    for row in rows:
        transition_groups[transition_label(row)].append(row)
    for transition in ["1->2", "2->3", "3->4", "4->5"]:
        group = transition_groups[transition]
        metrics[f"rq2_emvr_{transition}"] = rate(sum(to_bool(row["violation"]) for row in group), len(group))
    return metrics


def rq3_metrics(rows: list[dict]) -> dict[str, float]:
    scored = [row for row in rows if row.get("reader_delta") is not None]
    score_up = [row for row in scored if float(row["retriever_delta"]) > 0]
    score_down = [row for row in scored if float(row["retriever_delta"]) < 0]
    p_reader_up_given_score_up = rate(sum(float(row["reader_delta"]) > 0 for row in score_up), len(score_up))
    p_reader_up_given_score_down = rate(sum(float(row["reader_delta"]) > 0 for row in score_down), len(score_down))
    return {
        "rq3_p_reader_up_given_score_up": p_reader_up_given_score_up,
        "rq3_p_reader_up_given_score_down": p_reader_up_given_score_down,
        "rq3_alignment_difference": p_reader_up_given_score_up - p_reader_up_given_score_down,
    }


def bootstrap(
    rq1_rows: list[dict],
    rq3_rows: list[dict] | None,
    num_samples: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rq1_by_qid = group_by_qid(rq1_rows)
    rq3_by_qid = group_by_qid(rq3_rows or [])
    qids = sorted(rq1_by_qid)
    observed = rq_metrics(rq1_rows)
    if rq3_rows is not None:
        observed.update(rq3_metrics(rq3_rows))

    samples = {metric: [] for metric in observed}
    for _ in tqdm(range(num_samples), desc="Bootstrap samples", unit="sample"):
        sampled_qids = [rng.choice(qids) for _ in qids]
        sampled_rq1 = [row for qid in sampled_qids for row in rq1_by_qid[qid]]
        values = rq_metrics(sampled_rq1)
        if rq3_rows is not None:
            sampled_rq3 = [row for qid in sampled_qids for row in rq3_by_qid.get(qid, [])]
            values.update(rq3_metrics(sampled_rq3))
        for metric, value in values.items():
            samples[metric].append(value)

    output = []
    for metric, estimate in observed.items():
        values = samples[metric]
        output.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_low": percentile(values, 0.025),
                "ci_high": percentile(values, 0.975),
                "bootstrap_samples": num_samples,
                "num_questions": len(qids),
            }
        )
    return output


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print()
    print("Bootstrap CI summary")
    print("--------------------")
    print(f"{'metric':<38} {'estimate':>10} {'95% CI':>24}")
    for row in rows:
        print(
            f"{row['metric']:<38} "
            f"{row['estimate']:>10.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq1-transitions", required=True, type=Path)
    parser.add_argument("--rq3-transitions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rq1_rows = read_jsonl(args.rq1_transitions)
    rq3_rows = read_jsonl(args.rq3_transitions) if args.rq3_transitions else None
    rows = bootstrap(rq1_rows, rq3_rows, args.num_samples, args.seed)
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"Wrote bootstrap CIs to {args.output}")


if __name__ == "__main__":
    main()
