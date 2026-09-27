"""Select qualitative score-down / reader-up examples for the paper."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit("select_qualitative_example.py requires tqdm. Install it with: pip install tqdm") from error


def read_json_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "questions"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Expected a JSON list, or a JSON object with data/examples/questions.")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def get_qid(record: dict, fallback: int) -> str:
    return str(record.get("qid") or record.get("question_id") or record.get("id") or fallback)


def get_question(record: dict) -> str:
    return str(record.get("question") or record.get("query") or "")


def get_hints(record: dict) -> list[str]:
    hints = record.get("hints") or record.get("evidence") or record.get("facts")
    if not isinstance(hints, list):
        return []
    output = []
    for hint in hints:
        if isinstance(hint, dict):
            output.append(str(hint.get("hint") or hint.get("text") or hint.get("evidence") or ""))
        else:
            output.append(str(hint))
    return output


def hint_ids_from_subset(subset: str) -> list[int]:
    width = len(subset)
    return sorted(width - 1 - index for index, value in enumerate(subset) if value == "1")


def is_correct(prediction: object, answers: list[object]) -> bool:
    return prediction in answers


def reader_accuracy_by_model(subset_outputs: dict, answers: list[object]) -> dict[str, float]:
    totals = defaultdict(int)
    correct = defaultdict(int)
    for reader_outputs in subset_outputs.values():
        if not isinstance(reader_outputs, dict):
            continue
        for reader, prediction in reader_outputs.items():
            totals[reader] += 1
            correct[reader] += int(is_correct(prediction, answers))
    return {reader: correct[reader] / totals[reader] for reader in totals}


def order_key(hint_ids: list[int]) -> str:
    return "".join(str(hint_id) for hint_id in hint_ids)


def pick_reader_outputs(subset_outputs: dict, hint_ids: list[int]) -> tuple[str | None, dict]:
    if not isinstance(subset_outputs, dict) or not subset_outputs:
        return None, {}
    preferred = order_key(hint_ids)
    if preferred in subset_outputs and isinstance(subset_outputs[preferred], dict):
        return preferred, subset_outputs[preferred]
    for order, outputs in sorted(subset_outputs.items()):
        if isinstance(outputs, dict):
            return str(order), outputs
    return None, {}


def build_record_lookup(records: list[dict]) -> dict[str, dict]:
    return {get_qid(record, index): record for index, record in enumerate(records)}


def reader_improvements(record: dict, subset_before: str, subset_after: str) -> list[dict]:
    answers = record.get("answers") or []
    subsets = record.get("subsets") or {}
    before = subsets.get(subset_before) or {}
    after = subsets.get(subset_after) or {}
    before_accuracy = reader_accuracy_by_model(before, answers)
    after_accuracy = reader_accuracy_by_model(after, answers)
    readers = sorted(set(before_accuracy) | set(after_accuracy))

    improvements = []
    for reader in readers:
        before_value = before_accuracy.get(reader)
        after_value = after_accuracy.get(reader)
        if before_value is None or after_value is None:
            continue
        improvements.append(
            {
                "reader": reader,
                "accuracy_before": before_value,
                "accuracy_after": after_value,
                "delta": after_value - before_value,
            }
        )
    return sorted(improvements, key=lambda row: row["delta"], reverse=True)


def prediction_snapshot(record: dict, subset_before: str, subset_after: str) -> dict:
    before_ids = hint_ids_from_subset(subset_before)
    after_ids = hint_ids_from_subset(subset_after)
    subsets = record.get("subsets") or {}
    before_order, before_outputs = pick_reader_outputs(subsets.get(subset_before) or {}, before_ids)
    after_order, after_outputs = pick_reader_outputs(subsets.get(subset_after) or {}, after_ids)
    return {
        "before_order": before_order,
        "after_order": after_order,
        "before_outputs": before_outputs,
        "after_outputs": after_outputs,
    }


def enrich_candidate(row: dict, record: dict) -> dict:
    subset_before = str(row["subset_before"])
    subset_after = str(row["subset_after"])
    hints = get_hints(record)
    before_ids = hint_ids_from_subset(subset_before)
    after_ids = hint_ids_from_subset(subset_after)
    added_hint = int(row["added_hint"])
    improvements = reader_improvements(record, subset_before, subset_after)

    return {
        "qid": str(row["qid"]),
        "retriever": row.get("retriever", "unknown"),
        "question": get_question(record),
        "answers": record.get("answers") or [],
        "subset_before": subset_before,
        "subset_after": subset_after,
        "hint_ids_before": before_ids,
        "hint_ids_after": after_ids,
        "added_hint": added_hint,
        "hints_before": [{"hint_id": hint_id, "text": hints[hint_id]} for hint_id in before_ids],
        "added_hint_text": hints[added_hint],
        "score_before": float(row["score_before"]),
        "score_after": float(row["score_after"]),
        "retriever_delta": float(row.get("retriever_delta", row["delta"])),
        "reader_accuracy_before": row.get("reader_accuracy_before"),
        "reader_accuracy_after": row.get("reader_accuracy_after"),
        "reader_delta": row.get("reader_delta"),
        "reader_improvements": improvements,
        "prediction_snapshot": prediction_snapshot(record, subset_before, subset_after),
    }


def select_candidates(rows: list[dict], records: list[dict], top_k: int) -> list[dict]:
    records_by_qid = build_record_lookup(records)
    candidates = []
    for row in tqdm(rows, desc="Scanning RQ3 transitions", unit="transition"):
        retriever_delta = float(row.get("retriever_delta", row.get("delta", 0.0)))
        reader_delta = row.get("reader_delta")
        if reader_delta is None:
            continue
        reader_delta = float(reader_delta)
        if retriever_delta >= 0 or reader_delta <= 0:
            continue
        qid = str(row["qid"])
        if qid not in records_by_qid:
            continue
        candidate = enrich_candidate(row, records_by_qid[qid])
        candidate["selection_score"] = reader_delta + min(abs(retriever_delta), 10.0) / 10.0
        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda row: (
            float(row["reader_delta"]),
            abs(float(row["retriever_delta"])),
            row["selection_score"],
        ),
        reverse=True,
    )[:top_k]


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_float(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def markdown_for_candidate(candidate: dict, rank: int) -> list[str]:
    lines = [
        f"## Candidate {rank}: {candidate['qid']} ({candidate['retriever']})",
        "",
        f"Question: {candidate['question']}",
        "",
        f"Gold answer(s): {', '.join(str(answer) for answer in candidate['answers'])}",
        "",
        "| Quantity | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Retriever score | {candidate['score_before']:.4f} | "
            f"{candidate['score_after']:.4f} | {candidate['retriever_delta']:.4f} |"
        ),
        (
            f"| Reader accuracy | {format_float(candidate['reader_accuracy_before'])} | "
            f"{format_float(candidate['reader_accuracy_after'])} | "
            f"{format_float(candidate['reader_delta'])} |"
        ),
        "",
        "Evidence before:",
    ]
    for hint in candidate["hints_before"]:
        lines.append(f"- H{hint['hint_id']}: {hint['text']}")
    lines.extend(
        [
            "",
            f"Added hint H{candidate['added_hint']}: {candidate['added_hint_text']}",
            "",
            "Reader improvements:",
            "",
            "| Reader | Before | After | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in candidate["reader_improvements"]:
        if row["delta"] <= 0:
            continue
        lines.append(
            f"| {row['reader']} | {row['accuracy_before']:.4f} | "
            f"{row['accuracy_after']:.4f} | {row['delta']:.4f} |"
        )

    snapshot = candidate["prediction_snapshot"]
    lines.extend(
        [
            "",
            f"Example reader outputs, before order `{snapshot['before_order']}` "
            f"and after order `{snapshot['after_order']}`:",
            "",
            "| Reader | Before prediction | After prediction |",
            "| --- | --- | --- |",
        ]
    )
    readers = sorted(set(snapshot["before_outputs"]) | set(snapshot["after_outputs"]))
    for reader in readers:
        before = snapshot["before_outputs"].get(reader, "")
        after = snapshot["after_outputs"].get(reader, "")
        lines.append(f"| {reader} | {before} | {after} |")
    lines.append("")
    return lines


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qualitative Score-Down / Reader-Up Candidates",
        "",
        "Candidates satisfy `retriever_delta < 0` and `reader_delta > 0`.",
        "",
    ]
    for index, candidate in enumerate(rows, start=1):
        lines.extend(markdown_for_candidate(candidate, index))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_summary(rows: list[dict]) -> None:
    print()
    print("Qualitative example summary")
    print("---------------------------")
    print(f"Candidates written: {len(rows):,}")
    if rows:
        top = rows[0]
        print(f"Top qid:            {top['qid']}")
        print(f"Retriever:          {top['retriever']}")
        print(f"Retriever delta:    {top['retriever_delta']:.4f}")
        print(f"Reader delta:       {float(top['reader_delta']):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq3-transitions", required=True, type=Path)
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    candidates = select_candidates(
        read_jsonl(args.rq3_transitions),
        read_json_records(args.raw_data),
        args.top_k,
    )
    write_json(candidates, args.output_json)
    write_markdown(candidates, args.output_md)
    print_summary(candidates)
    print(f"Wrote qualitative candidates to {args.output_json}")
    print(f"Wrote qualitative Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
