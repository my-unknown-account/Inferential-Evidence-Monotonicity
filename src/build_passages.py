"""Build ordered hint passages from a QUIT-style JSON file."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "questions"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Expected a JSON list, or a JSON object with data/examples/questions.")


def get_hints(record: dict) -> list[str]:
    hints = record.get("hints") or record.get("evidence") or record.get("facts")
    if not isinstance(hints, list) or len(hints) != 5:
        raise ValueError("Each record must contain exactly five hints.")
    output = []
    for hint in hints:
        if isinstance(hint, dict):
            output.append(str(hint.get("hint") or hint.get("text") or hint.get("evidence") or ""))
        else:
            output.append(str(hint))
    return output


def get_question(record: dict) -> str:
    return str(record.get("question") or record.get("query") or "")


def get_question_id(record: dict, fallback: int) -> str:
    return str(record.get("qid") or record.get("question_id") or record.get("id") or fallback)


def get_reader_outputs(record: dict) -> dict:
    for key in ("reader_outputs", "reader_answers", "llm_outputs", "answers_by_reader"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return {}


def subset_mask(hint_ids: tuple[int, ...], num_hints: int) -> str:
    selected = set(hint_ids)
    return "".join("1" if (num_hints - 1 - index) in selected else "0" for index in range(num_hints))


def order_string(hint_ids: tuple[int, ...]) -> str:
    return "".join(str(hint_id) for hint_id in hint_ids)


def build_rows(records: list[dict]) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        qid = get_question_id(record, index)
        question = get_question(record)
        hints = get_hints(record)
        reader_outputs = get_reader_outputs(record)

        for size in range(1, len(hints) + 1):
            for subset in itertools.combinations(range(len(hints)), size):
                for order in itertools.permutations(subset):
                    ordered_hints = [hints[hint_id] for hint_id in order]
                    rows.append(
                        {
                            "qid": qid,
                            "question": question,
                            "subset": subset_mask(subset, len(hints)),
                            "order": order_string(order),
                            "num_hints": size,
                            "hint_ids": list(order),
                            "added_hint_metadata": None,
                            "passage": " ".join(ordered_hints),
                            "reader_outputs": reader_outputs,
                        }
                    )
    return rows


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

    records = load_records(args.input)
    rows = build_rows(records)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} ordered passages to {args.output}")
    if records:
        expected = len(records) * 325
        print(f"Expected rows for five hints/question: {expected}")


if __name__ == "__main__":
    main()
