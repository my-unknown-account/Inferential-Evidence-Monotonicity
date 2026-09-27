"""BM25 scoring backend using rank_bm25."""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
BM25_SCOPE = os.environ.get("BM25_SCOPE", "question")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def score(rows: list[dict]) -> list[float]:
    try:
        from rank_bm25 import BM25Okapi
        from tqdm import tqdm
    except ImportError as error:
        raise SystemExit(
            "BM25 scoring requires rank-bm25 and tqdm. Install them with: "
            "pip install rank-bm25 tqdm"
        ) from error

    if BM25_SCOPE != "question":
        raise ValueError("Only BM25_SCOPE=question is supported.")

    groups = defaultdict(list)
    for row_index, row in enumerate(rows):
        groups[row["qid"]].append(row_index)

    print(
        f"BM25: scoring {len(rows):,} passages grouped by {len(groups):,} questions",
        file=sys.stderr,
        flush=True,
    )

    scores = [0.0] * len(rows)
    for _qid, indices in tqdm(
        groups.items(),
        desc="BM25 questions",
        unit="question",
        file=sys.stderr,
    ):
        group_rows = [rows[index] for index in indices]
        corpus = [tokenize(row["passage"]) for row in group_rows]
        query_tokens = tokenize(group_rows[0]["question"])
        bm25 = BM25Okapi(corpus)
        group_scores = bm25.get_scores(query_tokens)
        for row_index, row_score in zip(indices, group_scores):
            scores[row_index] = float(row_score)
    return scores
