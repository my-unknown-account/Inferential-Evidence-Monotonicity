"""ColBERT scoring backend using PyLate reranking."""

from __future__ import annotations

import os
import sys
from collections import defaultdict


MODEL_NAME = os.environ.get("COLBERT_MODEL", "colbert-ir/colbertv2.0")
BATCH_SIZE = int(os.environ.get("COLBERT_BATCH_SIZE", "512"))


def load_colbert():
    try:
        from pylate import models, rank
        from tqdm import tqdm
    except ImportError as error:
        raise SystemExit(
            "ColBERT scoring requires pylate and tqdm. Install them with: "
            "pip install pylate tqdm"
        ) from error

    print(f"ColBERT: loading PyLate model {MODEL_NAME}", file=sys.stderr, flush=True)
    model = models.ColBERT(model_name_or_path=MODEL_NAME)
    return rank, tqdm, model


def group_by_qid(rows: list[dict]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for row_index, row in enumerate(rows):
        groups[row["qid"]].append(row_index)
    return groups


def normalize_rerank_output(reranked):
    if reranked and isinstance(reranked[0], list):
        return reranked[0]
    return reranked


def score_group(rank, model, query: str, documents: list[str]) -> list[float]:
    document_ids = list(range(len(documents)))
    queries_embeddings = model.encode(
        [query],
        batch_size=BATCH_SIZE,
        is_query=True,
        show_progress_bar=False,
    )
    documents_embeddings = model.encode(
        [documents],
        batch_size=BATCH_SIZE,
        is_query=False,
        show_progress_bar=False,
    )
    reranked = rank.rerank(
        documents_ids=[document_ids],
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )

    scores = [0.0] * len(documents)
    for result in normalize_rerank_output(reranked):
        document_id = result.get("id", result.get("document_id", result.get("doc_id")))
        if document_id is None:
            continue
        scores[int(document_id)] = float(result["score"])
    return scores


def score(rows: list[dict]) -> list[float]:
    rank, tqdm, model = load_colbert()
    groups = group_by_qid(rows)
    print(
        f"ColBERT: scoring {len(rows):,} passages grouped by {len(groups):,} questions",
        file=sys.stderr,
        flush=True,
    )

    scores = [0.0] * len(rows)
    for _qid, indices in tqdm(
        groups.items(),
        desc="ColBERT questions",
        unit="question",
        file=sys.stderr,
    ):
        group_rows = [rows[index] for index in indices]
        query = group_rows[0]["question"]
        documents = [row["passage"] for row in group_rows]
        group_scores = score_group(rank, model, query, documents)
        for row_index, score_value in zip(indices, group_scores):
            scores[row_index] = score_value
    return scores
