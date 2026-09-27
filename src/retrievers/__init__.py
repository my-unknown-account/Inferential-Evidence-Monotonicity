"""Retriever backend registry."""

from __future__ import annotations

from collections.abc import Callable

from . import bge, bm25, colbert, dpr, reasonir, splade


ScoreFn = Callable[[list[dict]], list[float]]


RETRIEVERS: dict[str, ScoreFn] = {
    "bm25": bm25.score,
    "bge": bge.score,
    "dpr": dpr.score,
    "splade": splade.score,
    "colbert": colbert.score,
    "reasonir": reasonir.score,
}


def get_retriever(name: str) -> ScoreFn:
    try:
        return RETRIEVERS[name]
    except KeyError as error:
        available = ", ".join(sorted(RETRIEVERS))
        raise ValueError(f"Unknown retriever '{name}'. Available: {available}") from error
