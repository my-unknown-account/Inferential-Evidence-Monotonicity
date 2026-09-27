"""BGE scoring backend using sentence-transformers."""

from __future__ import annotations

import os
import sys


MODEL_NAME = os.environ.get("BGE_MODEL", "BAAI/bge-base-en-v1.5")
QUERY_PREFIX = os.environ.get("BGE_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
BATCH_SIZE = int(os.environ.get("BGE_BATCH_SIZE", "256"))
BGE_CHUNK_SIZE = int(os.environ.get("BGE_CHUNK_SIZE", "10000"))
NORMALIZE = os.environ.get("BGE_NORMALIZE", "true").lower() in {"1", "true", "yes"}


def load_bge():
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
    except ImportError as error:
        raise SystemExit(
            "BGE scoring requires sentence-transformers, torch, numpy, and tqdm. "
            "Install them with: pip install sentence-transformers torch tqdm"
        ) from error

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device = "cuda"
    else:
        gpu_count = 0
        device = "cpu"

    print(f"BGE: loading model {MODEL_NAME}", file=sys.stderr, flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)

    if gpu_count > 1:
        devices = [f"cuda:{index}" for index in range(gpu_count)]
        print(f"BGE: using {gpu_count} GPUs with sentence-transformers multi-process encoding", file=sys.stderr, flush=True)
    elif gpu_count == 1:
        devices = ["cuda:0"]
        print("BGE: using 1 GPU", file=sys.stderr, flush=True)
    else:
        devices = ["cpu"]
        print("BGE: using CPU", file=sys.stderr, flush=True)

    return np, tqdm, model, devices


def chunks(items: list[str], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def encode_with_model(np, tqdm, model, texts: list[str], devices: list[str], desc: str):
    if not texts:
        return np.array([])

    if len(devices) > 1:
        pool = model.start_multi_process_pool(target_devices=devices)
        try:
            encoded_chunks = []
            for chunk in tqdm(
                list(chunks(texts, BGE_CHUNK_SIZE)),
                desc=desc,
                unit="chunk",
                file=sys.stderr,
            ):
                encoded_chunks.append(
                    model.encode_multi_process(
                        chunk,
                        pool,
                        batch_size=BATCH_SIZE,
                        normalize_embeddings=NORMALIZE,
                    )
                )
            return np.concatenate(encoded_chunks, axis=0)
        finally:
            model.stop_multi_process_pool(pool)

    encoded_chunks = []
    for chunk in tqdm(
        list(chunks(texts, BGE_CHUNK_SIZE)),
        desc=desc,
        unit="chunk",
        file=sys.stderr,
    ):
        encoded_chunks.append(
            model.encode(
                chunk,
                batch_size=BATCH_SIZE,
                normalize_embeddings=NORMALIZE,
                show_progress_bar=False,
            )
        )
    return np.concatenate(encoded_chunks, axis=0)


def score(rows: list[dict]) -> list[float]:
    np, tqdm, model, devices = load_bge()

    question_by_qid = {}
    for row in rows:
        question_by_qid.setdefault(row["qid"], row["question"])

    qids = list(question_by_qid)
    questions = [QUERY_PREFIX + question_by_qid[qid] for qid in qids]
    passages = [row["passage"] for row in rows]

    print(f"BGE: encoding {len(questions):,} unique questions", file=sys.stderr, flush=True)
    question_matrix = encode_with_model(np, tqdm, model, questions, devices, "BGE questions")
    question_embeddings = dict(zip(qids, question_matrix))

    print(f"BGE: encoding {len(passages):,} passages", file=sys.stderr, flush=True)
    passage_matrix = encode_with_model(np, tqdm, model, passages, devices, "BGE passages")

    scores = []
    for row, passage_embedding in tqdm(
        zip(rows, passage_matrix),
        total=len(rows),
        desc="BGE scoring",
        unit="pair",
        file=sys.stderr,
    ):
        score_value = np.dot(question_embeddings[row["qid"]], passage_embedding)
        scores.append(float(score_value))
    return scores
