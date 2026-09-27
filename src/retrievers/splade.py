"""SPLADE scoring backend using a Hugging Face masked-LM checkpoint."""

from __future__ import annotations

import os
import sys


MODEL_NAME = os.environ.get("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")
QUERY_MAX_LENGTH = int(os.environ.get("SPLADE_QUERY_MAX_LENGTH", "64"))
PASSAGE_MAX_LENGTH = int(os.environ.get("SPLADE_PASSAGE_MAX_LENGTH", "256"))
BATCH_SIZE = int(os.environ.get("SPLADE_BATCH_SIZE", "512"))


def encoder_logits(model, encoded):
    return model(**encoded).logits


def splade_pool(torch, logits, attention_mask):
    weights = torch.log1p(torch.relu(logits))
    weights = weights * attention_mask.unsqueeze(-1)
    return torch.max(weights, dim=1).values


def load_splade():
    try:
        import torch
        from tqdm import tqdm
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "SPLADE scoring requires torch, transformers, and tqdm. Install them with: "
            "pip install torch transformers tqdm"
        ) from error

    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
    else:
        device = "cpu"
        gpu_count = 0

    print(f"SPLADE: loading model {MODEL_NAME}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()

    if gpu_count > 1:
        print(f"SPLADE: using {gpu_count} GPUs with torch.nn.DataParallel", file=sys.stderr, flush=True)
        model = torch.nn.DataParallel(model)
    elif gpu_count == 1:
        print("SPLADE: using 1 GPU", file=sys.stderr, flush=True)
    else:
        print("SPLADE: using CPU", file=sys.stderr, flush=True)

    return torch, tqdm, device, tokenizer, model


def encode_questions(rows: list[dict], torch, tqdm, device: str, tokenizer, model) -> dict[str, object]:
    question_by_qid = {}
    for row in rows:
        question_by_qid.setdefault(row["qid"], row["question"])

    items = list(question_by_qid.items())
    embeddings = {}
    with torch.no_grad():
        for start in tqdm(
            range(0, len(items), BATCH_SIZE),
            desc="SPLADE questions",
            unit="batch",
            file=sys.stderr,
        ):
            batch = items[start : start + BATCH_SIZE]
            qids = [qid for qid, _question in batch]
            questions = [question for _qid, question in batch]
            encoded = tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=QUERY_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = encoder_logits(model, encoded)
            output = splade_pool(torch, logits, encoded["attention_mask"])
            for qid, embedding in zip(qids, output):
                embeddings[qid] = embedding.detach().cpu()
    return embeddings


def score_passages(rows: list[dict], question_embeddings: dict[str, object], torch, tqdm, device: str, tokenizer, model) -> list[float]:
    scores = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(rows), BATCH_SIZE),
            desc="SPLADE passages",
            unit="batch",
            file=sys.stderr,
        ):
            batch = rows[start : start + BATCH_SIZE]
            passages = [row["passage"] for row in batch]
            encoded = tokenizer(
                passages,
                padding=True,
                truncation=True,
                max_length=PASSAGE_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = encoder_logits(model, encoded)
            passage_embeddings = splade_pool(torch, logits, encoded["attention_mask"])
            question_batch = torch.stack([question_embeddings[row["qid"]] for row in batch]).to(device)
            batch_scores = (question_batch * passage_embeddings).sum(dim=1)
            scores.extend(float(score) for score in batch_scores.detach().cpu())
    return scores


def score(rows: list[dict]) -> list[float]:
    torch, tqdm, device, tokenizer, model = load_splade()
    print(f"SPLADE: encoding {len({row['qid'] for row in rows}):,} unique questions", file=sys.stderr, flush=True)
    question_embeddings = encode_questions(rows, torch, tqdm, device, tokenizer, model)
    print(f"SPLADE: scoring {len(rows):,} passages", file=sys.stderr, flush=True)
    return score_passages(rows, question_embeddings, torch, tqdm, device, tokenizer, model)

