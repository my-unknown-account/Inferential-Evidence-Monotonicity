"""DPR scoring backend using Hugging Face transformers."""

from __future__ import annotations

import os
import sys


QUESTION_MODEL = os.environ.get("DPR_QUESTION_MODEL", "facebook/dpr-question_encoder-single-nq-base")
CONTEXT_MODEL = os.environ.get("DPR_CONTEXT_MODEL", "facebook/dpr-ctx_encoder-single-nq-base")
QUESTION_MAX_LENGTH = int(os.environ.get("DPR_QUESTION_MAX_LENGTH", "64"))
PASSAGE_MAX_LENGTH = int(os.environ.get("DPR_PASSAGE_MAX_LENGTH", "256"))
BATCH_SIZE = int(os.environ.get("DPR_BATCH_SIZE", "512"))


def encoder_pooler_output(encoder, encoded):
    return encoder(**encoded).pooler_output


def load_dpr():
    try:
        import torch
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizerFast,
        )
        from tqdm import tqdm
    except ImportError as error:
        raise SystemExit(
            "DPR scoring requires torch, transformers, and tqdm. Install them with: "
            "pip install torch transformers tqdm"
        ) from error

    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
    else:
        device = "cpu"
        gpu_count = 0

    print(f"DPR: loading question encoder {QUESTION_MODEL}", file=sys.stderr, flush=True)
    question_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(QUESTION_MODEL)
    question_encoder = DPRQuestionEncoder.from_pretrained(QUESTION_MODEL).to(device).eval()

    print(f"DPR: loading context encoder {CONTEXT_MODEL}", file=sys.stderr, flush=True)
    context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(CONTEXT_MODEL)
    context_encoder = DPRContextEncoder.from_pretrained(CONTEXT_MODEL).to(device).eval()

    if gpu_count > 1:
        print(f"DPR: using {gpu_count} GPUs with torch.nn.DataParallel", file=sys.stderr, flush=True)
        question_encoder = torch.nn.DataParallel(question_encoder)
        context_encoder = torch.nn.DataParallel(context_encoder)
    elif gpu_count == 1:
        print("DPR: using 1 GPU", file=sys.stderr, flush=True)
    else:
        print("DPR: using CPU", file=sys.stderr, flush=True)

    return torch, tqdm, device, question_tokenizer, question_encoder, context_tokenizer, context_encoder


def encode_questions(rows: list[dict], torch, tqdm, device: str, tokenizer, encoder) -> dict[str, object]:
    question_by_qid = {}
    for row in rows:
        question_by_qid.setdefault(row["qid"], row["question"])

    items = list(question_by_qid.items())
    embeddings = {}
    with torch.no_grad():
        for start in tqdm(
            range(0, len(items), BATCH_SIZE),
            desc="DPR questions",
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
                max_length=QUESTION_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            output = encoder_pooler_output(encoder, encoded)
            for qid, embedding in zip(qids, output):
                embeddings[qid] = embedding.detach()
    return embeddings


def score_passages(rows: list[dict], question_embeddings: dict[str, object], torch, tqdm, device: str, tokenizer, encoder) -> list[float]:
    scores = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(rows), BATCH_SIZE),
            desc="DPR passages",
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
            passage_embeddings = encoder_pooler_output(encoder, encoded)
            question_batch = torch.stack([question_embeddings[row["qid"]] for row in batch]).to(device)
            batch_scores = (question_batch * passage_embeddings).sum(dim=1)
            scores.extend(float(score) for score in batch_scores.detach().cpu())
    return scores


def score(rows: list[dict]) -> list[float]:
    torch, tqdm, device, question_tokenizer, question_encoder, context_tokenizer, context_encoder = load_dpr()
    print(f"DPR: encoding {len({row['qid'] for row in rows}):,} unique questions", file=sys.stderr, flush=True)
    question_embeddings = encode_questions(rows, torch, tqdm, device, question_tokenizer, question_encoder)
    print(f"DPR: scoring {len(rows):,} passages", file=sys.stderr, flush=True)
    return score_passages(rows, question_embeddings, torch, tqdm, device, context_tokenizer, context_encoder)
