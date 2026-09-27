"""ReasonIR scoring backend using the official Hugging Face model."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version


MODEL_NAME = os.environ.get("REASONIR_MODEL", "reasonir/ReasonIR-8B")
MIN_TRANSFORMERS_VERSION = (4, 47)
QUERY_INSTRUCTION = os.environ.get("REASONIR_QUERY_INSTRUCTION", "")
DOC_INSTRUCTION = os.environ.get("REASONIR_DOC_INSTRUCTION", "")
BATCH_SIZE = int(os.environ.get("REASONIR_BATCH_SIZE", "64"))
CHUNK_SIZE = int(os.environ.get("REASONIR_CHUNK_SIZE", "1000"))
NORMALIZE = os.environ.get("REASONIR_NORMALIZE", "false").lower() in {"1", "true", "yes"}


def check_transformers_version() -> None:
    installed = version("transformers")
    major, minor, *_rest = (int(part) for part in installed.split(".")[:2])
    if major >= 5 or (major, minor) < MIN_TRANSFORMERS_VERSION:
        raise SystemExit(
            "ReasonIR requires transformers>=4.47,<5. "
            f"Found transformers=={installed}. Install a compatible version with: "
            "pip install 'transformers>=4.47,<5'"
        )


def load_reasonir():
    try:
        import numpy as np
        import torch
        from tqdm import tqdm
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "ReasonIR scoring requires transformers, torch, numpy, and tqdm. "
            "Install them with: pip install 'transformers>=4.47,<5' torch tqdm"
        ) from error

    check_transformers_version()

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device = "cuda"
    else:
        gpu_count = 0
        device = "cpu"

    print(f"ReasonIR: loading tokenizer {MODEL_NAME}", file=sys.stderr, flush=True)
    AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"ReasonIR: loading model {MODEL_NAME}", file=sys.stderr, flush=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    model.eval()

    if gpu_count > 1:
        devices = [f"cuda:{index}" for index in range(gpu_count)]
        print(
            f"ReasonIR: {gpu_count} GPUs are visible; using cuda:0 for instruction-aware encoding",
            file=sys.stderr,
            flush=True,
        )
    elif gpu_count == 1:
        devices = ["cuda:0"]
        print("ReasonIR: using 1 GPU", file=sys.stderr, flush=True)
    else:
        devices = ["cpu"]
        print("ReasonIR: using CPU", file=sys.stderr, flush=True)

    return np, torch, tqdm, model, devices


def chunks(items: list[str], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def to_numpy(np, torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def encode_single_process(np, torch, tqdm, model, texts: list[str], instruction: str, desc: str):
    encoded_chunks = []
    with torch.no_grad():
        for chunk in tqdm(
            list(chunks(texts, CHUNK_SIZE)),
            desc=desc,
            unit="chunk",
            file=sys.stderr,
        ):
            embeddings = model.encode(chunk, instruction=instruction)
            embeddings = to_numpy(np, torch, embeddings)
            if NORMALIZE:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, 1e-12, None)
            encoded_chunks.append(embeddings)
    return np.concatenate(encoded_chunks, axis=0)


def encode_with_model(np, torch, tqdm, model, texts: list[str], instruction: str, devices: list[str], desc: str):
    if not texts:
        return np.array([])

    if len(devices) > 1:
        print(
            "ReasonIR: multi-GPU encoding is not enabled because the official "
            "AutoModel encode API is custom; using cuda:0 for correctness.",
            file=sys.stderr,
            flush=True,
        )
    return encode_single_process(np, torch, tqdm, model, texts, instruction, desc)


def score(rows: list[dict]) -> list[float]:
    np, torch, tqdm, model, devices = load_reasonir()

    question_by_qid = {}
    for row in rows:
        question_by_qid.setdefault(row["qid"], row["question"])

    qids = list(question_by_qid)
    questions = [question_by_qid[qid] for qid in qids]
    passages = [row["passage"] for row in rows]

    print(f"ReasonIR: encoding {len(questions):,} unique questions", file=sys.stderr, flush=True)
    question_matrix = encode_with_model(np, torch, tqdm, model, questions, QUERY_INSTRUCTION, devices, "ReasonIR questions")
    question_embeddings = dict(zip(qids, question_matrix))

    print(f"ReasonIR: encoding {len(passages):,} passages", file=sys.stderr, flush=True)
    passage_matrix = encode_with_model(np, torch, tqdm, model, passages, DOC_INSTRUCTION, devices, "ReasonIR passages")

    scores = []
    for row, passage_embedding in tqdm(
        zip(rows, passage_matrix),
        total=len(rows),
        desc="ReasonIR scoring",
        unit="pair",
        file=sys.stderr,
    ):
        score_value = np.dot(question_embeddings[row["qid"]], passage_embedding)
        scores.append(float(score_value))
    return scores
