"""Embedding retrieval used by the MoraLink zero-shot experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from moralink.data import Benchmark
from moralink.metrics import compute_paper_metrics
from moralink.models import ModelSpec, instruction_prefix
from moralink.representations import build_document


def rank_embeddings(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: list[str],
    fable_ids: list[str],
) -> dict[str, list[str]]:
    """Rank normalized cosine scores, preserving fable order for exact ties."""

    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("Query and document embeddings must be two-dimensional")
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("The number of query IDs does not match the embeddings")
    if len(fable_ids) != document_embeddings.shape[0]:
        raise ValueError("The number of fable IDs does not match the embeddings")
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError("Query and document embedding dimensions do not match")

    normalized_queries = _normalize(query_embeddings)
    normalized_documents = _normalize(document_embeddings)
    scores = normalized_queries @ normalized_documents.T
    order = np.argsort(-scores, axis=1, kind="stable")
    return {
        query_id: [fable_ids[index] for index in indices]
        for query_id, indices in zip(query_ids, order, strict=True)
    }


def run_zero_shot(
    benchmark: Benchmark,
    model_spec: ModelSpec,
    representation: str,
    instruction: str,
    *,
    batch_size: int | None = None,
    device: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Encode all MoraLink queries and documents and return full rankings."""

    from sentence_transformers import SentenceTransformer

    resolved_device = device or _default_device()
    model = SentenceTransformer(
        model_spec.model_id,
        device=resolved_device,
        trust_remote_code=model_spec.trust_remote_code,
    )
    query_ids = [query.doc_id for query in benchmark.queries]
    fable_ids = [fable.doc_id for fable in benchmark.fables]
    prefix = instruction_prefix(instruction)
    query_texts = [f"{prefix}{query.text}" for query in benchmark.queries]
    documents = [
        build_document(
            fable,
            benchmark.summaries.get(fable.doc_id),
            representation,
        )
        for fable in benchmark.fables
    ]
    encode_kwargs: dict[str, Any] = {
        "batch_size": batch_size or model_spec.batch_size,
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "convert_to_numpy": True,
    }
    query_embeddings = model.encode(query_texts, **encode_kwargs)
    document_embeddings = model.encode(documents, **encode_kwargs)
    rankings = rank_embeddings(
        query_embeddings,
        document_embeddings,
        query_ids,
        fable_ids,
    )
    metrics = compute_paper_metrics(rankings, benchmark.qrels)
    return rankings, metrics


def save_ranking(
    path: Path,
    rankings: dict[str, list[str]],
    *,
    run_id: str,
    model_spec: ModelSpec,
    instruction: str,
    representation: str,
    summary_source: str,
    n_docs: int | None = None,
) -> None:
    """Save a complete ranking in the public ranking_v1 schema."""

    query_ids = list(rankings)
    ranked_fable_ids = [rankings[query_id] for query_id in query_ids]
    depth = len(ranked_fable_ids[0]) if ranked_fable_ids else 0
    document_count = depth if n_docs is None else n_docs
    if depth > document_count:
        raise ValueError("Ranking depth cannot exceed the document count")
    payload = {
        "schema_version": "ranking_v1",
        "run_id": run_id,
        "model_alias": model_spec.alias,
        "model_id": model_spec.model_id,
        "instruction_variant": instruction,
        "document_representation": representation,
        "summary_generator": summary_source if representation != "fable_only" else "",
        "generator_prompt_variant": (
            "direct_moral" if representation != "fable_only" else ""
        ),
        "n_queries": len(query_ids),
        "n_docs": document_count,
        "ranking_depth": depth,
        "query_ids": query_ids,
        "ranked_fable_ids": ranked_fable_ids,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
