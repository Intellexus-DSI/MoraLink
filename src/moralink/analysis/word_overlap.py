"""Content-word overlap and its association with ranking changes."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from moralink.data import Benchmark

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
PROMPT_VARIANTS = ("direct_moral", "conceptual_abstract", "cot_proverb")


def tokenize_content(text: str) -> set[str]:
    """Return lowercase content words after English stop-word removal."""

    tokens = {token.lower() for token in TOKEN_RE.findall(text or "")}
    return {
        token
        for token in tokens
        if token not in ENGLISH_STOP_WORDS and len(token) > 1
    }


def content_iou(left: str, right: str) -> float:
    """Return intersection over union for unique content-word sets."""

    left_tokens = tokenize_content(left)
    right_tokens = tokenize_content(right)
    union = left_tokens | right_tokens
    return 0.0 if not union else len(left_tokens & right_tokens) / len(union)


def spearman_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    """Return Spearman correlation using average ranks for tied values."""

    return _pearson(_rankdata(x_values), _rankdata(y_values))


def analyze_word_overlap(
    benchmark: Benchmark,
    summary_collections: dict[str, dict[str, dict[str, str]]],
    rank_change_rows: list[dict[str, str]],
    official_runs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build overlap summaries and paired rank-change correlations."""

    query_texts = {query.doc_id: query.text for query in benchmark.queries}
    raw_documents = {fable.doc_id: fable.text for fable in benchmark.fables}
    system_rows = [
        _summarize_system(
            "raw_fable",
            "",
            "",
            query_texts,
            raw_documents,
            benchmark.qrels,
        )
    ]
    for source, summaries in summary_collections.items():
        for prompt_variant in PROMPT_VARIANTS:
            documents: dict[str, str] = {}
            for fable in benchmark.fables:
                summary = summaries.get(fable.doc_id, {})
                text = summary.get(prompt_variant)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Missing {prompt_variant} summary for {source}, {fable.doc_id}"
                    )
                documents[fable.doc_id] = text
            system_rows.append(
                _summarize_system(
                    f"{source}__{prompt_variant}",
                    source,
                    prompt_variant,
                    query_texts,
                    documents,
                    benchmark.qrels,
                )
            )
    correlation_rows = aggregate_rank_change_correlations(
        rank_change_rows,
        official_runs,
    )
    return system_rows, correlation_rows


def aggregate_rank_change_correlations(
    rows: list[dict[str, str]],
    official_runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate released rank changes after validating their raw-run pairing."""

    raw_pairs = {
        (str(run["model_alias"]), str(run["instruction_variant"]))
        for run in official_runs
        if run.get("document_representation") == "fable_only"
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)

    result: list[dict[str, object]] = []
    for run_id in sorted(grouped):
        run_rows = grouped[run_id]
        metadata = {
            (
                row["model_alias"],
                row["instruction_variant"],
                row["summary_generator"],
                row["prompt_variant"],
            )
            for row in run_rows
        }
        if len(metadata) != 1:
            raise ValueError(f"Inconsistent metadata within rank-change run: {run_id}")
        model, instruction, generator, prompt = next(iter(metadata))
        if instruction in {"no_instr", "general"} and (
            model,
            instruction,
        ) not in raw_pairs:
            raise ValueError(
                f"Run {run_id} has no paired raw run with the same model and instruction"
            )
        expected_prefix = f"{model}__{instruction}__"
        if not run_id.startswith(expected_prefix):
            raise ValueError(
                f"Run {run_id} does not match its model and instruction metadata"
            )
        if instruction not in {"no_instr", "general"} and any(
            not row.get("raw_best_rank") or not row.get("raw_reciprocal_rank")
            for row in run_rows
        ):
            raise ValueError(
                f"Historical run {run_id} does not include its paired raw ranks"
            )
        query_ids = [row["query_id"] for row in run_rows]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError(f"Duplicate query rows in rank-change run: {run_id}")

        overlap = [float(row["overlap_improvement"]) for row in run_rows]
        reciprocal_rank = [
            float(row["reciprocal_rank_improvement"]) for row in run_rows
        ]
        rank = [float(row["rank_improvement"]) for row in run_rows]
        result.append(
            {
                "run_id": run_id,
                "model_alias": model,
                "instruction_variant": instruction,
                "summary_generator": generator,
                "prompt_variant": prompt,
                "n_queries": len(run_rows),
                "pearson_overlap_vs_reciprocal_rank_improvement": _pearson(
                    overlap, reciprocal_rank
                ),
                "spearman_overlap_vs_reciprocal_rank_improvement": (
                    spearman_correlation(overlap, reciprocal_rank)
                ),
                "pearson_overlap_vs_rank_improvement": _pearson(overlap, rank),
                "spearman_overlap_vs_rank_improvement": spearman_correlation(
                    overlap, rank
                ),
            }
        )
    result.sort(
        key=lambda row: (
            str(row["model_alias"]),
            str(row["instruction_variant"]),
            str(row["summary_generator"]),
            str(row["prompt_variant"]),
        )
    )
    return result


def _summarize_system(
    system: str,
    summary_generator: str,
    prompt_variant: str,
    queries: dict[str, str],
    documents: dict[str, str],
    qrels: dict[str, frozenset[str]],
) -> dict[str, object]:
    pair_values: list[float] = []
    query_means: list[float] = []
    query_maxima: list[float] = []
    for query_id, relevant in qrels.items():
        values = [
            content_iou(queries[query_id], documents[fable_id])
            for fable_id in relevant
        ]
        pair_values.extend(values)
        query_means.append(statistics.fmean(values))
        query_maxima.append(max(values))
    return {
        "system": system,
        "summary_generator": summary_generator,
        "prompt_variant": prompt_variant,
        "n_queries": len(qrels),
        "n_relevant_pairs": len(pair_values),
        "mean_pair_iou": statistics.fmean(pair_values),
        "median_pair_iou": statistics.median(pair_values),
        "mean_query_mean_iou": statistics.fmean(query_means),
        "mean_query_max_iou": statistics.fmean(query_maxima),
    }


def _rankdata(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    sorted_values = array[order]
    index = 0
    while index < len(array):
        end = index + 1
        while end < len(array) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks.tolist()


def _pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values):
        raise ValueError("Correlation inputs must have the same length")
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])
