"""Analysis of queries with one or several relevant fables."""

from __future__ import annotations

from collections import defaultdict

METRICS = ("MRR", "MAP@10", "Recall@10")
SUBSET_NAMES = {
    "one": "one_relevant",
    "one_relevant": "one_relevant",
    "several": "several_relevant",
    "several_relevant": "several_relevant",
}

PUBLIC_REPRESENTATIONS = {
    "fable_only": ("fable_only", "raw"),
    "summary_only": ("summary_only", "direct_moral_only"),
    "summary_plus_fable": (
        "summary_plus_fable_summary_first",
        "direct_moral_fable",
    ),
}


def split_relevance_groups(
    qrels: dict[str, set[str] | frozenset[str]],
) -> dict[str, set[str]]:
    """Split queries by whether one or several fables are relevant."""

    empty = sorted(query_id for query_id, relevant in qrels.items() if not relevant)
    if empty:
        raise ValueError(f"Queries have no relevant fables: {empty}")
    return {
        "one": {
            query_id for query_id, relevant in qrels.items() if len(relevant) == 1
        },
        "several": {
            query_id for query_id, relevant in qrels.items() if len(relevant) > 1
        },
    }


def aggregate_relevance_groups(
    per_query_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Average released per-query paper metrics for each run and group."""

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen_queries: set[tuple[str, str]] = set()
    for row in per_query_rows:
        run_id = row.get("run_id", "")
        query_id = row.get("query_id", "")
        raw_subset = row.get("subset", "")
        if not run_id or not query_id:
            raise ValueError("Every per-query row must include run_id and query_id")
        if raw_subset not in SUBSET_NAMES:
            raise ValueError(f"Unknown relevance subset: {raw_subset}")
        identity = (run_id, query_id)
        if identity in seen_queries:
            raise ValueError(f"Duplicate per-query row: {run_id}, {query_id}")
        seen_queries.add(identity)
        grouped[(run_id, SUBSET_NAMES[raw_subset])].append(row)

    run_ids = sorted({run_id for run_id, _ in grouped})
    result: list[dict[str, object]] = []
    for run_id in run_ids:
        for subset in ("one_relevant", "several_relevant"):
            rows = grouped.get((run_id, subset), [])
            if not rows:
                raise ValueError(f"Run {run_id} is missing the {subset} group")
            result.append(
                {
                    "run_id": run_id,
                    "subset": subset,
                    "n_queries": len(rows),
                    **{
                        metric: sum(float(row[metric]) for row in rows) / len(rows)
                        for metric in METRICS
                    },
                }
            )
    return result


def public_representation_fields(
    document_representation: str,
) -> tuple[str, str]:
    """Return the labels used by the frozen camera-ready result table."""

    try:
        return PUBLIC_REPRESENTATIONS[document_representation]
    except KeyError as error:
        raise ValueError(
            f"Unknown document representation: {document_representation}"
        ) from error
