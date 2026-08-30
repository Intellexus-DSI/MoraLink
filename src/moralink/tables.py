"""Deterministic Markdown and LaTeX rendering for released result tables."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableSpec:
    name: str
    fields: tuple[str, ...]
    metrics: tuple[str, ...]
    group_key: Callable[[dict[str, str]], tuple[str, ...]]


TABLE_SPECS = (
    TableSpec(
        name="zero_shot",
        fields=(
            "model_alias",
            "instruction_variant",
            "document_representation",
            "MRR@10",
            "MAP@10",
            "Recall@10",
        ),
        metrics=("MRR@10", "MAP@10", "Recall@10"),
        group_key=lambda row: (row["model_alias"], row["instruction_variant"]),
    ),
    TableSpec(
        name="fine_tuning",
        fields=(
            "section",
            "model",
            "setting",
            "comparison",
            "MRR@10",
            "MRR@10_std",
            "MAP@10",
            "MAP@10_std",
            "Recall@10",
            "Recall@10_std",
            "delta_MRR@10",
            "delta_MRR@10_pct",
            "delta_MAP@10",
            "delta_MAP@10_pct",
            "delta_Recall@10",
            "delta_Recall@10_pct",
        ),
        metrics=(
            "MRR@10",
            "MAP@10",
            "Recall@10",
            "delta_MRR@10",
            "delta_MRR@10_pct",
            "delta_MAP@10",
            "delta_MAP@10_pct",
            "delta_Recall@10",
            "delta_Recall@10_pct",
        ),
        group_key=lambda row: (
            row["section"],
            row["model"] if row["section"] == "per_model" else "",
        ),
    ),
    TableSpec(
        name="relevance_groups",
        fields=(
            "model_alias",
            "instruction_variant",
            "subset",
            "document_representation",
            "n_queries",
            "MRR",
            "MAP@10",
            "Recall@10",
        ),
        metrics=("MRR", "MAP@10", "Recall@10"),
        group_key=lambda row: (
            row["model_alias"],
            row["instruction_variant"],
            row["subset"],
        ),
    ),
    TableSpec(
        name="word_overlap",
        fields=(
            "model_alias",
            "instruction_variant",
            "summary_generator",
            "prompt_variant",
            "n_queries",
            "pearson_overlap_vs_reciprocal_rank_improvement",
            "spearman_overlap_vs_reciprocal_rank_improvement",
            "pearson_overlap_vs_rank_improvement",
            "spearman_overlap_vs_rank_improvement",
        ),
        metrics=(
            "pearson_overlap_vs_reciprocal_rank_improvement",
            "spearman_overlap_vs_reciprocal_rank_improvement",
            "pearson_overlap_vs_rank_improvement",
            "spearman_overlap_vs_rank_improvement",
        ),
        group_key=lambda row: (
            row["model_alias"],
            row["instruction_variant"],
            row["summary_generator"],
        ),
    ),
)


def load_result_tables(root: Path) -> dict[str, list[dict[str, str]]]:
    """Load the four frozen result tables without changing numeric precision."""

    result = {}
    for spec in TABLE_SPECS:
        path = root / f"{spec.name}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            result[spec.name] = list(csv.DictReader(handle))
    return result


def create_paper_tables(results_dir: Path, output_dir: Path) -> list[Path]:
    """Create Markdown and LaTeX views of all frozen result CSV files."""

    tables = load_result_tables(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in TABLE_SPECS:
        rows = tables[spec.name]
        maxima = _group_maxima(rows, spec)
        markdown_path = output_dir / f"{spec.name}.md"
        latex_path = output_dir / f"{spec.name}.tex"
        markdown_path.write_text(_render_markdown(rows, spec, maxima), encoding="utf-8")
        latex_path.write_text(_render_latex(rows, spec, maxima), encoding="utf-8")
        paths.extend((markdown_path, latex_path))
    return paths


def _group_maxima(
    rows: list[dict[str, str]], spec: TableSpec
) -> dict[tuple[tuple[str, ...], str], float]:
    maxima: dict[tuple[tuple[str, ...], str], float] = {}
    for row in rows:
        group = spec.group_key(row)
        for metric in spec.metrics:
            value = _as_float(row.get(metric, ""))
            if value is None:
                continue
            key = (group, metric)
            maxima[key] = max(value, maxima.get(key, value))
    return maxima


def _render_markdown(
    rows: list[dict[str, str]],
    spec: TableSpec,
    maxima: dict[tuple[tuple[str, ...], str], float],
) -> str:
    lines = [
        "| " + " | ".join(_escape_markdown(field) for field in spec.fields) + " |",
        "| "
        + " | ".join(
            "---:" if field in spec.metrics else "---" for field in spec.fields
        )
        + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row, field, spec, maxima) for field in spec.fields
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_latex(
    rows: list[dict[str, str]],
    spec: TableSpec,
    maxima: dict[tuple[tuple[str, ...], str], float],
) -> str:
    alignment = "".join("r" if field in spec.metrics else "l" for field in spec.fields)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_escape_latex(field) for field in spec.fields) + r" \\",
        "\\midrule",
    ]
    lines.extend(
        " & ".join(_latex_cell(row, field, spec, maxima) for field in spec.fields)
        + r" \\"
        for row in rows
    )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines) + "\n"


def _markdown_cell(
    row: dict[str, str],
    field: str,
    spec: TableSpec,
    maxima: dict[tuple[tuple[str, ...], str], float],
) -> str:
    value = _display_value(row.get(field, ""), numeric=field in spec.metrics)
    value = _escape_markdown(value)
    if _is_maximum(row, field, spec, maxima):
        return f"**{value}**"
    return value


def _latex_cell(
    row: dict[str, str],
    field: str,
    spec: TableSpec,
    maxima: dict[tuple[tuple[str, ...], str], float],
) -> str:
    value = _escape_latex(
        _display_value(row.get(field, ""), numeric=field in spec.metrics)
    )
    if _is_maximum(row, field, spec, maxima):
        return f"\\textbf{{{value}}}"
    return value


def _is_maximum(
    row: dict[str, str],
    field: str,
    spec: TableSpec,
    maxima: dict[tuple[tuple[str, ...], str], float],
) -> bool:
    value = _as_float(row.get(field, ""))
    if value is None or field not in spec.metrics:
        return False
    return value == maxima[(spec.group_key(row), field)]


def _display_value(value: str, *, numeric: bool) -> str:
    parsed = _as_float(value)
    if numeric and parsed is not None:
        return f"{parsed:.3f}"
    return value


def _as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _escape_markdown(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)
