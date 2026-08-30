from pathlib import Path

from moralink.tables import create_paper_tables

ROOT = Path(__file__).resolve().parents[1]


def test_table_generation_is_deterministic(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_paths = create_paper_tables(ROOT / "results", first)
    second_paths = create_paper_tables(ROOT / "results", second)

    assert len(first_paths) == 8
    assert [path.name for path in first_paths] == [path.name for path in second_paths]
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]


def test_rendered_tables_round_only_at_display_time(tmp_path: Path):
    paths = create_paper_tables(ROOT / "results", tmp_path)
    zero_shot_markdown = next(
        path.read_text(encoding="utf-8")
        for path in paths
        if path.name == "zero_shot.md"
    )

    assert "0.095" in zero_shot_markdown
    assert "0.095177" not in zero_shot_markdown
    assert "**" in zero_shot_markdown


def test_latex_tables_escape_special_characters(tmp_path: Path):
    paths = create_paper_tables(ROOT / "results", tmp_path)
    relevance_latex = next(
        path.read_text(encoding="utf-8")
        for path in paths
        if path.name == "relevance_groups.tex"
    )

    assert "\\begin{tabular}" in relevance_latex
    assert "one\\_relevant" in relevance_latex
    assert "\\textbf{" in relevance_latex
