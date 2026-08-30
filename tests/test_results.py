import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def test_released_result_tables_have_frozen_counts():
    assert row_count(RESULTS / "zero_shot.csv") == 144
    assert row_count(RESULTS / "fine_tuning.csv") == 36
    assert row_count(RESULTS / "relevance_groups.csv") == 288
    assert row_count(RESULTS / "word_overlap.csv") == 159


def test_zero_shot_table_has_the_reported_run_schema():
    with (RESULTS / "zero_shot.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len({row["run_id"] for row in rows}) == 144
    assert {row["document_representation"] for row in rows} == {
        "fable_only",
        "summary_only",
        "summary_plus_fable_summary_first",
    }
    assert {row["instruction_variant"] for row in rows} == {
        "general",
        "no_instr",
    }


def test_frozen_csv_files_use_unix_line_endings():
    for path in RESULTS.glob("*.csv"):
        assert b"\r\n" not in path.read_bytes()
