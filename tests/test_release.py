from pathlib import Path

import pytest

from moralink.release_validation import (
    scan_forbidden_content,
    validate_result_counts,
)


def test_forbidden_content_scan_reports_private_paths(tmp_path: Path):
    (tmp_path / "bad.txt").write_text("/data/lior/private", encoding="utf-8")

    with pytest.raises(ValueError, match="private path"):
        scan_forbidden_content(tmp_path)


def test_forbidden_content_scan_skips_virtual_environments_and_lock_files(
    tmp_path: Path,
):
    virtual_environment = tmp_path / ".venv"
    virtual_environment.mkdir()
    (virtual_environment / "cached.txt").write_text(
        "BEGIN OPENSSH PRIVATE KEY", encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_text("sk-example", encoding="utf-8")

    scan_forbidden_content(tmp_path)


def test_frozen_result_counts_are_checked(tmp_path: Path):
    expected = {
        "zero_shot.csv": 144,
        "fine_tuning.csv": 36,
        "relevance_groups.csv": 288,
        "word_overlap.csv": 159,
    }
    for filename, count in expected.items():
        rows = "value\n" + "\n".join(str(index) for index in range(count)) + "\n"
        (tmp_path / filename).write_text(rows, encoding="utf-8")

    assert validate_result_counts(tmp_path) == expected

    (tmp_path / "zero_shot.csv").write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zero_shot.csv"):
        validate_result_counts(tmp_path)
