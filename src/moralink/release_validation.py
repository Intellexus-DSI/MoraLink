"""End-to-end validation for the public MoraLink release."""

from __future__ import annotations

import ast
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

import yaml

from moralink.artifacts import (
    load_artifact_manifest,
    validate_artifacts,
    validate_checksum_manifest,
)

EXPECTED_RESULTS = {
    "zero_shot.csv": 144,
    "fine_tuning.csv": 36,
    "relevance_groups.csv": 288,
    "word_overlap.csv": 159,
}

SKIPPED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}


def scan_forbidden_content(root: Path) -> None:
    """Reject private paths, credential markers, and likely secret values."""

    literal_markers = {
        "/data/" + "lior": "private path",
        "Para" + "beLink": "private project name",
        "GPU_" + "PASSWORD": "credential marker",
        "TG_BOT_" + "TOKEN": "credential marker",
        "TG_CHAT_" + "ID": "credential marker",
        "WANDB_API_" + "KEY": "credential marker",
        "BEGIN OPENSSH " + "PRIVATE KEY": "private key",
        "Lior" + "DSLab": "credential value",
    }
    secret_patterns = {
        re.compile(r"s" + r"k-[A-Za-z0-9]{16,}"): "possible API key",
        re.compile(r"AI" + r"za[A-Za-z0-9_-]{20,}"): "possible API key",
    }

    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIPPED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == "tests/test_release.py" or path.suffix == ".lock":
            continue
        text = _read_scannable_text(path)
        if text is None:
            continue
        for marker, description in literal_markers.items():
            if marker in text:
                raise ValueError(f"Found {description} in {relative}")
        for pattern, description in secret_patterns.items():
            if pattern.search(text):
                raise ValueError(f"Found {description} in {relative}")


def validate_result_counts(root: Path) -> dict[str, int]:
    """Check the frozen row count of each public result table."""

    counts = {}
    for filename, expected in EXPECTED_RESULTS.items():
        path = root / filename
        with path.open(newline="", encoding="utf-8") as handle:
            actual = sum(1 for _ in csv.DictReader(handle))
        if actual != expected:
            raise ValueError(f"Expected {expected} rows in {filename}, found {actual}")
        counts[filename] = actual
    return counts


def validate_release(repository_root: Path, data_root: Path) -> dict[str, object]:
    """Validate code, downloaded artifacts, results, docs, and security."""

    repository_root = repository_root.resolve()
    data_root = data_root.resolve()
    _validate_package(repository_root)

    manifest_path = repository_root / "configs" / "artifacts.yaml"
    validate_artifacts(data_root, manifest_path)
    validate_checksum_manifest(data_root)
    counts = _release_counts(data_root, manifest_path)

    validate_result_counts(repository_root / "results")
    _validate_documentation(repository_root)
    scan_forbidden_content(repository_root)
    scan_forbidden_content(data_root)

    return {
        "package": "ok",
        "artifacts": "ok",
        "counts": counts,
        "results": "ok",
        "documentation": "ok",
        "security_scan": "ok",
    }


def _validate_package(root: Path) -> None:
    required = (
        root / "pyproject.toml",
        root / "README.md",
        root / "src" / "moralink" / "__init__.py",
    )
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package files: " + ", ".join(missing))
    for path in sorted((root / "src").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _release_counts(data_root: Path, manifest_path: Path) -> dict[str, int]:
    specs = load_artifact_manifest(manifest_path)
    fables = _load_json(data_root / specs["fables"].path)
    queries = _load_json(data_root / specs["queries"].path)
    groups = _load_json(data_root / specs["group_mapping"].path)
    ranking_payload = _load_json(data_root / specs["rankings_manifest"].path)
    if not isinstance(ranking_payload, dict) or not isinstance(
        ranking_payload.get("rankings"), list
    ):
        raise TypeError("Ranking manifest must contain a rankings list")

    summary_specs = [
        spec for name, spec in specs.items() if name.startswith("summaries_")
    ]
    for spec in summary_specs:
        summaries = _load_json(data_root / spec.path)
        if not isinstance(summaries, list) or len(summaries) != 709:
            raise ValueError(f"Expected 709 summaries in {spec.path}")

    counts = {
        "fables": len(fables),
        "queries": len(queries),
        "groups": len(groups),
        "summary_collections": len(summary_specs),
        "official_rankings": len(ranking_payload["rankings"]),
    }
    expected = {
        "fables": 709,
        "queries": 668,
        "groups": 558,
        "summary_collections": 4,
        "official_rankings": 144,
    }
    if counts != expected:
        raise ValueError(f"Release counts do not match: {counts}")
    return counts


def _validate_documentation(root: Path) -> None:
    required = (
        "README.md",
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
        "docs/REPRODUCING.md",
        "docs/DATA.md",
        "docs/MORAL_GROUPING.md",
        "docs/RESULTS.md",
        "docs/LICENSING.md",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Missing documentation file: {relative}")

    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("LICENSE is not the Apache 2.0 text")
    citation: Any = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    if not isinstance(citation, dict) or len(citation.get("authors", [])) != 5:
        raise ValueError("CITATION.cff must contain the five paper authors")

    for path in (root / "README.md", *sorted((root / "docs").glob("*.md"))):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            local_target = target.split("#", 1)[0]
            if (
                not local_target
                or "://" in local_target
                or local_target.startswith("mailto:")
            ):
                continue
            if not (path.parent / local_target).resolve().exists():
                raise ValueError(f"Broken link {target!r} in {path.relative_to(root)}")


def _read_scannable_text(path: Path) -> str | None:
    try:
        if path.name.endswith((".csv.gz", ".json.gz", ".txt.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return handle.read()
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
