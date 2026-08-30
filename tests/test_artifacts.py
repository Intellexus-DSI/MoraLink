import gzip
import hashlib
from pathlib import Path

import pytest

from moralink.artifacts import (
    load_artifact_manifest,
    validate_artifacts,
    validate_checksum_manifest,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_validate_artifacts_checks_hash_and_json_count(tmp_path: Path):
    payload = b'[{"id": 1}, {"id": 2}]'
    artifact = tmp_path / "items.json"
    artifact.write_bytes(payload)
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        f"items:\n  path: items.json\n  sha256: {_sha256(payload)}\n  count: 2\n",
        encoding="utf-8",
    )

    specs = load_artifact_manifest(manifest)

    assert specs["items"].count == 2
    assert validate_artifacts(tmp_path, manifest) == {"files": 1, "items": 2}


def test_validate_artifacts_rejects_hash_mismatch(tmp_path: Path):
    (tmp_path / "items.json").write_text("[]", encoding="utf-8")
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        "items:\n  path: items.json\n  sha256: deadbeef\n  count: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_artifacts(tmp_path, manifest)


def test_validate_artifacts_counts_a_named_list_field(tmp_path: Path):
    payload = b'{"rankings": [{"run_id": "a"}, {"run_id": "b"}]}'
    (tmp_path / "manifest.json").write_bytes(payload)
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        "rankings:\n"
        "  path: manifest.json\n"
        f"  sha256: {_sha256(payload)}\n"
        "  count: 2\n"
        "  count_field: rankings\n",
        encoding="utf-8",
    )

    assert validate_artifacts(tmp_path, manifest) == {"files": 1, "items": 2}


def test_validate_artifacts_counts_gzip_csv_rows(tmp_path: Path):
    artifact = tmp_path / "rows.csv.gz"
    with gzip.open(artifact, "wt", encoding="utf-8", newline="") as handle:
        handle.write("run_id,score\na,0.1\nb,0.2\n")
    payload = artifact.read_bytes()
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        "rows:\n"
        "  path: rows.csv.gz\n"
        f"  sha256: {_sha256(payload)}\n"
        "  count: 2\n",
        encoding="utf-8",
    )

    assert validate_artifacts(tmp_path, manifest) == {"files": 1, "items": 2}


def test_validate_artifacts_rejects_missing_count_field(tmp_path: Path):
    payload = b'{"runs": []}'
    (tmp_path / "manifest.json").write_bytes(payload)
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        "rankings:\n"
        "  path: manifest.json\n"
        f"  sha256: {_sha256(payload)}\n"
        "  count: 0\n"
        "  count_field: rankings\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="count_field"):
        validate_artifacts(tmp_path, manifest)


def test_checksum_manifest_covers_and_validates_every_release_file(tmp_path: Path):
    files = {
        "README.md": b"release card\n",
        "rankings/run.json": b'{"rankings": []}\n',
    }
    for relative, payload in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    lines = [f"{_sha256(payload)}  {relative}" for relative, payload in files.items()]
    (tmp_path / "FILE_HASHES.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    assert validate_checksum_manifest(tmp_path) == 2


def test_checksum_manifest_rejects_a_hash_mismatch(tmp_path: Path):
    (tmp_path / "README.md").write_text("release card\n", encoding="utf-8")
    (tmp_path / "FILE_HASHES.sha256").write_text(
        f"{'0' * 64}  README.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_checksum_manifest(tmp_path)


def test_checksum_manifest_rejects_an_unlisted_release_file(tmp_path: Path):
    payload = b"release card\n"
    (tmp_path / "README.md").write_bytes(payload)
    (tmp_path / "unlisted.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "FILE_HASHES.sha256").write_text(
        f"{_sha256(payload)}  README.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not listed"):
        validate_checksum_manifest(tmp_path)
