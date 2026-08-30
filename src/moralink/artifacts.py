"""Download and validate MoraLink release artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HF_DATASET_REPO = "Intellexus/MoraLink"


@dataclass(frozen=True)
class ArtifactSpec:
    """Expected location and integrity checks for one released artifact."""

    path: str
    sha256: str | None = None
    count: int | None = None
    count_field: str | None = None


def load_artifact_manifest(path: Path) -> dict[str, ArtifactSpec]:
    """Load the artifact contract from a YAML file."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Artifact manifest must be a mapping: {path}")

    specs: dict[str, ArtifactSpec] = {}
    allowed_fields = {"path", "sha256", "count", "count_field"}
    for name, values in raw.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise TypeError(f"Invalid artifact entry: {name!r}")
        unknown_fields = set(values) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown fields for artifact {name!r}: {sorted(unknown_fields)}"
            )
        if not isinstance(values.get("path"), str):
            raise TypeError(f"Artifact {name!r} must define a string path")
        try:
            specs[name] = ArtifactSpec(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid artifact entry {name!r}: {exc}") from exc
    return specs


def download_release(output_dir: Path, revision: str = "main") -> Path:
    """Download the public MoraLink dataset snapshot from Hugging Face."""

    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir,
    )
    return Path(downloaded_path)


def validate_artifacts(root: Path, manifest_path: Path) -> dict[str, int]:
    """Validate every released artifact against hashes and expected counts."""

    specs = load_artifact_manifest(manifest_path)
    total_items = 0
    for name, spec in specs.items():
        artifact_path = root / spec.path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing artifact {name!r}: {artifact_path}")

        if spec.sha256 is not None:
            actual_hash = _sha256(artifact_path)
            if actual_hash != spec.sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {name!r}: "
                    f"expected {spec.sha256}, got {actual_hash}"
                )

        if spec.count is not None:
            actual_count = _count_items(artifact_path, spec)
            if actual_count != spec.count:
                raise ValueError(
                    f"Count mismatch for {name!r}: "
                    f"expected {spec.count}, got {actual_count}"
                )
            total_items += actual_count

    return {"files": len(specs), "items": total_items}


def validate_checksum_manifest(
    root: Path,
    manifest_name: str = "FILE_HASHES.sha256",
) -> int:
    """Validate the checksum and coverage of every released data file."""

    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest_path}")

    listed: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise ValueError(f"Invalid checksum line {line_number}: {line!r}")
        digest, relative = parts
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or ".cache" in relative_path.parts
        ):
            raise ValueError(f"Unsafe checksum path on line {line_number}: {relative!r}")
        if relative == manifest_name:
            raise ValueError("The checksum manifest cannot list itself")
        if relative in listed:
            raise ValueError(f"Duplicate checksum path: {relative}")
        listed[relative] = digest

    if not listed:
        raise ValueError("The checksum manifest is empty")

    for relative, expected_hash in listed.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing checksummed file: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {relative!r}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    released_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != manifest_name
        and ".cache" not in path.relative_to(root).parts
    }
    unlisted = sorted(released_files - set(listed))
    if unlisted:
        raise ValueError(f"Release files are not listed in {manifest_name}: {unlisted}")
    stale = sorted(set(listed) - released_files)
    if stale:
        raise ValueError(f"Checksum entries do not exist: {stale}")
    return len(listed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_items(path: Path, spec: ArtifactSpec) -> int:
    if path.name.endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    if path.suffix == ".json":
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if spec.count_field is not None:
            if not isinstance(payload, dict) or spec.count_field not in payload:
                raise ValueError(
                    f"JSON artifact {path} does not contain count_field "
                    f"{spec.count_field!r}"
                )
            payload = payload[spec.count_field]
        if not isinstance(payload, (dict, list)):
            raise ValueError(f"Cannot count JSON artifact {path}")
        return len(payload)
    raise ValueError(f"Cannot count unsupported artifact type: {path}")
