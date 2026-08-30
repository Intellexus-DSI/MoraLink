"""Public model and query-instruction configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GENERAL_INSTRUCTION = (
    "Instruct: Given a text, retrieve the most relevant passage that answers "
    "the query\nQuery: "
)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    batch_size: int
    trust_remote_code: bool
    parameters: str | None = None


def load_model_specs(path: Path) -> dict[str, ModelSpec]:
    """Load the 24 official embedding-model configurations."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise TypeError("Model configuration must contain a models list")

    specs: dict[str, ModelSpec] = {}
    for row in payload["models"]:
        if not isinstance(row, dict):
            raise TypeError("Each model configuration must be a mapping")
        spec = ModelSpec(**row)
        if spec.alias in specs:
            raise ValueError(f"Duplicate model alias: {spec.alias}")
        if spec.batch_size < 1:
            raise ValueError(f"Invalid batch size for model: {spec.alias}")
        specs[spec.alias] = spec
    return specs


def instruction_prefix(variant: str) -> str:
    """Return the exact query prefix used by an official instruction variant."""

    if variant == "no_instr":
        return ""
    if variant == "general":
        return GENERAL_INSTRUCTION
    raise ValueError(f"Unknown instruction variant: {variant}")
