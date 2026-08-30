"""Fine-tuning utilities for the four MoraLink training strategies."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml


@dataclass(frozen=True)
class TrainingExample:
    """One moral query, positive narrative, and shared moral label."""

    query: str
    document: str
    label: str


@dataclass(frozen=True)
class PreparedStage:
    """A validated training stage with label-disjoint train and validation data."""

    source: str
    training: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]


def build_same_label_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return off-diagonal pairs that express the same moral."""

    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask.fill_diagonal_(False)
    return mask


def labels_to_ids(labels: Sequence[str]) -> list[int]:
    """Map string labels to stable, contiguous integer IDs."""

    mapping = {label: index for index, label in enumerate(sorted(set(labels)))}
    return [mapping[label] for label in labels]


def split_by_label(
    examples: Sequence[TrainingExample],
    validation_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split whole moral groups so a label never appears in both splits."""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        raise ValueError("Cannot split an empty training set")

    grouped: dict[str, list[TrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.label, []).append(example)
    labels = sorted(grouped)
    if len(labels) < 2:
        raise ValueError("At least two moral labels are required for a split")

    rng = random.Random(seed)
    rng.shuffle(labels)
    validation_count = max(1, round(len(labels) * validation_ratio))
    validation_count = min(validation_count, len(labels) - 1)
    validation_labels = set(labels[:validation_count])

    training = [
        example
        for label in labels
        if label not in validation_labels
        for example in grouped[label]
    ]
    validation = [
        example
        for label in labels
        if label in validation_labels
        for example in grouped[label]
    ]
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


class MoralContrastiveLoss(torch.nn.Module):
    """MoraLink's in-batch query-document and query-query contrastive loss."""

    def __init__(self, model: torch.nn.Module, temperature: float = 0.05):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(
        self,
        sentence_features: list[dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(sentence_features) != 2:
            raise ValueError("MoraLink training expects query-positive pairs only")

        query_embeddings = F.normalize(
            self.model(sentence_features[0])["sentence_embedding"], dim=-1
        )
        document_embeddings = F.normalize(
            self.model(sentence_features[1])["sentence_embedding"], dim=-1
        )
        batch_size = query_embeddings.size(0)

        document_scores = (
            torch.mm(query_embeddings, document_embeddings.T) / self.temperature
        )
        if labels is not None:
            document_scores = document_scores.masked_fill(
                build_same_label_mask(labels), float("-inf")
            )

        query_scores = torch.mm(query_embeddings, query_embeddings.T) / self.temperature
        query_mask = torch.eye(
            batch_size, dtype=torch.bool, device=query_embeddings.device
        )
        if labels is not None:
            query_mask |= labels.unsqueeze(0) == labels.unsqueeze(1)
        query_scores = query_scores.masked_fill(query_mask, float("-inf"))

        logits = torch.cat((document_scores, query_scores), dim=1)
        targets = torch.arange(batch_size, device=query_embeddings.device)
        return F.cross_entropy(logits, targets)


def load_storal_examples(path: Path) -> list[TrainingExample]:
    """Load the cleaned STORAL story-moral pairs used by the paper."""

    rows = _load_json_list(path)
    examples = []
    for row in rows:
        story = _nonempty_string(row, "story", path)
        moral = _nonempty_string(row, "moral", path)
        examples.append(
            TrainingExample(
                query=moral,
                document=story,
                label=_normalize_moral(moral),
            )
        )
    return examples


def load_tf1_examples(
    root: Path, fables_per_moral: int = 16
) -> list[TrainingExample]:
    """Load the first 16 low-overlap TF1 fables for each of 95 morals."""

    if fables_per_moral < 1:
        raise ValueError("fables_per_moral must be positive")
    processed = root / "processed"
    morals = _load_json_list(processed / "morals_corpus.json")
    fables = _load_json_list(processed / "fables_corpus.json")
    qrels = _load_json_list(processed / "qrels_moral_to_fable.json")

    moral_text = {
        _nonempty_string(row, "doc_id", processed): _nonempty_string(
            row, "text", processed
        )
        for row in morals
    }
    fable_text = {
        _nonempty_string(row, "doc_id", processed): _nonempty_string(
            row, "text", processed
        )
        for row in fables
    }
    relevant: dict[str, list[str]] = {moral_id: [] for moral_id in moral_text}
    for row in qrels:
        if row.get("relevance", 1) <= 0:
            continue
        moral_id = _nonempty_string(row, "query_id", processed)
        fable_id = _nonempty_string(row, "doc_id", processed)
        if moral_id not in relevant:
            raise ValueError(f"Unknown TF1 moral ID: {moral_id}")
        if fable_id not in fable_text:
            raise ValueError(f"Unknown TF1 fable ID: {fable_id}")
        relevant[moral_id].append(fable_id)

    examples: list[TrainingExample] = []
    for moral_id in sorted(moral_text):
        selected = relevant[moral_id][:fables_per_moral]
        if len(selected) != fables_per_moral:
            raise ValueError(
                f"TF1 moral {moral_id} has {len(selected)} relevant fables; "
                f"expected at least {fables_per_moral}"
            )
        examples.extend(
            TrainingExample(
                query=moral_text[moral_id],
                document=fable_text[fable_id],
                label=moral_id,
            )
            for fable_id in selected
        )
    return examples


def load_training_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a public fine-tuning configuration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML object in {path}")
    for field in (
        "strategy",
        "seeds",
        "epochs",
        "early_stopping_patience",
        "validation_ratio",
        "temperature",
        "stages",
        "models",
    ):
        if field not in config:
            raise ValueError(f"Missing {field!r} in {path}")
    return config


def prepare_stages(
    config: dict[str, Any],
    *,
    seed: int,
    storal_path: Path | None = None,
    tf1_path: Path | None = None,
) -> list[PreparedStage]:
    """Load, count-check, and split all stages in one training strategy."""

    sources = {stage["source"] for stage in config["stages"]}
    needs_storal = bool(sources & {"storal", "mixed"})
    needs_synthetic = bool(sources & {"synthetic", "mixed"})
    if needs_storal and storal_path is None:
        raise ValueError("This strategy requires --storal-path")
    if needs_synthetic and tf1_path is None:
        raise ValueError("This strategy requires --tf1-path")

    storal = load_storal_examples(storal_path) if needs_storal else []
    synthetic = (
        load_tf1_examples(
            tf1_path,
            fables_per_moral=int(config["expected_counts"]["fables_per_moral"]),
        )
        if needs_synthetic
        else []
    )
    _check_expected_counts(config, storal=storal, synthetic=synthetic)

    prepared = []
    for stage in config["stages"]:
        source = stage["source"]
        if source == "storal":
            examples = list(storal)
        elif source == "synthetic":
            examples = list(synthetic)
        elif source == "mixed":
            examples = list(storal) + list(synthetic)
            random.Random(seed).shuffle(examples)
        else:
            raise ValueError(f"Unknown training source: {source}")
        training, validation = split_by_label(
            examples,
            validation_ratio=float(config["validation_ratio"]),
            seed=seed,
        )
        prepared.append(
            PreparedStage(source, tuple(training), tuple(validation))
        )
    return prepared


def dry_run_summary(
    config: dict[str, Any],
    *,
    model_alias: str,
    seed: int,
    storal_path: Path | None = None,
    tf1_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a run and return its resolved settings without loading a model."""

    if model_alias not in config["models"]:
        raise ValueError(f"Unknown model alias: {model_alias}")
    if seed not in config["seeds"]:
        raise ValueError(f"Seed {seed} is not one of the reported seeds")
    stages = prepare_stages(
        config,
        seed=seed,
        storal_path=storal_path,
        tf1_path=tf1_path,
    )
    return {
        "strategy": config["strategy"],
        "model": model_alias,
        "model_id": config["models"][model_alias]["model_id"],
        "seed": seed,
        "temperature": config["temperature"],
        "epochs_per_stage": config["epochs"],
        "stages": [
            {
                "source": stage.source,
                "training_pairs": len(stage.training),
                "validation_pairs": len(stage.validation),
                "training_labels": len({row.label for row in stage.training}),
                "validation_labels": len({row.label for row in stage.validation}),
            }
            for stage in stages
        ],
    }


def train(
    config: dict[str, Any],
    *,
    model_alias: str,
    seed: int,
    output: Path,
    storal_path: Path | None = None,
    tf1_path: Path | None = None,
) -> None:
    """Run one reported fine-tuning configuration and save the final model."""

    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from transformers import EarlyStoppingCallback, set_seed
    except ImportError as error:
        raise RuntimeError("Fine-tuning requires: uv sync --extra training") from error

    if model_alias not in config["models"]:
        raise ValueError(f"Unknown model alias: {model_alias}")
    if seed not in config["seeds"]:
        raise ValueError(f"Seed {seed} is not one of the reported seeds")
    stages = prepare_stages(
        config,
        seed=seed,
        storal_path=storal_path,
        tf1_path=tf1_path,
    )
    model_settings = config["models"][model_alias]
    set_seed(seed)
    model = SentenceTransformer(
        model_settings["model_id"],
        trust_remote_code=model_settings.get("trust_remote_code", False),
    )
    model.max_seq_length = int(model_settings["max_seq_length"])
    if model_settings.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    if "lora" in model_settings:
        model.add_adapter(_lora_config(model_settings["lora"]))

    output.mkdir(parents=True, exist_ok=True)
    for stage_index, stage in enumerate(stages, start=1):
        stage_output = output / f"stage-{stage_index}-{stage.source}"
        label_ids = labels_to_ids(
            [row.label for row in (*stage.training, *stage.validation)]
        )
        training_size = len(stage.training)
        train_dataset = Dataset.from_dict(
            {
                "anchor": [row.query for row in stage.training],
                "positive": [row.document for row in stage.training],
                "label": label_ids[:training_size],
            }
        )
        validation_dataset = Dataset.from_dict(
            {
                "anchor": [row.query for row in stage.validation],
                "positive": [row.document for row in stage.validation],
                "label": label_ids[training_size:],
            }
        )
        arguments = SentenceTransformerTrainingArguments(
            output_dir=str(stage_output),
            num_train_epochs=int(config["epochs"]),
            per_device_train_batch_size=int(model_settings["batch_size"]),
            per_device_eval_batch_size=int(model_settings["batch_size"]),
            gradient_accumulation_steps=int(
                model_settings["gradient_accumulation_steps"]
            ),
            learning_rate=float(model_settings["learning_rate"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            seed=seed,
            data_seed=seed,
            report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            loss=MoralContrastiveLoss(
                model, temperature=float(config["temperature"])
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=int(
                        config["early_stopping_patience"]
                    )
                )
            ],
        )
        trainer.train()
    model.save_pretrained(str(output / "final"))


def _lora_config(settings: dict[str, Any]) -> Any:
    try:
        from peft import LoraConfig, TaskType
    except ImportError as error:
        raise RuntimeError("LoRA fine-tuning requires: uv sync --extra training") from error
    return LoraConfig(
        r=int(settings["r"]),
        lora_alpha=int(settings["alpha"]),
        lora_dropout=float(settings["dropout"]),
        target_modules=settings["target_modules"],
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
    )


def _check_expected_counts(
    config: dict[str, Any],
    *,
    storal: Sequence[TrainingExample],
    synthetic: Sequence[TrainingExample],
) -> None:
    expected = config["expected_counts"]
    if "storal_pairs" in expected and len(storal) != expected["storal_pairs"]:
        raise ValueError(
            f"Expected {expected['storal_pairs']} STORAL pairs, found {len(storal)}"
        )
    if "synthetic_pairs" in expected and len(synthetic) != expected["synthetic_pairs"]:
        raise ValueError(
            f"Expected {expected['synthetic_pairs']} synthetic pairs, "
            f"found {len(synthetic)}"
        )
    if "synthetic_morals" in expected:
        moral_count = len({row.label for row in synthetic})
        if moral_count != expected["synthetic_morals"]:
            raise ValueError(
                f"Expected {expected['synthetic_morals']} synthetic morals, "
                f"found {moral_count}"
            )


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"Expected a list of objects in {path}")
    return payload


def _nonempty_string(row: dict[str, Any], field: str, path: Path) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid {field!r} in {path}")
    return value


def _normalize_moral(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
