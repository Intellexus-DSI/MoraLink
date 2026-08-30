from pathlib import Path

import torch

from moralink.training import (
    TrainingExample,
    build_same_label_mask,
    labels_to_ids,
    split_by_label,
)


def test_same_label_examples_are_masked_as_false_negatives():
    labels = torch.tensor([10, 10, 20])
    mask = build_same_label_mask(labels)

    assert mask.tolist() == [
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]


def test_split_by_label_never_leaks_a_label_between_splits():
    examples = [
        TrainingExample(query=f"q-{label}-{index}", document="d", label=label)
        for label in ("a", "b", "c", "d")
        for index in range(2)
    ]

    training, validation = split_by_label(examples, validation_ratio=0.25, seed=42)

    assert {example.label for example in training}.isdisjoint(
        {example.label for example in validation}
    )
    assert len(training) + len(validation) == len(examples)


def test_string_labels_map_to_sorted_contiguous_ids():
    assert labels_to_ids(["z", "a", "z", "m"]) == [2, 0, 2, 1]


def test_training_install_message_uses_public_repository_root():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "moralink" / "training.py"
    ).read_text(encoding="utf-8")
    assert "uv sync --extra training" in source
    assert "--project public_release" not in source
