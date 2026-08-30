from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "fine_tuning"
CONFIG_NAMES = ("storal", "synthetic", "curriculum", "mixed")
EXPECTED_MODELS = {
    "MiniLM-L6",
    "BGE-M3",
    "Qwen3-0.6B",
    "Instructor-XL",
    "Qwen3-4B",
}


def test_fine_tuning_configs_preserve_the_reported_common_settings():
    for name in CONFIG_NAMES:
        config = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
        assert config["strategy"] == name
        assert config["seeds"] == [42, 43, 44]
        assert config["epochs"] == 10
        assert config["early_stopping_patience"] == 3
        assert config["validation_ratio"] == 0.10
        assert config["temperature"] == 0.05
        assert set(config["models"]) == EXPECTED_MODELS


def test_fine_tuning_configs_have_exact_source_counts_and_no_private_paths():
    configs = {
        name: yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
        for name in CONFIG_NAMES
    }

    assert configs["storal"]["expected_counts"] == {"storal_pairs": 1595}
    assert configs["synthetic"]["expected_counts"] == {
        "synthetic_pairs": 1520,
        "synthetic_morals": 95,
        "fables_per_moral": 16,
    }
    assert [stage["source"] for stage in configs["curriculum"]["stages"]] == [
        "synthetic",
        "storal",
    ]
    assert configs["mixed"]["stages"] == [{"source": "mixed"}]
    serialized = str(configs)
    assert "/data/" not in serialized
    assert "Para" + "beLink" not in serialized
    assert "model_output_dir" not in serialized
