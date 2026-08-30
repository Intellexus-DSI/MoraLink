from pathlib import Path

import yaml

from moralink.models import instruction_prefix, load_model_specs

ROOT = Path(__file__).resolve().parents[1]


def test_official_runs_cover_paper_matrix():
    runs = yaml.safe_load(
        (ROOT / "configs" / "official_runs.yaml").read_text(encoding="utf-8")
    )

    assert len(runs["runs"]) == 144
    assert len({row["model_alias"] for row in runs["runs"]}) == 24
    assert {row["instruction_variant"] for row in runs["runs"]} == {
        "no_instr",
        "general",
    }
    assert {row["document_representation"] for row in runs["runs"]} == {
        "fable_only",
        "summary_only",
        "summary_plus_fable",
    }
    forbidden = (
        "/data/",
        "Para" + "beLink",
        "remote_results",
        "metric_artifact",
    )
    serialized = str(runs)
    assert not any(value in serialized for value in forbidden)


def test_model_specs_cover_all_official_models():
    specs = load_model_specs(ROOT / "configs" / "models.yaml")

    assert len(specs) == 24
    assert all(spec.batch_size > 0 for spec in specs.values())
    assert all(spec.trust_remote_code for spec in specs.values())
    assert {alias: spec.parameters for alias, spec in specs.items()} == {
        "all-MiniLM-L6-v2": "22M",
        "all-mpnet-base-v2": "110M",
        "BGE-M3": "568M",
        "Multilingual-E5-large-instruct": "560M",
        "Qwen3-Embedding-0.6B": "596M",
        "Instructor-xl": "1.24B",
        "Lychee-Embed": "1.54B",
        "Qwen3-Embedding-4B": "4.02B",
        "BGE-en-ICL": "7.11B",
        "E5-mistral-7b": "7.11B",
        "Linq-Embed-Mistral": "7.11B",
        "SFR-Embedding-Mistral": "7.11B",
        "Qwen3-Embedding-8B": "7.57B",
        "BGE-base-en-v1.5": "109M",
        "Contriever": "110M",
        "E5-base-v2": "110M",
        "Instructor-base": "110M",
        "TART-dual-contriever": "110M",
        "BGE-large-en-v1.5": "335M",
        "E5-large-v2": "335M",
        "Nomic-Embed-v2-MoE": "475M",
        "Multilingual-E5-large": "560M",
        "DRAMA-1B": "1.24B",
        "Llama-Embed-Nemotron-8B": "7.50B",
    }


def test_query_instruction_is_the_exact_paper_prefix():
    assert instruction_prefix("no_instr") == ""
    assert instruction_prefix("general") == (
        "Instruct: Given a text, retrieve the most relevant passage that answers "
        "the query\nQuery: "
    )
