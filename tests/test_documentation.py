import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_release_links_and_current_counts():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "https://huggingface.co/datasets/Intellexus/MoraLink",
        "https://github.com/Intellexus-DSI/MoraLink",
        "709",
        "668",
        "558",
        "Apache-2.0",
        "CC BY-NC 4.0",
    ):
        assert required in text
    for forbidden in (
        "678",
        "0.893",
        "/data/" + "lior",
        "Para" + "beLink",
    ):
        assert forbidden not in text


def test_readme_summarizes_the_main_results_and_names_the_authors():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Main results" in text
    assert "Retrieving original fables" in text
    assert "Generated moral-style summaries" in text
    for family_name in ("Livyatan", "Golan Hashiloni", "Amar", "Rahamim", "Bar"):
        assert family_name in text


def test_all_relative_markdown_links_resolve():
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    missing = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            clean_target = target.split("#", 1)[0]
            if not clean_target or "://" in clean_target or clean_target.startswith("mailto:"):
                continue
            if not (path.parent / clean_target).resolve().exists():
                missing.append((path.name, target))
    assert missing == []


def test_citation_has_final_author_order_and_parses_as_yaml():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["title"] == "MoraLink: Bridging Morals and Narrative Fables for Retrieval"
    assert [author["family-names"] for author in citation["authors"]] == [
        "Livyatan",
        "Golan Hashiloni",
        "Amar",
        "Rahamim",
        "Bar",
    ]
    assert citation["repository-code"] == "https://github.com/Intellexus-DSI/MoraLink"
    assert all(
        "Herzliya, Israel" in author["affiliation"]
        for author in citation["authors"]
    )


def test_reproduction_guide_covers_every_public_rerun_command():
    text = (ROOT / "docs" / "REPRODUCING.md").read_text(encoding="utf-8")
    for required in (
        "scripts/run_zero_shot.py",
        "scripts/run_bm25.py",
        "scripts/run_llm_ranking.py",
        "scripts/evaluate_rankings.py",
        "scripts/run_fine_tuning.py",
        "--extra training",
        "--extra llm",
    ):
        assert required in text


def test_documentation_explains_the_two_metric_definitions():
    text = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert "paper-compatible MRR" in text
    assert "standard MRR@10" in text
    assert "rank 11" in text


def test_official_apache_license_is_included():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "END OF TERMS AND CONDITIONS" in text
