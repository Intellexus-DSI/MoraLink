import pytest

from moralink.analysis.word_overlap import (
    aggregate_rank_change_correlations,
    content_iou,
    spearman_correlation,
    tokenize_content,
)


def test_content_iou_is_case_insensitive_and_removes_stop_words():
    assert tokenize_content("The Fox and the Grapes") == {"fox", "grapes"}
    assert content_iou("Fox seeks grapes", "Grapes escape fox") == 2 / 4


def test_spearman_correlation_uses_average_ranks_for_ties():
    assert spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)


def test_rank_change_rows_require_a_matching_raw_model_and_instruction():
    rows = [
        {
            "run_id": "model-a__general__summary-run",
            "model_alias": "model-a",
            "instruction_variant": "general",
            "summary_generator": "gemini",
            "prompt_variant": "direct_moral",
            "query_id": "q1",
            "overlap_improvement": "0.2",
            "reciprocal_rank_improvement": "0.1",
            "rank_improvement": "2",
        },
        {
            "run_id": "model-a__general__summary-run",
            "model_alias": "model-a",
            "instruction_variant": "general",
            "summary_generator": "gemini",
            "prompt_variant": "direct_moral",
            "query_id": "q2",
            "overlap_improvement": "0.4",
            "reciprocal_rank_improvement": "0.3",
            "rank_improvement": "4",
        },
    ]
    official_runs = [
        {
            "run_id": "raw-run",
            "model_alias": "model-a",
            "instruction_variant": "no_instr",
            "document_representation": "fable_only",
        }
    ]

    with pytest.raises(ValueError, match="paired raw run"):
        aggregate_rank_change_correlations(rows, official_runs)


def test_rank_change_rows_produce_one_correlation_per_run():
    rows = [
        {
            "run_id": "model-a__general__summary-run",
            "model_alias": "model-a",
            "instruction_variant": "general",
            "summary_generator": "gemini",
            "prompt_variant": "direct_moral",
            "query_id": "q1",
            "overlap_improvement": "0.2",
            "reciprocal_rank_improvement": "0.1",
            "rank_improvement": "2",
        },
        {
            "run_id": "model-a__general__summary-run",
            "model_alias": "model-a",
            "instruction_variant": "general",
            "summary_generator": "gemini",
            "prompt_variant": "direct_moral",
            "query_id": "q2",
            "overlap_improvement": "0.4",
            "reciprocal_rank_improvement": "0.3",
            "rank_improvement": "4",
        },
    ]
    official_runs = [
        {
            "run_id": "raw-run",
            "model_alias": "model-a",
            "instruction_variant": "general",
            "document_representation": "fable_only",
        }
    ]

    result = aggregate_rank_change_correlations(rows, official_runs)

    assert len(result) == 1
    assert result[0]["n_queries"] == 2
    assert result[0]["spearman_overlap_vs_reciprocal_rank_improvement"] == pytest.approx(
        1.0
    )


def test_rank_change_correlations_use_frozen_table_order():
    rows = []
    for generator, prompt in (
        ("gemma4-E2B", "direct_moral"),
        ("gemini", "cot_proverb"),
        ("gemini", "conceptual_abstract"),
    ):
        run_id = f"model-a__general__{generator}__{prompt}"
        for index in range(2):
            rows.append(
                {
                    "run_id": run_id,
                    "model_alias": "model-a",
                    "instruction_variant": "general",
                    "summary_generator": generator,
                    "prompt_variant": prompt,
                    "query_id": f"q{index}",
                    "overlap_improvement": str(index),
                    "reciprocal_rank_improvement": str(index),
                    "rank_improvement": str(index),
                }
            )
    official_runs = [
        {
            "run_id": "raw-run",
            "model_alias": "model-a",
            "instruction_variant": "general",
            "document_representation": "fable_only",
        }
    ]

    result = aggregate_rank_change_correlations(rows, official_runs)

    assert [
        (row["summary_generator"], row["prompt_variant"]) for row in result
    ] == [
        ("gemini", "conceptual_abstract"),
        ("gemini", "cot_proverb"),
        ("gemma4-E2B", "direct_moral"),
    ]
