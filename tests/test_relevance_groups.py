from moralink.analysis.relevance_groups import (
    aggregate_relevance_groups,
    public_representation_fields,
    split_relevance_groups,
)


def test_split_relevance_groups_uses_number_of_relevant_fables():
    qrels = {
        "q1": {"f1"},
        "q2": {"f2", "f3"},
    }

    assert split_relevance_groups(qrels) == {
        "one": {"q1"},
        "several": {"q2"},
    }


def test_aggregate_relevance_groups_returns_two_rows_per_run():
    rows = [
        {
            "run_id": "run-a",
            "query_id": "q1",
            "subset": "one_relevant",
            "MRR": "0.5",
            "MAP@10": "0.4",
            "Recall@10": "0.6",
        },
        {
            "run_id": "run-a",
            "query_id": "q2",
            "subset": "several_relevant",
            "MRR": "0.25",
            "MAP@10": "0.2",
            "Recall@10": "0.3",
        },
        {
            "run_id": "run-a",
            "query_id": "q3",
            "subset": "several_relevant",
            "MRR": "0.75",
            "MAP@10": "0.6",
            "Recall@10": "0.9",
        },
    ]

    aggregates = aggregate_relevance_groups(rows)

    assert aggregates == [
        {
            "run_id": "run-a",
            "subset": "one_relevant",
            "n_queries": 1,
            "MRR": 0.5,
            "MAP@10": 0.4,
            "Recall@10": 0.6,
        },
        {
            "run_id": "run-a",
            "subset": "several_relevant",
            "n_queries": 2,
            "MRR": 0.5,
            "MAP@10": 0.4,
            "Recall@10": 0.6,
        },
    ]


def test_public_representation_fields_match_frozen_result_labels():
    assert public_representation_fields("fable_only") == ("fable_only", "raw")
    assert public_representation_fields("summary_only") == (
        "summary_only",
        "direct_moral_only",
    )
    assert public_representation_fields("summary_plus_fable") == (
        "summary_plus_fable_summary_first",
        "direct_moral_fable",
    )
