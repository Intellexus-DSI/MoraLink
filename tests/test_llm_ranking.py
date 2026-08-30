import pytest

from moralink.data import Fable
from moralink.llm_ranking import (
    build_corpus_block,
    parse_ranked_ids,
    render_ranking_prompt,
)


def test_rendered_prompt_contains_query_corpus_and_requested_depth():
    prompt = render_ranking_prompt("Patience is rewarded.", "[f1] story", 10)

    assert "Patience is rewarded." in prompt
    assert "[f1] story" in prompt
    assert "10" in prompt


def test_corpus_block_keeps_public_fable_ids_next_to_documents():
    block = build_corpus_block(
        [
            Fable(doc_id="f1", alias="a1", text="Original one"),
            Fable(doc_id="f2", alias="a2", text="Original two"),
        ],
        ["Document one", "Document two"],
    )

    assert block == "[f1]\nDocument one\n\n[f2]\nDocument two"


def test_parser_accepts_an_object_or_array():
    allowed = {"f1", "f2"}

    assert parse_ranked_ids('{"ids": ["f2", "f1"]}', allowed) == ["f2", "f1"]
    assert parse_ranked_ids('["f1", "f2"]', allowed) == ["f1", "f2"]


def test_parser_rejects_unknown_or_duplicate_ids():
    with pytest.raises(ValueError):
        parse_ranked_ids('{"ids": ["f1", "f1"]}', {"f1", "f2"})
    with pytest.raises(ValueError):
        parse_ranked_ids('{"ids": ["f3"]}', {"f1", "f2"})


def test_parser_rejects_more_than_requested_depth():
    with pytest.raises(ValueError, match="more than 2"):
        parse_ranked_ids('["f1", "f2", "f3"]', {"f1", "f2", "f3"}, top_k=2)
