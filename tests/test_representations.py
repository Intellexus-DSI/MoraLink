import pytest

from moralink.data import Fable
from moralink.representations import build_document

FABLE = Fable(doc_id="fable_0001", alias="aesop_1", text="A fox saw grapes.")
SUMMARY = {"direct_moral": "Do not despise what you cannot have."}


def test_document_representations_match_paper_layouts():
    assert build_document(FABLE, SUMMARY, "fable_only") == FABLE.text
    assert build_document(FABLE, SUMMARY, "summary_only") == SUMMARY["direct_moral"]
    assert build_document(FABLE, SUMMARY, "summary_plus_fable") == (
        "Moral summary: Do not despise what you cannot have.\n\n"
        "A fox saw grapes."
    )
    assert build_document(FABLE, SUMMARY, "fable_plus_summary") == (
        "A fox saw grapes.\n\n"
        "Moral summary: Do not despise what you cannot have."
    )


def test_fable_only_does_not_require_a_summary():
    assert build_document(FABLE, None, "fable_only") == FABLE.text


def test_summary_layout_rejects_a_missing_summary():
    with pytest.raises(ValueError, match="summary"):
        build_document(FABLE, None, "summary_only")


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="Unknown representation"):
        build_document(FABLE, SUMMARY, "unknown")
