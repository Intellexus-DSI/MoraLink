"""Document layouts evaluated in the MoraLink paper."""

from __future__ import annotations

from moralink.data import Fable

TEMPLATES = {
    "fable_only": "{fable}",
    "summary_only": "{direct_moral}",
    "summary_plus_fable": "Moral summary: {direct_moral}\n\n{fable}",
    "fable_plus_summary": "{fable}\n\nMoral summary: {direct_moral}",
}


def build_document(
    fable: Fable,
    summary: dict[str, str] | None,
    representation: str,
) -> str:
    """Build one candidate document using an exact paper layout."""

    if representation not in TEMPLATES:
        raise ValueError(f"Unknown representation: {representation}")
    if representation == "fable_only":
        return fable.text
    if summary is None or not summary.get("direct_moral"):
        raise ValueError(
            f"The {representation} representation requires a direct moral summary"
        )
    return TEMPLATES[representation].format(
        fable=fable.text,
        direct_moral=summary["direct_moral"],
    )
