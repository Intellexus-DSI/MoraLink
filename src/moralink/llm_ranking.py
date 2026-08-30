"""Optional direct LLM ranking prompt, parser, and provider adapters."""

from __future__ import annotations

import json
import os
from typing import Any

from moralink.data import Fable

SYSTEM_PROMPT = "You are a retrieval system. Return only the requested JSON."
USER_TEMPLATE = """Moral: {moral}

Corpus:
{corpus}

Return a JSON array of the {top_k} fable IDs most relevant to this moral, ranked by relevance. IDs are in the format fable_XXXX."""


def build_corpus_block(fables: list[Fable], documents: list[str]) -> str:
    """Render the complete candidate corpus with stable public fable IDs."""

    if len(fables) != len(documents):
        raise ValueError("The number of fables and documents must match")
    return "\n\n".join(
        f"[{fable.doc_id}]\n{document}"
        for fable, document in zip(fables, documents, strict=True)
    )


def render_ranking_prompt(moral: str, corpus: str, top_k: int = 10) -> str:
    """Render the exact minimal direct-ranking user prompt."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    return USER_TEMPLATE.format(moral=moral, corpus=corpus, top_k=top_k)


def parse_ranked_ids(
    payload: str,
    allowed_ids: set[str],
    top_k: int = 10,
) -> list[str]:
    """Strictly parse a JSON array or an object containing an ``ids`` array."""

    text = payload.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON") from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("ids")
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError("LLM response must be a JSON array of fable IDs")
    if len(parsed) > top_k:
        raise ValueError(f"LLM response contains more than {top_k} IDs")
    if len(set(parsed)) != len(parsed):
        raise ValueError("LLM response contains duplicate fable IDs")
    unknown = set(parsed) - allowed_ids
    if unknown:
        raise ValueError(f"LLM response contains unknown fable IDs: {sorted(unknown)}")
    return parsed


def request_ranking(provider: str, model: str, user_prompt: str) -> str:
    """Request one ranking, reading only the selected provider's API key."""

    if provider == "openai":
        return _request_openai(model, user_prompt)
    if provider == "google":
        return _request_google(model, user_prompt)
    raise ValueError(f"Unknown provider: {provider}")


def _request_openai(model: str, user_prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI provider")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )
    return response.output_text


def _request_google(model: str, user_prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for the Google provider")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        raise ValueError("Google returned an empty response")
    return response.text
