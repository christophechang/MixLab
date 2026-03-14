from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from mixlab.models import MixConcept, Track

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MINIMAX_URL = "https://api.minimaxi.chat/v1/chat/completions"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

def _concepts_payload(id_offset: int = 0) -> str:
    return json.dumps(
        [
            {
                "title": "Dark Rollers",
                "mood": "heavy and relentless",
                "track_ids": [str(id_offset), str(id_offset + 1), str(id_offset + 2), str(id_offset + 3)],
            },
            {
                "title": "Liquid Dreams",
                "mood": "smooth and atmospheric",
                "track_ids": [str(id_offset + 4), str(id_offset + 5), str(id_offset + 6), str(id_offset + 7)],
            },
        ]
    )


def _chat_response(id_offset: int = 0) -> dict[str, object]:
    return {"choices": [{"message": {"content": _concepts_payload(id_offset)}}]}


def _make_tracks(n: int, genre: str = "Drum & Bass") -> list[Track]:
    return [
        Track(track_id=str(i), artist=f"Artist {i}", title=f"Title {i}", bpm=174.0, camelot_key="8A", genre=genre)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


@respx.mock
async def test_stage1_skips_provider_if_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage1_concepts

    # Only provide Groq key — Minimax should be skipped silently.
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    respx.post(_GROQ_URL).mock(return_value=Response(200, json=_chat_response()))

    concepts = await stage1_concepts(_make_tracks(10), "Drum & Bass")
    assert len(concepts) == 2
    assert concepts[0].title == "Dark Rollers"


@respx.mock
async def test_stage1_falls_through_cascade_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage1_concepts

    monkeypatch.setenv("MINIMAX_API_KEY", "bad-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    respx.post(_MINIMAX_URL).mock(return_value=Response(500, text="error"))
    respx.post(_GROQ_URL).mock(return_value=Response(200, json=_chat_response()))

    concepts = await stage1_concepts(_make_tracks(10), "Drum & Bass")
    assert concepts[0].title == "Dark Rollers"


@respx.mock
async def test_stage1_chunks_large_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage1_concepts

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # 50 tracks → two chunks (40 + 10); each call returns IDs valid for its chunk.
    respx.post(_GROQ_URL).mock(
        side_effect=[
            Response(200, json=_chat_response(id_offset=0)),
            Response(200, json=_chat_response(id_offset=40)),
        ]
    )

    concepts = await stage1_concepts(_make_tracks(50), "Drum & Bass")
    # Two chunks × 2 concepts each = 4 total
    assert len(concepts) == 4


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


@respx.mock
async def test_stage2_raises_loudly_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(401, json={"error": "unauthorized"}))

    concepts = [MixConcept(title="Test", mood="dark", track_ids=["1"])]
    tracks_by_id = {"1": Track(track_id="1", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")}

    with pytest.raises(RuntimeError):
        await stage2_report(concepts, tracks_by_id)


# ---------------------------------------------------------------------------
# Shortfall warning
# ---------------------------------------------------------------------------


def test_shortfall_warning_triggered_below_threshold() -> None:
    from mixlab.llm import _shortfall_warning

    # DnB min is 10 — a concept with 2 tracks is 8 below, which is > 4.
    concept = MixConcept(title="Too Small", mood="dark", track_ids=["1", "2"])
    warning = _shortfall_warning(concept, "Drum & Bass")
    assert warning is not None
    assert "2 tracks found" in warning
    assert "needs" in warning


def test_shortfall_warning_not_triggered_near_minimum() -> None:
    from mixlab.llm import _shortfall_warning

    # DnB min is 10 — 8 tracks is 2 below, which is ≤ 4 (within tolerance).
    concept = MixConcept(title="Nearly There", mood="dark", track_ids=[str(i) for i in range(8)])
    warning = _shortfall_warning(concept, "Drum & Bass")
    assert warning is None


def test_shortfall_warning_not_triggered_at_minimum() -> None:
    from mixlab.llm import _shortfall_warning

    concept = MixConcept(title="Full Set", mood="dark", track_ids=[str(i) for i in range(10)])
    warning = _shortfall_warning(concept, "Drum & Bass")
    assert warning is None


# ---------------------------------------------------------------------------
# Edge: all providers missing
# ---------------------------------------------------------------------------


async def test_stage1_raises_if_all_providers_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage1_concepts

    for key in ("MINIMAX_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        await stage1_concepts(_make_tracks(3), "Drum & Bass")


async def test_stage2_raises_if_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_report

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    concepts = [MixConcept(title="Test", mood="dark", track_ids=["1"])]
    tracks_by_id: dict[str, Track] = {}

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await stage2_report(concepts, tracks_by_id)
