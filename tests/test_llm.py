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


def _shortlist_payload(id_offset: int = 0) -> str:
    """Stage 1 response — candidate pools, each with ≥8 track IDs."""
    return json.dumps(
        [
            {
                "title": "Deep 122 BPM / 4A–7A Pool",
                "mood": "heavy and relentless",
                "track_ids": [str(id_offset + i) for i in range(9)],
            },
            {
                "title": "Liquid 124 BPM / 8A–11A Pool",
                "mood": "smooth and atmospheric",
                "track_ids": [str(id_offset + 9 + i) for i in range(8)],
            },
        ]
    )


def _curated_payload() -> str:
    """Stage 2 response — curated concepts with embedded report."""
    return json.dumps(
        [
            {
                "title": "Dark Rollers",
                "mood": "heavy and relentless",
                "track_ids": ["1", "2", "3", "4"],
                "report": "CONCEPT: Dark Rollers\n\nA relentless journey.\n\nTrack order (Camelot / BPM):\nArtist 1 — Title 1 [8A · 174.0 BPM]\n\nArc: Builds relentlessly.\n\nOpener: Sets a dark tone.\n\nCloser: Resolves with weight.\n\nStandout transitions or calculated risks: The 8A to 9A move is deliberate.\n\nAssumptions: Energy levels inferred from BPM and key metadata.",
            }
        ]
    )


def _chat_response(id_offset: int = 0) -> dict[str, object]:
    return {"choices": [{"message": {"content": _shortlist_payload(id_offset)}}]}


def _anthropic_response(content: str) -> dict[str, object]:
    return {"content": [{"text": content}]}


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

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    respx.post(_GROQ_URL).mock(return_value=Response(200, json=_chat_response()))

    shortlists = await stage1_concepts(_make_tracks(20), "Drum & Bass")
    assert len(shortlists) == 2
    assert shortlists[0].title == "Deep 122 BPM / 4A–7A Pool"


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

    shortlists = await stage1_concepts(_make_tracks(20), "Drum & Bass")
    assert shortlists[0].title == "Deep 122 BPM / 4A–7A Pool"


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

    shortlists = await stage1_concepts(_make_tracks(50), "Drum & Bass")
    # Chunk 1 (tracks 0–39): both pools have valid IDs → 2 shortlists.
    # Chunk 2 (tracks 40–49): only the first pool (IDs 40–48) fits within the chunk;
    # the second pool starts at ID 49 and only "49" is valid → filtered below _MIN_SHORTLIST_TRACKS → 1 shortlist.
    assert len(shortlists) == 3


@respx.mock
async def test_stage1_filters_pools_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage1_concepts

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Pool with only 3 valid IDs (below _MIN_SHORTLIST_TRACKS=8) should be dropped.
    tiny_pool = json.dumps([{"title": "Tiny", "mood": "x", "track_ids": ["0", "1", "2"]}])
    respx.post(_GROQ_URL).mock(return_value=Response(200, json={"choices": [{"message": {"content": tiny_pool}}]}))

    shortlists = await stage1_concepts(_make_tracks(10), "Drum & Bass")
    assert shortlists == []


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


@respx.mock
async def test_stage2_returns_curated_concepts_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(
            track_id=str(i), artist=f"Artist {i}", title=f"Title {i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass"
        )
        for i in range(1, 5)
    }

    concepts, report = await stage2_curate_and_report(shortlists, tracks_by_id)

    assert len(concepts) == 1
    assert concepts[0].title == "Dark Rollers"
    assert concepts[0].track_ids == ["1", "2", "3", "4"]
    assert "CONCEPT: Dark Rollers" in report


@respx.mock
async def test_stage2_strips_hallucinated_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "T",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4", "999"],
                "report": "CONCEPT: T\n\nBrief.\n\nTrack order (Camelot / BPM):\n\nArc: x\n\nOpener: x\n\nCloser: x\n\nStandout transitions or calculated risks: x\n\nAssumptions: x",
            }
        ]
    )
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(payload)))

    shortlists = [MixConcept(title="Pool", mood="m", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    concepts, _ = await stage2_curate_and_report(shortlists, tracks_by_id)
    assert "999" not in concepts[0].track_ids


@respx.mock
async def test_stage2_raises_loudly_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(401, json={"error": "unauthorized"}))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1"])]
    tracks_by_id = {"1": Track(track_id="1", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")}

    with pytest.raises(RuntimeError):
        await stage2_curate_and_report(shortlists, tracks_by_id)


async def test_stage2_raises_if_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1"])]
    tracks_by_id: dict[str, Track] = {}

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await stage2_curate_and_report(shortlists, tracks_by_id)


# ---------------------------------------------------------------------------
# _parse_curated_concepts
# ---------------------------------------------------------------------------


def test_parse_curated_concepts_extracts_concepts_and_report() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps(
        [{"title": "Set A", "mood": "dark", "track_ids": ["1", "2", "3", "4"], "report": "The report text."}]
    )
    valid_ids = {"1", "2", "3", "4"}
    concepts, report = _parse_curated_concepts(raw, valid_ids)

    assert len(concepts) == 1
    assert concepts[0].title == "Set A"
    assert concepts[0].track_ids == ["1", "2", "3", "4"]
    assert report == "The report text."


def test_parse_curated_concepts_strips_hallucinated_ids() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps([{"title": "T", "mood": "m", "track_ids": ["1", "999", "2", "3", "4"], "report": "x"}])
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert "999" not in concepts[0].track_ids


def test_parse_curated_concepts_drops_below_minimum() -> None:
    from mixlab.llm import _parse_curated_concepts

    # Only 2 valid IDs after filtering — below _MIN_CONCEPT_TRACKS=4, should be dropped.
    raw = json.dumps([{"title": "T", "mood": "m", "track_ids": ["1", "2"], "report": "x"}])
    concepts, report = _parse_curated_concepts(raw, {"1", "2"})
    assert concepts == []
    assert report == ""


def test_parse_curated_concepts_joins_multiple_reports() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps(
        [
            {"title": "A", "mood": "m", "track_ids": ["1", "2", "3", "4"], "report": "Report A."},
            {"title": "B", "mood": "m", "track_ids": ["5", "6", "7", "8"], "report": "Report B."},
        ]
    )
    valid_ids = {"1", "2", "3", "4", "5", "6", "7", "8"}
    concepts, report = _parse_curated_concepts(raw, valid_ids)

    assert len(concepts) == 2
    assert "Report A." in report
    assert "Report B." in report
    assert "---" in report  # separator between concepts


def test_parse_curated_concepts_handles_markdown_fences() -> None:
    from mixlab.llm import _parse_curated_concepts

    inner = json.dumps([{"title": "T", "mood": "m", "track_ids": ["1", "2", "3", "4"], "report": "x"}])
    raw = f"```json\n{inner}\n```"
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert len(concepts) == 1


# ---------------------------------------------------------------------------
# Shortfall warning
# ---------------------------------------------------------------------------


def test_shortfall_warning_triggered_below_threshold() -> None:
    from mixlab.llm import _shortfall_warning

    concept = MixConcept(title="Too Small", mood="dark", track_ids=["1", "2"])
    warning = _shortfall_warning(concept, "Drum & Bass")
    assert warning is not None
    assert "2 tracks found" in warning
    assert "needs" in warning


def test_shortfall_warning_not_triggered_near_minimum() -> None:
    from mixlab.llm import _shortfall_warning

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
