from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from mixlab.models import MixConcept, Track

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MINIMAX_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
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
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    respx.post(_GROQ_URL).mock(return_value=Response(200, json=_chat_response()))

    shortlists = await stage1_concepts(_make_tracks(20), "Drum & Bass", make_cascade_state())
    assert len(shortlists) == 2
    assert shortlists[0].title == "Deep 122 BPM / 4A–7A Pool"


@respx.mock
async def test_stage1_falls_through_cascade_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("MINIMAX_API_KEY", "bad-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    respx.post(_MINIMAX_URL).mock(return_value=Response(500, text="error"))
    respx.post(_GROQ_URL).mock(return_value=Response(200, json=_chat_response()))

    shortlists = await stage1_concepts(_make_tracks(20), "Drum & Bass", make_cascade_state())
    assert shortlists[0].title == "Deep 122 BPM / 4A–7A Pool"


@respx.mock
async def test_stage1_chunks_large_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 50 tracks → two chunks (40 + 10); each call returns IDs valid for its chunk.
    respx.post(_GROQ_URL).mock(
        side_effect=[
            Response(200, json=_chat_response(id_offset=0)),
            Response(200, json=_chat_response(id_offset=40)),
        ]
    )

    shortlists = await stage1_concepts(_make_tracks(50), "Drum & Bass", make_cascade_state())
    # Chunk 1 (tracks 0–39): both pools have valid IDs → 2 shortlists.
    # Chunk 2 (tracks 40–49): only the first pool (IDs 40–48) fits within the chunk;
    # the second pool starts at ID 49 and only "49" is valid → filtered below _MIN_SHORTLIST_TRACKS → 1 shortlist.
    assert len(shortlists) == 3


@respx.mock
async def test_stage1_filters_pools_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Pool with only 3 valid IDs (below _MIN_SHORTLIST_TRACKS=8) should be dropped.
    tiny_pool = json.dumps([{"title": "Tiny", "mood": "x", "track_ids": ["0", "1", "2"]}])
    respx.post(_GROQ_URL).mock(return_value=Response(200, json={"choices": [{"message": {"content": tiny_pool}}]}))

    shortlists = await stage1_concepts(_make_tracks(10), "Drum & Bass", make_cascade_state())
    assert shortlists == []


@respx.mock
async def test_stage1_parses_response_with_trailing_prose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models that append explanatory text after the JSON array should still parse correctly."""
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    payload_with_prose = _shortlist_payload() + "\n\nNote: grouped by BPM proximity and harmonic compatibility."
    respx.post(_GROQ_URL).mock(
        return_value=Response(200, json={"choices": [{"message": {"content": payload_with_prose}}]})
    )

    shortlists = await stage1_concepts(_make_tracks(20), "Drum & Bass", make_cascade_state())
    assert len(shortlists) == 2


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
    assert "Claude Sonnet 4.6" in report


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
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(401, json={"error": "unauthorized"}))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1"])]
    tracks_by_id = {"1": Track(track_id="1", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")}

    with pytest.raises(RuntimeError):
        await stage2_curate_and_report(shortlists, tracks_by_id)


@respx.mock
async def test_stage2_falls_back_to_minimax_on_anthropic_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(402, json={"error": "credit_limit"}))
    respx.post(_MINIMAX_URL).mock(
        return_value=Response(200, json={"choices": [{"message": {"content": _curated_payload()}}]})
    )

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
    assert "MiniMax M2.7 (Anthropic fallback)" in report


async def test_stage2_raises_if_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("STAGE2_PROVIDER", raising=False)
    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1"])]
    tracks_by_id: dict[str, Track] = {}

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await stage2_curate_and_report(shortlists, tracks_by_id)


@respx.mock
async def test_stage2_uses_minimax_when_stage2_provider_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("STAGE2_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    respx.post(_MINIMAX_URL).mock(
        return_value=Response(200, json={"choices": [{"message": {"content": _curated_payload()}}]})
    )

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
    assert "CONCEPT: Dark Rollers" in report
    assert "MiniMax M2.7" in report


async def test_stage2_raises_if_minimax_key_missing_when_provider_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("STAGE2_PROVIDER", "minimax")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1"])]
    tracks_by_id: dict[str, Track] = {}

    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
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


def test_parse_curated_concepts_repairs_literal_newlines_in_strings() -> None:
    from mixlab.llm import _parse_curated_concepts

    # Simulate model outputting literal newlines inside a JSON string value.
    broken = '[{"title":"T","mood":"m","track_ids":["1","2","3","4"],"report":"line one\nline two\nline three"}]'
    concepts, report = _parse_curated_concepts(broken, {"1", "2", "3", "4"})
    assert len(concepts) == 1
    assert "line one" in report
    assert "line two" in report


# ---------------------------------------------------------------------------
# _extract_complete_objects — truncation recovery
# ---------------------------------------------------------------------------


def test_extract_complete_objects_returns_complete_objects_from_truncated_array() -> None:
    from mixlab.llm import _extract_complete_objects

    # Two complete objects followed by a truncated third — simulates a thinking model
    # that exhausted its token budget mid-output.
    truncated = (
        '[{"title": "Pool 1", "mood": "deep", "track_ids": ["1", "2", "3"]},'
        '{"title": "Pool 2", "mood": "airy", "track_ids": ["4", "5", "6"]},'
        '{"title": "Pool 3", "mood": "dark", "track_ids": ["7'  # truncated here
    )
    objects = _extract_complete_objects(truncated)
    assert len(objects) == 2
    assert objects[0]["title"] == "Pool 1"
    assert objects[1]["title"] == "Pool 2"


def test_extract_complete_objects_raises_when_no_complete_objects() -> None:
    from mixlab.llm import _extract_complete_objects

    # Truncated before even completing the first object.
    truncated = '[{"title": "Pool 1", "mood": "deep", "track_ids": ["1", "2'
    with pytest.raises(ValueError, match="No complete JSON objects"):
        _extract_complete_objects(truncated)


def test_extract_complete_objects_raises_when_no_array_start() -> None:
    from mixlab.llm import _extract_complete_objects

    with pytest.raises(ValueError, match="No JSON array start"):
        _extract_complete_objects("not json at all")


def test_parse_concepts_recovers_from_truncated_json() -> None:
    """_parse_concepts must salvage complete shortlists when the outer array is unclosed."""
    from mixlab.llm import _parse_concepts

    # One complete object + one truncated — outer "]" is missing.
    truncated = (
        '[{"title": "Pool 1", "mood": "deep", "track_ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]},'
        '{"title": "Pool 2", "mood": "airy", "track_ids": ["10'  # truncated
    )
    concepts = _parse_concepts(truncated)
    assert len(concepts) == 1
    assert concepts[0].title == "Pool 1"
    assert len(concepts[0].track_ids) == 9


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
    from mixlab.llm import make_cascade_state, stage1_concepts

    for key in ("MINIMAX_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="All Stage 1 providers exhausted"):
        await stage1_concepts(_make_tracks(3), "Drum & Bass", make_cascade_state())


# ---------------------------------------------------------------------------
# Sticky cascade behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_stage1_sticky_stays_on_successful_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index stays at 0 when Groq (now first) succeeds on both chunks of a large cluster."""
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    respx.post(_GROQ_URL).mock(
        side_effect=[
            Response(200, json=_chat_response(id_offset=0)),
            Response(200, json=_chat_response(id_offset=40)),
        ]
    )

    state = make_cascade_state()
    await stage1_concepts(_make_tracks(50), "Drum & Bass", state)

    assert state.index == 0  # sticky on Groq (index 0)
    assert state.consecutive_failures == 0


@respx.mock
async def test_stage1_advances_on_failure_and_stays_on_next(monkeypatch: pytest.MonkeyPatch) -> None:
    """Groq fails on chunk 1 → Gemini takes over → index stays on Gemini for chunk 2."""
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    respx.post(_GROQ_URL).mock(return_value=Response(500, text="error"))
    respx.post(_GEMINI_URL).mock(
        side_effect=[
            Response(200, json=_chat_response(id_offset=0)),
            Response(200, json=_chat_response(id_offset=40)),
        ]
    )

    state = make_cascade_state()
    await stage1_concepts(_make_tracks(50), "Drum & Bass", state)

    assert state.index == 1  # sticky on Gemini after Groq failed
    assert state.consecutive_failures == 0


@respx.mock
async def test_stage1_skips_unconfigured_without_counting_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """None return (unconfigured provider) advances index but does not count as a failure."""
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    respx.post(_GEMINI_URL).mock(return_value=Response(200, json=_chat_response()))

    state = make_cascade_state()
    await stage1_concepts(_make_tracks(20), "Drum & Bass", state)

    assert state.consecutive_failures == 0
    assert state.index == 1  # advanced past unconfigured Groq to Gemini (index 1)


@respx.mock
async def test_stage1_raises_after_n_consecutive_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four providers fail → RuntimeError after len(_CASCADE) consecutive failures."""
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("MISTRAL_API_KEY", "key")
    monkeypatch.setenv("MINIMAX_API_KEY", "key")

    respx.post(_GROQ_URL).mock(return_value=Response(500, text="error"))
    respx.post(_GEMINI_URL).mock(return_value=Response(500, text="error"))
    respx.post(_MISTRAL_URL).mock(return_value=Response(500, text="error"))
    respx.post(_MINIMAX_URL).mock(return_value=Response(500, text="error"))

    with pytest.raises(RuntimeError, match="consecutive failures"):
        await stage1_concepts(_make_tracks(20), "Drum & Bass", make_cascade_state())


@respx.mock
async def test_stage1_consecutive_failures_reset_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failures before the first success are cleared; state ends clean on Gemini."""
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    # Chunk 1: Groq fails → Gemini succeeds (consecutive_failures reset to 0).
    # Chunk 2: Gemini is sticky and succeeds again.
    respx.post(_GROQ_URL).mock(return_value=Response(500, text="error"))
    respx.post(_GEMINI_URL).mock(
        side_effect=[
            Response(200, json=_chat_response(id_offset=0)),
            Response(200, json=_chat_response(id_offset=40)),
        ]
    )

    state = make_cascade_state()
    await stage1_concepts(_make_tracks(50), "Drum & Bass", state)

    assert state.consecutive_failures == 0
    assert state.index == 1  # sticky on Gemini


# ---------------------------------------------------------------------------
# _tracks_to_text — Stage 1 track line formatting
# ---------------------------------------------------------------------------


def test_tracks_to_text_basic_line() -> None:
    from mixlab.llm import _tracks_to_text

    tracks = [
        Track(track_id="1", artist="Photek", title="Ni Ten Ichi Ryu", bpm=174.0, camelot_key="10A", genre="Drum & Bass")
    ]
    result = _tracks_to_text(tracks)
    assert result == "ID:1 | Photek — Ni Ten Ichi Ryu | 174.0 BPM | 10A"


def test_tracks_to_text_includes_year_when_present() -> None:
    from mixlab.llm import _tracks_to_text

    tracks = [
        Track(
            track_id="1",
            artist="Photek",
            title="Ni Ten Ichi Ryu",
            bpm=174.0,
            camelot_key="10A",
            genre="Drum & Bass",
            year=1997,
        )
    ]
    result = _tracks_to_text(tracks)
    assert "| 1997" in result


def test_tracks_to_text_energy_scale_is_8() -> None:
    from mixlab.llm import _tracks_to_text

    tracks = [Track(track_id="1", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass", energy=7)]
    result = _tracks_to_text(tracks)
    assert "energy:7/8" in result
    assert "energy:7/5" not in result


def test_tracks_to_text_omits_year_when_absent() -> None:
    from mixlab.llm import _tracks_to_text

    tracks = [Track(track_id="1", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")]
    result = _tracks_to_text(tracks)
    # Should not contain a bare 4-digit year segment
    parts = result.split(" | ")
    assert len(parts) == 4  # ID, artist—title, BPM, key — nothing extra


# ---------------------------------------------------------------------------
# Stage 2 — enriched track line in sent prompt
# ---------------------------------------------------------------------------


@respx.mock
async def test_stage2_prompt_includes_enriched_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify enrichment fields appear in the prompt body sent to the API."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        "1": Track(
            track_id="1",
            artist="Photek",
            title="Ni Ten Ichi Ryu",
            bpm=174.0,
            camelot_key="10A",
            genre="Drum & Bass",
            year=1997,
            label="Science",
            album="Modus Operandi",
            remixer="Mafia Kiss",
            mix=["Drum n Bass", "Jungle"],
            energy=7,
            play_count=0,
            enrichment_confidence="high",
        ),
        "2": Track(track_id="2", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass"),
        "3": Track(track_id="3", artist="B", title="U", bpm=174.0, camelot_key="9A", genre="Drum & Bass"),
        "4": Track(track_id="4", artist="C", title="V", bpm=174.0, camelot_key="11A", genre="Drum & Bass"),
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    sent_body = route.calls[0].request.content.decode()
    assert "1997" in sent_body
    assert "Science" in sent_body
    assert "remix by Mafia Kiss" in sent_body
    assert "mix:Drum n Bass, Jungle" in sent_body
    assert "energy:7/8" in sent_body
    assert "unplayed" in sent_body


@respx.mock
async def test_stage2_prompt_includes_unverified_flag_for_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(
            track_id=str(i),
            artist=f"A{i}",
            title=f"T{i}",
            bpm=174.0,
            camelot_key="8A",
            genre="Drum & Bass",
            label="Bootleg",
            enrichment_confidence="low",
        )
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    sent_body = route.calls[0].request.content.decode()
    assert "[unverified]" in sent_body


@respx.mock
async def test_stage2_prompt_omits_unverified_flag_for_high_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(
            track_id=str(i),
            artist=f"A{i}",
            title=f"T{i}",
            bpm=174.0,
            camelot_key="8A",
            genre="Drum & Bass",
            enrichment_confidence="high",
        )
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "[unverified]" not in user_prompt


@respx.mock
async def test_stage2_unenriched_track_produces_clean_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Track with no enrichment data must not produce extra separators or empty fields."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    # No empty fields or junk separators
    assert "| |" not in user_prompt
    assert "[unverified]" not in user_prompt


# ---------------------------------------------------------------------------
# select_stage1_window — random contiguous window for custom genre pools
# ---------------------------------------------------------------------------


def test_select_stage1_window_returns_all_when_at_or_below_max() -> None:
    from mixlab.llm import select_stage1_window

    tracks = _make_tracks(10)
    result = select_stage1_window(tracks, 120)
    assert result == tracks


def test_select_stage1_window_limits_to_max_count() -> None:
    from mixlab.llm import select_stage1_window

    tracks = _make_tracks(200)
    result = select_stage1_window(tracks, 120)
    assert len(result) == 120


def test_select_stage1_window_returns_contiguous_slice() -> None:
    from mixlab.llm import select_stage1_window

    tracks = _make_tracks(200)
    result = select_stage1_window(tracks, 120)
    # The returned slice must be a contiguous subsequence of the input.
    result_ids = [t.track_id for t in result]
    all_ids = [t.track_id for t in tracks]
    start = all_ids.index(result_ids[0])
    assert all_ids[start : start + 120] == result_ids


def test_select_stage1_window_varies_across_runs() -> None:
    from mixlab.llm import select_stage1_window

    tracks = _make_tracks(500)
    starts = {select_stage1_window(tracks, 60)[0].track_id for _ in range(30)}
    assert len(starts) > 1  # different starting positions across runs


# ---------------------------------------------------------------------------
# select_shortlists_for_stage2 — randomised selection
# ---------------------------------------------------------------------------


def test_select_shortlists_returns_all_when_at_or_below_cap() -> None:
    from mixlab.llm import select_shortlists_for_stage2

    shortlists = [MixConcept(title=str(i), mood="m", track_ids=[str(i)] * (5 + i)) for i in range(6)]
    result = select_shortlists_for_stage2(shortlists)
    assert len(result) == 6


def test_select_shortlists_caps_at_stage2_cap() -> None:
    from mixlab.llm import select_shortlists_for_stage2

    shortlists = [MixConcept(title=str(i), mood="m", track_ids=[str(i)] * (5 + i)) for i in range(15)]
    result = select_shortlists_for_stage2(shortlists)
    assert len(result) == 6


def test_select_shortlists_samples_from_top_candidates() -> None:
    """Every returned shortlist must be one of the _STAGE2_CANDIDATE_POOL (12) largest."""
    from mixlab.llm import _STAGE2_CANDIDATE_POOL, select_shortlists_for_stage2

    # 20 shortlists with sizes 1..20
    shortlists = [MixConcept(title=str(i), mood="m", track_ids=[str(i)] * (i + 1)) for i in range(20)]
    # Top 12 by size are the last 12 (indices 8–19, sizes 9–20)
    top_titles = {str(i) for i in range(20 - _STAGE2_CANDIDATE_POOL, 20)}
    result = select_shortlists_for_stage2(shortlists)
    for s in result:
        assert s.title in top_titles


def test_select_shortlists_varies_across_runs() -> None:
    """Multiple calls with the same large input should produce different orderings/selections."""
    from mixlab.llm import select_shortlists_for_stage2

    shortlists = [MixConcept(title=str(i), mood="m", track_ids=[str(i)] * (10 + i)) for i in range(20)]
    results = [frozenset(s.title for s in select_shortlists_for_stage2(shortlists)) for _ in range(20)]
    # With 20 samples from top-12 choose-6 there should be more than one unique selection.
    assert len(set(results)) > 1


def test_tracks_to_text_emits_seed_annotation() -> None:
    from mixlab.llm import _tracks_to_text

    text = _tracks_to_text(_make_tracks(1), seed_ids=frozenset({"0"}))
    assert "[seed]" in text


def test_tracks_to_text_no_seed_annotation_without_seed_ids() -> None:
    from mixlab.llm import _tracks_to_text

    text = _tracks_to_text(_make_tracks(1))
    assert "[seed]" not in text


def test_select_shortlists_for_playlist_stage2_ranks_by_seed_count() -> None:
    from mixlab.llm import select_shortlists_for_playlist_stage2

    shortlists = [
        MixConcept(title="one", mood="m", track_ids=["1", "9"]),
        MixConcept(title="two", mood="m", track_ids=["1", "2", "9"]),
        MixConcept(title="zero", mood="m", track_ids=["9"]),
    ]
    result = select_shortlists_for_playlist_stage2(shortlists, frozenset({"1", "2"}))
    assert [shortlist.title for shortlist in result] == ["two", "one", "zero"]


def test_select_shortlists_for_playlist_stage2_caps_at_stage2_cap() -> None:
    from mixlab.llm import _STAGE2_CAP, select_shortlists_for_playlist_stage2

    shortlists = [MixConcept(title=str(i), mood="m", track_ids=["1", str(i)]) for i in range(_STAGE2_CAP + 3)]
    result = select_shortlists_for_playlist_stage2(shortlists, frozenset({"1"}))
    assert len(result) == _STAGE2_CAP


# ---------------------------------------------------------------------------
# Stage 2 custom genre cross-genre prompt injection
# ---------------------------------------------------------------------------


@respx.mock
async def test_stage2_prompt_includes_custom_genre_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """When custom_genre_label is set, the user prompt must contain the cross-genre justification instruction."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        custom_genre_label="170",
        custom_genre_sub_genres=["drum_and_bass", "jungle"],
    )

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "170" in user_prompt
    assert "drum_and_bass" in user_prompt
    assert "jungle" in user_prompt
    assert "cross-genre" in user_prompt.lower() or "sub-genre" in user_prompt.lower()


@respx.mock
async def test_stage2_prompt_has_no_custom_genre_section_for_standard_genre(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standard genre runs must not have the custom genre paragraph appended."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "multi-genre custom pool" not in user_prompt


@respx.mock
async def test_stage2_rendering_includes_seed_annotation_for_seed_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, playlist_name="Monday Night", seed_ids=frozenset({"1"}))

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "[seed]" in user_prompt


@respx.mock
async def test_stage2_rendering_no_seed_annotation_without_seed_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "[seed]" not in user_prompt


@respx.mock
async def test_stage2_rendering_unplayed_uses_unplayed_ids_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        "1": Track(
            track_id="1", artist="A1", title="T1", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=1
        ),
        "2": Track(
            track_id="2", artist="A2", title="T2", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=0
        ),
        "3": Track(
            track_id="3", artist="A3", title="T3", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=1
        ),
        "4": Track(
            track_id="4", artist="A4", title="T4", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=1
        ),
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, playlist_name="Monday Night", unplayed_ids={"1"})

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "ID:1" in user_prompt and "unplayed" in user_prompt
    line_for_two = next(line for line in user_prompt.splitlines() if "ID:2" in line)
    assert "unplayed" not in line_for_two


@respx.mock
async def test_stage2_rendering_unplayed_falls_back_to_play_count_when_unplayed_ids_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        "1": Track(
            track_id="1", artist="A1", title="T1", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=0
        ),
        "2": Track(
            track_id="2", artist="A2", title="T2", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=1
        ),
        "3": Track(
            track_id="3", artist="A3", title="T3", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=1
        ),
        "4": Track(
            track_id="4", artist="A4", title="T4", bpm=174.0, camelot_key="8A", genre="Drum & Bass", play_count=1
        ),
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, playlist_name="Monday Night")

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    line_for_one = next(line for line in user_prompt.splitlines() if "ID:1" in line)
    assert "unplayed" in line_for_one


@respx.mock
async def test_stage2_playlist_mode_prompt_contains_three_concept_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, playlist_name="Monday Night")

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "EXACTLY THREE" in user_prompt


@respx.mock
async def test_stage2_playlist_mode_prompt_contains_playlist_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, playlist_name="Monday Night")

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "Monday Night" in user_prompt


@respx.mock
async def test_stage2_playlist_mode_prompt_contains_seed_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, playlist_name="Monday Night")

    import json as _json

    body = _json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "Tracks marked [seed]" in user_prompt


def test_rewrite_playlist_report_injects_summary_at_track_order_marker() -> None:
    """With the new compact header 'Track order:', summary is injected before the track list."""
    from mixlab.llm import _rewrite_playlist_report  # noqa: PLC2701

    concept = MixConcept(title="Set", mood="practical", track_ids=["1", "2", "5", "6"])
    tracks_by_id = {
        str(i): Track(
            track_id=str(i),
            artist=f"Artist {i}",
            title=f"Title {i}",
            bpm=120.0,
            camelot_key="8A",
            genre="House",
        )
        for i in range(1, 7)
    }
    report = (
        "CONCEPT: Set\n\nA driving set.\n\n"
        "Track order:\n"
        "1. Artist 1 — Title 1 [8A · 120.0] | Role: opener | Why: sets tone | Risk: none\n"
        "2. Artist 2 — Title 2 [8A · 120.0] | Role: builder | Why: builds | Risk: none"
    )
    rewritten = _rewrite_playlist_report(report, "Monday", concept, ["1", "2", "3", "4"], tracks_by_id)

    assert "Seed tracks retained: 2" in rewritten
    assert "Seed tracks dropped: 2." in rewritten
    # Summary must appear BEFORE the track list, not appended at end
    summary_pos = rewritten.index("Seed tracks retained")
    track_list_pos = rewritten.index("Track order:")
    assert summary_pos < track_list_pos  # summary is injected before the "Track order:" marker line


def test_rewrite_playlist_report_overwrites_incorrect_counts() -> None:
    from mixlab.llm import _rewrite_playlist_report

    concept = MixConcept(title="Set", mood="m", track_ids=["1", "2", "5", "6"])
    tracks_by_id = {
        str(i): Track(
            track_id=str(i), artist=f"Artist {i}", title=f"Title {i}", bpm=120.0, camelot_key="8A", genre="House"
        )
        for i in range(1, 7)
    }
    report = (
        "CONCEPT: Set\n\nThesis.\n\n"
        "Source playlist: Monday Night\n"
        "Seed tracks retained: 99.\n"
        "Seed tracks dropped: 0.\n"
        "Library tracks added: 0.\n\n"
        "Track order:\nArtist 1 — Title 1 [8A · 120.0]"
    )

    rewritten = _rewrite_playlist_report(report, "Monday Night", concept, ["1", "2", "3", "4"], tracks_by_id)

    assert "Seed tracks retained: 2" in rewritten
    assert "Seed tracks dropped: 2." in rewritten
    assert "Library tracks added: 2." in rewritten
    assert "Seed tracks retained: 99" not in rewritten


@respx.mock
async def test_stage2_playlist_mode_report_rewrites_seed_counts_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "Set",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4", "5", "6", "7", "8"],
                "report": (
                    "CONCEPT: Set\n\nThesis.\n\n"
                    "Source playlist: Monday Night\n"
                    "Seed tracks retained: 1.\n"
                    "Seed tracks dropped: 0.\n"
                    "Library tracks added: 7.\n\n"
                    "Track order (Camelot / BPM):\nArtist 1 — Title 1 [8A · 120.0]"
                ),
            }
        ]
    )
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(payload)))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=[str(i) for i in range(1, 9)])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=120.0 + i, camelot_key="8A", genre="House")
        for i in range(1, 9)
    }

    _concepts, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1", "2", "3", "4", "9", "10"}),
        seed_track_ids=["1", "2", "3", "4", "9", "10"],
    )

    assert "Seed tracks retained: 4" in report
    assert "Seed tracks dropped: 2." in report
    assert "Library tracks added: 4." in report


@respx.mock
async def test_stage2_playlist_mode_raises_when_retention_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "Set",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4", "11", "12", "13", "14", "15", "16", "17", "18"],
                "report": "CONCEPT: Set\n\nThesis.\n\nTrack order (Camelot / BPM):\nArtist 1 — Title 1 [8A · 120.0]",
            }
        ]
    )
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(payload)))

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=[str(i) for i in range(1, 19)])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=120.0 + i, camelot_key="8A", genre="House")
        for i in range(1, 19)
    }

    with pytest.raises(RuntimeError, match="retained too few seed tracks"):
        await stage2_curate_and_report(
            shortlists,
            tracks_by_id,
            playlist_name="Monday Night",
            seed_ids=frozenset({str(i) for i in range(1, 11)}),
            seed_track_ids=[str(i) for i in range(1, 11)],
        )


# ---------------------------------------------------------------------------
# _parse_curated_concepts — transitions parsing
# ---------------------------------------------------------------------------


def test_parse_curated_concepts_parses_transitions() -> None:
    from mixlab.llm import _parse_curated_concepts  # noqa: PLC2701

    raw = json.dumps(
        [
            {
                "title": "T",
                "mood": "practical",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [
                    {"from_id": "1", "to_id": "2", "is_risky": False, "risk_type": ""},
                    {"from_id": "2", "to_id": "3", "is_risky": True, "risk_type": "chapter_pivot"},
                ],
                "report": "x",
            }
        ]
    )
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert len(concepts[0].transitions) == 2
    assert concepts[0].transitions[1].is_risky is True
    assert concepts[0].transitions[1].risk_type == "chapter_pivot"


def test_parse_curated_concepts_missing_transitions_key_yields_empty_list() -> None:
    from mixlab.llm import _parse_curated_concepts  # noqa: PLC2701

    raw = json.dumps([{"title": "T", "mood": "m", "track_ids": ["1", "2", "3", "4"], "report": "x"}])
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert concepts[0].transitions == []


def test_parse_curated_concepts_unmatched_transition_ids_stored_as_is() -> None:
    """Transition IDs not in track_ids are stored without filtering — ignored at scoring time."""
    from mixlab.llm import _parse_curated_concepts  # noqa: PLC2701

    raw = json.dumps(
        [
            {
                "title": "T",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [
                    {"from_id": "99", "to_id": "100", "is_risky": True, "risk_type": "cut_only"},
                ],
                "report": "x",
            }
        ]
    )
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    # stored verbatim — scorer ignores them when looking up consecutive pairs
    assert len(concepts[0].transitions) == 1
    assert concepts[0].transitions[0].from_id == "99"
