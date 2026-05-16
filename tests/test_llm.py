from __future__ import annotations

import json
from typing import get_args

import pytest
import respx
from httpx import Response

from mixlab.models import (
    CanvasRoleCandidates,
    CanvasScore,
    CompletionVariant,
    ContrastAssets,
    DJPracticalityScore,
    MixCanvas,
    MixConcept,
    Track,
)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _shortlist_payload() -> str:
    """Stage 1 response — candidate pools using T-prefixed aliases (T001, T002, …)."""
    return json.dumps(
        [
            {
                "title": "Deep 122 BPM / 4A–7A Pool",
                "mood": "heavy and relentless",
                "track_ids": [f"T{i:03d}" for i in range(1, 10)],
            },
            {
                "title": "Liquid 124 BPM / 8A–11A Pool",
                "mood": "smooth and atmospheric",
                "track_ids": [f"T{i:03d}" for i in range(10, 18)],
            },
        ]
    )


def _curated_payload() -> str:
    """Stage 2 selection-only response — no report field."""
    return json.dumps(
        [
            {
                "title": "Dark Rollers",
                "name_reason": "Relentless drive from open to close.",
                "mood": "heavy and relentless",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [
                    {"from_id": "1", "to_id": "2", "is_risky": False, "risk_type": ""},
                    {"from_id": "2", "to_id": "3", "is_risky": False, "risk_type": ""},
                    {"from_id": "3", "to_id": "4", "is_risky": False, "risk_type": ""},
                ],
            }
        ]
    )


def _chat_response() -> dict[str, object]:
    return {"choices": [{"message": {"content": _shortlist_payload()}}]}


_REPORT_TEXT = "CONCEPT: Dark Rollers\n\nA relentless journey.\n\nTrack order:\n1. Artist 1 — Title 1 [8A · 174.0] | Role: opener | Why: sets dark tone | Risk: none"


def _anthropic_response(content: str) -> dict[str, object]:
    return {"content": [{"text": content}], "stop_reason": "end_turn"}


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

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    respx.post(_GROQ_URL).mock(return_value=Response(200, json=_chat_response()))

    shortlists = await stage1_concepts(_make_tracks(20), "Drum & Bass", make_cascade_state())
    assert len(shortlists) == 2
    assert shortlists[0].title == "Deep 122 BPM / 4A–7A Pool"


@respx.mock
async def test_stage1_logs_provider_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    respx.post(_GROQ_URL).mock(return_value=Response(200, json=_chat_response()))

    await stage1_concepts(_make_tracks(20), "Drum & Bass", make_cascade_state())

    captured = capsys.readouterr()
    assert "Stage 1 trying provider: _try_groq | genre=Drum & Bass | input=20 tracks" in captured.out
    assert (
        "Stage 1 provider: _try_groq | genre=Drum & Bass | input=20 tracks | parsed=2 | cleaned=2 | kept=2"
        in captured.out
    )
    assert "Deep 122 BPM / 4A–7A Pool | raw=9 | kept=9 | kept" in captured.out
    assert "Liquid 124 BPM / 8A–11A Pool | raw=8 | kept=8 | kept" in captured.out


@respx.mock
async def test_stage1_falls_through_cascade_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "bad-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    respx.post(_GROQ_URL).mock(return_value=Response(500, text="error"))
    respx.post(_GEMINI_URL).mock(return_value=Response(200, json=_chat_response()))

    shortlists = await stage1_concepts(_make_tracks(20), "Drum & Bass", make_cascade_state())
    assert shortlists[0].title == "Deep 122 BPM / 4A–7A Pool"


@respx.mock
async def test_stage1_chunks_large_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 50 tracks → two chunks (40 + 10); each call returns aliases valid for its own chunk.
    respx.post(_GROQ_URL).mock(
        side_effect=[
            Response(200, json=_chat_response()),
            Response(200, json=_chat_response()),
        ]
    )

    shortlists = await stage1_concepts(_make_tracks(50), "Drum & Bass", make_cascade_state())
    # Chunk 1 (tracks 0–39, aliases T001–T040): both pools (T001–T009, T010–T017) valid → 2 shortlists.
    # Chunk 2 (tracks 40–49, aliases T001–T010): first pool (T001–T009) valid → 1 shortlist;
    # second pool requests T010–T017 but only T010 exists in the 10-track chunk → dropped.
    assert len(shortlists) == 3


@respx.mock
async def test_stage1_filters_pools_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Pool with only 3 valid aliases (below _MIN_SHORTLIST_TRACKS=8) should be dropped.
    tiny_pool = json.dumps([{"title": "Tiny", "mood": "x", "track_ids": ["T001", "T002", "T003"]}])
    respx.post(_GROQ_URL).mock(return_value=Response(200, json={"choices": [{"message": {"content": tiny_pool}}]}))

    shortlists = await stage1_concepts(_make_tracks(10), "Drum & Bass", make_cascade_state())
    assert shortlists == []


@respx.mock
async def test_stage1_logs_when_all_shortlists_are_dropped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    tiny_pool = json.dumps([{"title": "Tiny", "mood": "x", "track_ids": ["T001", "T002", "T003"]}])
    respx.post(_GROQ_URL).mock(return_value=Response(200, json={"choices": [{"message": {"content": tiny_pool}}]}))

    shortlists = await stage1_concepts(_make_tracks(10), "Drum & Bass", make_cascade_state())

    assert shortlists == []
    captured = capsys.readouterr()
    assert (
        "Stage 1 provider: _try_groq | genre=Drum & Bass | input=10 tracks | parsed=1 | cleaned=1 | kept=0"
        in captured.out
    )
    assert "Tiny | raw=3 | kept=3 | dropped (<8)" in captured.out


@respx.mock
async def test_stage1_parses_response_with_trailing_prose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models that append explanatory text after the JSON array should still parse correctly."""
    from mixlab.llm import make_cascade_state, stage1_concepts

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
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
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
                "name_reason": "n/a",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4", "999"],
                "transitions": [],
            }
        ]
    )
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(payload)),
            Response(200, json=_anthropic_response("CONCEPT: T\n\nBrief.")),
        ]
    )

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


def test_parse_curated_concepts_repairs_literal_newlines_in_strings() -> None:
    from mixlab.llm import _parse_curated_concepts

    # Simulate model outputting literal newlines inside a JSON string value.
    broken = '[{"title":"T","mood":"m","track_ids":["1","2","3","4"],"report":"line one\nline two\nline three"}]'
    concepts, report = _parse_curated_concepts(broken, {"1", "2", "3", "4"})
    assert len(concepts) == 1
    assert "line one" in report
    assert "line two" in report


def test_parse_curated_concepts_extracts_arc_type() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps(
        [{"title": "T", "mood": "m", "track_ids": ["1", "2", "3", "4"], "arc_type": "wave", "report": "x"}]
    )
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert concepts[0].arc_type == "wave"


def test_parse_curated_concepts_missing_arc_type_is_none() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps([{"title": "T", "mood": "m", "track_ids": ["1", "2", "3", "4"], "report": "x"}])
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert concepts[0].arc_type is None


def test_parse_curated_concepts_invalid_arc_type_is_none() -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps(
        [
            {
                "title": "T",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4"],
                "arc_type": "not-a-real-arc",
                "report": "x",
            }
        ]
    )
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert concepts[0].arc_type is None


def test_parse_curated_concepts_normalises_arc_type_underscores_and_case() -> None:
    from mixlab.llm import _parse_curated_concepts

    # Model may emit "build_and_drop" or "Build-And-Drop" — parser should normalise to canonical form.
    raw = json.dumps(
        [
            {
                "title": "T",
                "mood": "m",
                "track_ids": ["1", "2", "3", "4"],
                "arc_type": "Build_And_Drop",
                "report": "x",
            }
        ]
    )
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})
    assert concepts[0].arc_type == "build-and-drop"


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
    from mixlab.config import shortfall_warning

    concept = MixConcept(title="Too Small", mood="dark", track_ids=["1", "2"])
    warning = shortfall_warning(concept, "Drum & Bass")
    assert warning is not None
    assert "2 tracks found" in warning
    assert "needs" in warning


def test_shortfall_warning_not_triggered_near_minimum() -> None:
    from mixlab.config import shortfall_warning

    concept = MixConcept(title="Nearly There", mood="dark", track_ids=[str(i) for i in range(8)])
    warning = shortfall_warning(concept, "Drum & Bass")
    assert warning is None


def test_shortfall_warning_not_triggered_at_minimum() -> None:
    from mixlab.config import shortfall_warning

    concept = MixConcept(title="Full Set", mood="dark", track_ids=[str(i) for i in range(10)])
    warning = shortfall_warning(concept, "Drum & Bass")
    assert warning is None


# ---------------------------------------------------------------------------
# Edge: all providers missing
# ---------------------------------------------------------------------------


async def test_stage1_raises_if_all_providers_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import make_cascade_state, stage1_concepts

    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY"):
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
            Response(200, json=_chat_response()),
            Response(200, json=_chat_response()),
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
            Response(200, json=_chat_response()),
            Response(200, json=_chat_response()),
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

    respx.post(_GEMINI_URL).mock(return_value=Response(200, json=_chat_response()))

    state = make_cascade_state()
    await stage1_concepts(_make_tracks(20), "Drum & Bass", state)

    assert state.consecutive_failures == 0
    assert state.index == 1  # advanced past unconfigured Groq to Gemini (index 1)


@respx.mock
async def test_stage1_raises_after_n_consecutive_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """All five providers fail → RuntimeError after len(_CASCADE) consecutive failures."""
    from mixlab.llm import make_cascade_state, stage1_concepts

    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("MISTRAL_API_KEY", "key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")

    respx.post(_GROQ_URL).mock(return_value=Response(500, text="error"))
    respx.post(_GEMINI_URL).mock(return_value=Response(500, text="error"))
    respx.post(_MISTRAL_URL).mock(return_value=Response(500, text="error"))
    respx.post(_OPENROUTER_URL).mock(return_value=Response(500, text="error"))

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
            Response(200, json=_chat_response()),
            Response(200, json=_chat_response()),
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

    body = json.loads(route.calls[0].request.content.decode())
    user_prompt: str = body["messages"][0]["content"]
    assert "[unverified]" not in user_prompt


@respx.mock
async def test_stage2_marks_only_unplayed_tracks_when_unplayed_ids_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    """When unplayed_ids is provided, only those tracks get the 'unplayed' marker — not all tracks."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    # All tracks have play_count > 0 — the play_count fallback would mark NONE of them. So any
    # "unplayed" markers we see in the prompt must come from the explicit unplayed_ids set.
    tracks_by_id = {
        str(i): Track(
            track_id=str(i),
            artist=f"Artist{i}",
            title=f"Title{i}",
            bpm=174.0,
            camelot_key="8A",
            genre="Drum & Bass",
            play_count=5,
        )
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, unplayed_ids={"1", "3"})

    body = json.loads(route.calls[0].request.content)
    prompt_text = next(m["content"] for m in body["messages"] if m["role"] == "user")
    lines_by_tid: dict[str, str] = {}
    for line in prompt_text.split("\n"):
        for tid in ("1", "2", "3", "4"):
            if f"ID:{tid} " in line or f"ID:{tid}|" in line:
                lines_by_tid[tid] = line
    assert "unplayed" in lines_by_tid["1"]
    assert "unplayed" in lines_by_tid["3"]
    assert "unplayed" not in lines_by_tid["2"]
    assert "unplayed" not in lines_by_tid["4"]

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
                "transitions": [],
            }
        ]
    )
    prose_report = (
        "CONCEPT: Set\n\nThesis.\n\n"
        "Source playlist: Monday Night\n"
        "Seed tracks retained: 1.\n"
        "Seed tracks dropped: 0.\n"
        "Library tracks added: 7.\n\n"
        "Track order:\nArtist 1 — Title 1 [8A · 120.0] | Role: opener | Why: opens | Risk: none"
    )
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(payload)),
            Response(200, json=_anthropic_response(prose_report)),
        ]
    )

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
                "transitions": [],
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


@respx.mock
async def test_stage2_playlist_mode_returns_winner_first_with_labeled_titles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "Balanced Set",
                "mood": "balanced",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [],
            },
            {
                "title": "Practical Set",
                "mood": "practical",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [],
            },
            {
                "title": "Adventurous Set",
                "mood": "adventurous",
                "track_ids": ["1", "2", "3", "4"],
                "transitions": [],
            },
        ]
    )

    def make_report(title: str) -> str:
        return f"CONCEPT: {title}\n\nThesis.\n\nTrack order:\n1. A1 — T1 [8A · 121.0] | Role: opener | Why: opens | Risk: none"

    # ordered_variants = winner (Practical) + sorted remainder → Practical, Balanced, Adventurous
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(payload)),
            Response(200, json=_anthropic_response(make_report("Practical Set"))),
            Response(200, json=_anthropic_response(make_report("Balanced Set"))),
            Response(200, json=_anthropic_response(make_report("Adventurous Set"))),
        ]
    )

    def fake_score_variant(
        concept: MixConcept,
        seed_track_ids: list[str],
        intent_brief: object,
        tracks_by_id: dict[str, Track],
    ) -> CompletionVariant:
        del seed_track_ids, intent_brief, tracks_by_id
        score_by_strategy = {
            "practical": 0.9,
            "balanced": 0.7,
            "adventurous": 0.6,
        }
        score = score_by_strategy[concept.mood]
        return CompletionVariant(
            strategy=concept.mood,  # type: ignore[arg-type]
            concept=concept,
            anchor_retention_rate=1.0,
            practicality_score=DJPracticalityScore(score, score, score, score),
        )

    monkeypatch.setattr("mixlab.llm._score_variant", fake_score_variant)

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=120.0 + i, camelot_key="8A", genre="House")
        for i in range(1, 5)
    }

    concepts, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1", "2", "3", "4"}),
        seed_track_ids=["1", "2", "3", "4"],
    )

    assert [concept.title for concept in concepts] == [
        "WINNER - PRACTICAL - Practical Set",
        "BALANCED - Balanced Set",
        "ADVENTUROUS - Adventurous Set",
    ]
    assert report.startswith("VARIANT: WINNER - PRACTICAL")
    assert "VARIANT: BALANCED" in report
    assert "VARIANT: ADVENTUROUS" in report
    assert report.index("VARIANT: WINNER - PRACTICAL") < report.index("VARIANT: BALANCED")
    assert report.index("VARIANT: BALANCED") < report.index("VARIANT: ADVENTUROUS")
    assert (
        "Alternative strategies considered: balanced (practicality: 0.70, anchor retention: 100%) — not selected; "
        "adventurous (practicality: 0.60, anchor retention: 100%) — not selected."
    ) in report
    assert "Alternative strategies considered: practical" not in report


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


def test_stage2_playlist_system_caps_tracks_at_fourteen() -> None:
    from mixlab.llm import _STAGE2_SYSTEM_PLAYLIST

    assert "10–12 tracks" not in _STAGE2_SYSTEM_PLAYLIST
    assert "10–14 tracks" in _STAGE2_SYSTEM_PLAYLIST
    assert "Do not exceed 14" in _STAGE2_SYSTEM_PLAYLIST


# ---------------------------------------------------------------------------
# _make_selection_system and derived constants
# ---------------------------------------------------------------------------


def test_make_selection_system_removes_report_schema_field() -> None:
    from mixlab.llm import _STAGE2_SYSTEM, _make_selection_system

    result = _make_selection_system(_STAGE2_SYSTEM)
    assert '"report":' not in result
    assert '"track_ids":' in result
    assert '"transitions":' in result
    assert "Respond ONLY with the JSON array." in result


def test_make_selection_system_removes_report_format_instructions() -> None:
    from mixlab.llm import _STAGE2_SYSTEM, _make_selection_system

    result = _make_selection_system(_STAGE2_SYSTEM)
    assert 'The "report" value must be a single string' not in result
    assert "Role options: opener" not in result  # part of the report-format section


def test_make_selection_system_preserves_curation_instructions() -> None:
    from mixlab.llm import _STAGE2_SYSTEM, _make_selection_system

    result = _make_selection_system(_STAGE2_SYSTEM)
    # Key curation guidance must survive
    assert "peak weapons" in result
    assert "name_reason" in result
    assert "chapter_pivot" in result


def test_selection_system_playlist_variant_has_practical_balanced_adventurous() -> None:
    from mixlab.llm import _STAGE2_SYSTEM_PLAYLIST_SELECTION

    assert '"practical"' in _STAGE2_SYSTEM_PLAYLIST_SELECTION
    assert '"balanced"' in _STAGE2_SYSTEM_PLAYLIST_SELECTION
    assert '"adventurous"' in _STAGE2_SYSTEM_PLAYLIST_SELECTION
    assert '"report":' not in _STAGE2_SYSTEM_PLAYLIST_SELECTION


def test_used_mix_names_are_injected_into_standard_selection_system() -> None:
    """Regression: the replace() search string previously had a spurious backslash that never matched."""
    from mixlab.llm import _STAGE2_SYSTEM_SELECTION

    names_str = "Slow Burn, Night Drive"
    first_name = "Slow Burn"
    system = _STAGE2_SYSTEM_SELECTION.replace(
        'The name should make someone curious, not nod in recognition. Add a "name_reason" field',
        "The name should make someone curious, not nod in recognition. "
        f"Do not reuse or closely echo any of these existing mix names from the DJ's catalogue: {names_str}. "
        "Avoid borrowing any word, phrase, or trope from those names — even as a prefix, suffix, or modifier "
        f"(e.g. if '{first_name}' is in the list, '{first_name} Vol. 2' and any variation is forbidden). "
        'Add a "name_reason" field',
    )
    assert "Slow Burn" in system
    assert "Night Drive" in system
    assert "Slow Burn Vol. 2" in system  # the example in the injected warning


def test_used_mix_names_are_injected_into_playlist_selection_system() -> None:
    from mixlab.llm import _STAGE2_SYSTEM_PLAYLIST_SELECTION

    names_str = "Slow Burn"
    first_name = "Slow Burn"
    system = _STAGE2_SYSTEM_PLAYLIST_SELECTION.replace(
        'The name should make someone curious, not nod in recognition. Add a "name_reason" field',
        "The name should make someone curious, not nod in recognition. "
        f"Do not reuse or closely echo any of these existing mix names from the DJ's catalogue: {names_str}. "
        "Avoid borrowing any word, phrase, or trope from those names — even as a prefix, suffix, or modifier "
        f"(e.g. if '{first_name}' is in the list, '{first_name} Vol. 2' and any variation is forbidden). "
        'Add a "name_reason" field',
    )
    assert "Slow Burn" in system


# ---------------------------------------------------------------------------
# Stage 2 canvas rules language (role hints, bridge form, canvas skip)
# ---------------------------------------------------------------------------


def test_stage2_canvas_rules_state_role_hints_can_be_overridden() -> None:
    from mixlab.llm import _STAGE2_CANVAS_RULES

    assert "hints, not assignments" in _STAGE2_CANVAS_RULES
    assert "Override them freely" in _STAGE2_CANVAS_RULES


def test_stage2_canvas_rules_require_specific_structural_reason_for_bridge_wildcard() -> None:
    from mixlab.llm import _STAGE2_CANVAS_RULES

    assert "specific structural role" in _STAGE2_CANVAS_RULES
    assert '"interesting track"' in _STAGE2_CANVAS_RULES


def test_stage2_canvas_rules_allow_canvas_skip_on_unfixable_risks() -> None:
    from mixlab.llm import _STAGE2_CANVAS_RULES

    assert "may skip that canvas" in _STAGE2_CANVAS_RULES
    assert "not obligated to produce a concept from every canvas" in _STAGE2_CANVAS_RULES


# ---------------------------------------------------------------------------
# Stage 2 schema — arc_type enum
# ---------------------------------------------------------------------------


def test_stage2_system_documents_arc_type_field_in_schema() -> None:
    from mixlab.llm import _STAGE2_SYSTEM

    assert '"arc_type":' in _STAGE2_SYSTEM


def test_stage2_system_documents_all_arc_type_enum_values() -> None:
    from mixlab.llm import _STAGE2_SYSTEM
    from mixlab.models import ArcType

    for value in get_args(ArcType):
        assert f'"{value}"' in _STAGE2_SYSTEM, f"arc_type enum value {value!r} missing from _STAGE2_SYSTEM"


def test_stage2_selection_system_retains_arc_type_field() -> None:
    from mixlab.llm import _STAGE2_SYSTEM_SELECTION

    # Selection-pass system prompt strips the report field but must keep arc_type for the selection JSON output.
    assert '"arc_type":' in _STAGE2_SYSTEM_SELECTION
    assert '"report":' not in _STAGE2_SYSTEM_SELECTION


def test_stage2_playlist_selection_system_retains_arc_type_field() -> None:
    from mixlab.llm import _STAGE2_SYSTEM_PLAYLIST_SELECTION

    assert '"arc_type":' in _STAGE2_SYSTEM_PLAYLIST_SELECTION


# ---------------------------------------------------------------------------
# _call_stage2_reports — parallel report generation
# ---------------------------------------------------------------------------


@respx.mock
async def test_call_stage2_reports_returns_one_report_per_concept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import _call_stage2_reports

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    concepts = [
        MixConcept(title="Set A", mood="dark", track_ids=["1", "2", "3", "4"]),
        MixConcept(title="Set B", mood="light", track_ids=["1", "2", "3", "4"]),
    ]
    tracks_by_id = {
        str(i): Track(
            track_id=str(i), artist=f"Artist {i}", title=f"Title {i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass"
        )
        for i in range(1, 5)
    }

    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json={"content": [{"text": "CONCEPT: Set A\n\nReport A."}], "stop_reason": "end_turn"}),
            Response(200, json={"content": [{"text": "CONCEPT: Set B\n\nReport B."}], "stop_reason": "end_turn"}),
        ]
    )

    reports = await _call_stage2_reports(concepts, tracks_by_id, None, None, "test-key")

    assert len(reports) == 2
    assert "Report A" in reports[0]
    assert "Report B" in reports[1]


@respx.mock
async def test_call_stage2_reports_fires_parallel_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import _call_stage2_reports

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    call_count = 0

    def count_calls(request: object) -> Response:
        nonlocal call_count
        call_count += 1
        return Response(200, json={"content": [{"text": f"Report {call_count}"}], "stop_reason": "end_turn"})

    concepts = [MixConcept(title=f"Set {i}", mood="dark", track_ids=["1", "2", "3", "4"]) for i in range(3)]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }

    respx.post(_ANTHROPIC_URL).mock(side_effect=count_calls)

    reports = await _call_stage2_reports(concepts, tracks_by_id, None, None, "test-key")

    assert len(reports) == 3
    assert call_count == 3


# ---------------------------------------------------------------------------
# Bold moves annotation — bridge/wildcard usage surfaced in concept report (#15)
# ---------------------------------------------------------------------------


def _make_canvas_full(
    core_ids: list[str],
    bridge_ids: list[str] | None = None,
    wildcard_ids: list[str] | None = None,
) -> MixCanvas:
    """Canvas factory that accepts wildcard tracks (the default helper only supports bridge)."""
    all_ids = core_ids + (bridge_ids or []) + (wildcard_ids or [])
    concept = MixConcept(title="T", mood="dark", track_ids=all_ids)
    return MixCanvas(
        canvas_id="test_canvas",
        genre="Drum & Bass",
        bpm_range=(160.0, 180.0),
        dominant_bpm=172.0,
        dominant_camelot="4A",
        core_track_ids=core_ids,
        bridge_track_ids=bridge_ids or [],
        wildcard_track_ids=wildcard_ids or [],
        roles=CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[]),
        contrast=ContrastAssets(
            vocal_moments=[],
            texture_changes=[],
            darker_turns=[],
            brighter_lifts=[],
            lower_pressure_resets=[],
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=concept,
    )


def test_format_bold_moves_all_core_returns_none() -> None:
    from mixlab.llm import _format_bold_moves

    ids = [str(i) for i in range(1, 9)]
    canvas = _make_canvas_full(core_ids=ids)
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    result = _format_bold_moves(concept, canvas, _lib(ids))
    assert result == "Bold moves: none"


def test_format_bold_moves_counts_bridge_and_wildcard() -> None:
    from mixlab.llm import _format_bold_moves
    from mixlab.models import Transition

    core_ids = ["1", "2", "3", "4", "5", "6"]
    bridge_ids = ["7"]
    wildcard_ids = ["8"]
    canvas = _make_canvas_full(core_ids=core_ids, bridge_ids=bridge_ids, wildcard_ids=wildcard_ids)
    concept = MixConcept(
        title="T",
        mood="dark",
        track_ids=core_ids + bridge_ids + wildcard_ids,
        transitions=[
            Transition(from_id="6", to_id="7", is_risky=True, risk_type="chapter_pivot"),
            Transition(from_id="7", to_id="8", is_risky=True, risk_type="peak_impact"),
        ],
    )
    result = _format_bold_moves(concept, canvas, _lib(core_ids + bridge_ids + wildcard_ids))
    first_line = result.split("\n")[0]
    assert first_line == "Bold moves: 1 bridge, 1 wildcard"
    assert "(bridge): chapter pivot" in result
    assert "(wildcard): peak impact" in result


def test_format_bold_moves_skips_bullet_when_no_risk_type() -> None:
    """Bridge track without a named mechanism gets the count but no bullet line."""
    from mixlab.llm import _format_bold_moves
    from mixlab.models import Transition

    core_ids = ["1", "2", "3", "4", "5", "6", "7"]
    bridge_ids = ["8"]
    canvas = _make_canvas_full(core_ids=core_ids, bridge_ids=bridge_ids)
    concept = MixConcept(
        title="T",
        mood="dark",
        track_ids=core_ids + bridge_ids,
        # Incoming transition has empty risk_type — should not produce a bullet.
        transitions=[Transition(from_id="7", to_id="8", is_risky=False, risk_type="")],
    )
    result = _format_bold_moves(concept, canvas, _lib(core_ids + bridge_ids))
    assert result == "Bold moves: 1 bridge"


def test_match_canvas_for_concept_picks_canvas_with_most_overlap() -> None:
    from mixlab.llm import _match_canvas_for_concept

    canvas_a = _make_canvas_full(core_ids=["1", "2", "3", "4"])  # 4-track pool
    canvas_b = _make_canvas_full(core_ids=["10", "11", "12", "13"])  # disjoint pool
    concept = MixConcept(title="T", mood="dark", track_ids=["1", "2", "3"])
    match = _match_canvas_for_concept(concept, [canvas_a, canvas_b])
    assert match is canvas_a


def test_match_canvas_for_concept_returns_none_when_no_overlap() -> None:
    from mixlab.llm import _match_canvas_for_concept

    canvas = _make_canvas_full(core_ids=["100", "101", "102"])
    concept = MixConcept(title="T", mood="dark", track_ids=["1", "2", "3"])
    assert _match_canvas_for_concept(concept, [canvas]) is None


def test_append_bold_moves_to_report_no_canvases_returns_report_unchanged() -> None:
    from mixlab.llm import _append_bold_moves_to_report

    concept = MixConcept(title="T", mood="dark", track_ids=["1", "2", "3", "4"])
    original = "CONCEPT: T\n\nSome report body."
    result = _append_bold_moves_to_report(original, concept, None, _lib(["1", "2", "3", "4"]))
    assert result == original


def test_append_bold_moves_to_report_appends_annotation_block() -> None:
    from mixlab.llm import _append_bold_moves_to_report

    ids = [str(i) for i in range(1, 9)]
    canvas = _make_canvas_full(core_ids=ids[:7], bridge_ids=[ids[7]])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    original = "CONCEPT: T\n\nSome report body."
    result = _append_bold_moves_to_report(original, concept, [canvas], _lib(ids))
    assert result.startswith(original)
    assert "Bold moves: 1 bridge" in result


# ---------------------------------------------------------------------------
# validate_stage2_output
# ---------------------------------------------------------------------------


def _make_canvas(core_ids: list[str], bridge_ids: list[str] | None = None) -> MixCanvas:
    concept = MixConcept(title="T", mood="dark", track_ids=core_ids + (bridge_ids or []))
    return MixCanvas(
        canvas_id="test_canvas",
        genre="Drum & Bass",
        bpm_range=(168.0, 178.0),
        dominant_bpm=172.0,
        dominant_camelot="4A",
        core_track_ids=core_ids,
        bridge_track_ids=bridge_ids or [],
        wildcard_track_ids=[],
        roles=CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[]),
        contrast=ContrastAssets(
            vocal_moments=[],
            texture_changes=[],
            darker_turns=[],
            brighter_lifts=[],
            lower_pressure_resets=[],
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=concept,
    )


def _lib(ids: list[str], bpm: float = 174.0, key: str = "8A") -> dict[str, Track]:
    return {
        i: Track(track_id=i, artist=f"Artist_{i}", title=f"Title_{i}", bpm=bpm, camelot_key=key, genre="Drum & Bass")
        for i in ids
    }


def test_validate_stage2_output_passes_clean_concept() -> None:
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]  # 8 tracks — minimum
    concept = MixConcept(title="Clean", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
    assert warnings == []


def test_validate_stage2_output_missing_track_id() -> None:
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids + ["MISSING"])
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
    assert any("MISSING" in w for w in warnings)


def test_validate_stage2_output_denylist_track() -> None:
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), denylist_ids={"1"})
    assert any("denylisted" in w for w in warnings)


def test_validate_stage2_output_played_track_flagged() -> None:
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), played_ids={"3"}, denylist_ids=set())
    assert any("played" in w for w in warnings)


def test_validate_stage2_output_played_track_allowed_when_flag_set() -> None:
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output(
        [concept], [canvas], _lib(ids), played_ids={"3"}, denylist_ids=set(), allow_played=True
    )
    assert not any("played" in w for w in warnings)


def test_validate_stage2_output_bpm_jump_warning() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=200.0, camelot_key="8A", genre="DnB"),  # jump=26
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("BPM jump" in w for w in warnings)


def test_validate_stage2_output_camelot_jump_warning() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="1A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=174.0, camelot_key="8B", genre="DnB"),  # dist=6
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=174.0, camelot_key="1A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("Camelot jump" in w for w in warnings)


def test_validate_stage2_output_artist_repeat_warning() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        str(i): Track(track_id=str(i), artist="Same Artist", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 9)
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("Same Artist" in w for w in warnings)


@respx.mock
async def test_stage2_prompt_includes_canvas_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canvas metadata block appears in the prompt sent to Anthropic."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    ids = ["1", "2", "3", "4"]
    concept = MixConcept(title="Pool", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    canvas.canvas_id = "dnb_174.0_8A"
    canvas.score.novelty = 0.75
    tracks_by_id = _lib(ids)

    await stage2_curate_and_report(shortlists=[concept], tracks_by_id=tracks_by_id, canvases=[canvas])

    body = route.calls[0].request.content.decode()
    assert "[Canvas dnb_174.0_8A" in body
    assert "novelty:" in body
    assert "Core:" in body
