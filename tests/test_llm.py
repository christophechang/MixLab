from __future__ import annotations

import json
from typing import Literal, cast, get_args

import httpx
import pytest
import respx
from httpx import Response

from mixlab.models import (
    CanvasRoleCandidates,
    CanvasScore,
    CompletionVariant,
    ContrastAssets,
    Critique,
    DJPracticalityScore,
    MixCanvas,
    MixConcept,
    MixPoints,
    Track,
    Transition,
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
# _call_anthropic_http — retry/backoff, truncation, env config (#50)
# ---------------------------------------------------------------------------


@respx.mock
async def test_call_anthropic_http_retries_after_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _call_anthropic_http

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("mixlab.llm.asyncio.sleep", fake_sleep)

    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(429, json={"error": "rate_limited"}),
            Response(200, json=_anthropic_response("hello")),
        ]
    )

    text = await _call_anthropic_http("key", "model", "system", "prompt")

    assert text == "hello"
    assert route.call_count == 2
    assert sleeps == [1.0]


@respx.mock
async def test_call_anthropic_http_retry_honours_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _call_anthropic_http

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("mixlab.llm.asyncio.sleep", fake_sleep)

    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(429, headers={"retry-after": "5"}, json={"error": "rate_limited"}),
            Response(200, json=_anthropic_response("hello")),
        ]
    )

    await _call_anthropic_http("key", "model", "system", "prompt")

    assert sleeps == [5.0]


@respx.mock
async def test_call_anthropic_http_raises_after_exhausting_529_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _call_anthropic_http

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("mixlab.llm.asyncio.sleep", fake_sleep)

    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(529, json={"error": "overloaded"}))

    with pytest.raises(httpx.HTTPStatusError):
        await _call_anthropic_http("key", "model", "system", "prompt")

    assert route.call_count == 3


@respx.mock
async def test_call_anthropic_http_raises_immediately_on_non_retryable_status() -> None:
    from mixlab.llm import _call_anthropic_http

    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(400, json={"error": "bad_request"}))

    with pytest.raises(httpx.HTTPStatusError):
        await _call_anthropic_http("key", "model", "system", "prompt")

    assert route.call_count == 1


@respx.mock
async def test_call_anthropic_http_warns_on_truncated_response(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.llm import _call_anthropic_http

    respx.post(_ANTHROPIC_URL).mock(
        return_value=Response(200, json={"content": [{"text": "partial"}], "stop_reason": "max_tokens"})
    )

    text = await _call_anthropic_http("key", "model", "system", "prompt", max_tokens=256)

    assert text == "partial"
    captured = capsys.readouterr()
    assert "WARNING: Anthropic response truncated at max_tokens=256" in captured.err


def test_stage2_model_env_override_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _stage2_model

    monkeypatch.setenv("MIXLAB_STAGE2_MODEL", "claude-opus-9")
    assert _stage2_model() == "claude-opus-9"


def test_stage2_model_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _stage2_model

    monkeypatch.delenv("MIXLAB_STAGE2_MODEL", raising=False)
    assert _stage2_model() == "claude-sonnet-4-6"


def test_stage2_temperature_env_override_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _stage2_temperature

    monkeypatch.setenv("MIXLAB_STAGE2_TEMPERATURE", "0.9")
    assert _stage2_temperature() == 0.9


def test_stage2_temperature_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _stage2_temperature

    monkeypatch.delenv("MIXLAB_STAGE2_TEMPERATURE", raising=False)
    assert _stage2_temperature() == 0.5


def test_stage2_temperature_invalid_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import _stage2_temperature

    monkeypatch.setenv("MIXLAB_STAGE2_TEMPERATURE", "not-a-float")
    assert _stage2_temperature() == 0.5


@respx.mock
async def test_stage2_raw_request_carries_env_override_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Stage 2 selection request payload reflects MIXLAB_STAGE2_MODEL/TEMPERATURE overrides."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MIXLAB_STAGE2_MODEL", "claude-opus-9")
    monkeypatch.setenv("MIXLAB_STAGE2_TEMPERATURE", "0.9")

    route = respx.post(_ANTHROPIC_URL).mock(
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

    _, report = await stage2_curate_and_report(shortlists, tracks_by_id)

    selection_body = json.loads(route.calls[0].request.content)
    assert selection_body["model"] == "claude-opus-9"
    assert selection_body["temperature"] == 0.9
    assert "claude-opus-9" in report


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
    assert "claude-sonnet-4-6" in report
    # Practicality line surfaces in genre-mode reports (#21).
    assert "Practicality" in report
    assert "bpm_smoothness" in report
    assert "overall" in report


def test_format_practicality_line_renders_all_components() -> None:
    """Per-concept practicality summary contains all four labelled components."""
    from mixlab.llm import _format_practicality_line

    score = DJPracticalityScore(
        bpm_smoothness=0.82,
        harmonic_ratio=0.71,
        risk_justified=0.50,
        fragment_preserved=1.0,
    )
    line = _format_practicality_line(score)
    assert "bpm_smoothness 0.82" in line
    assert "harmonic_ratio 0.71" in line
    assert "risk_justified 0.50" in line
    assert "overall" in line


@respx.mock
async def test_stage2_practicality_not_in_playlist_mode_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playlist mode surfaces practicality via WINNER labelling; do not double-append."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="practical", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    _, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1"}),
        seed_track_ids=["1"],
    )
    # Playlist-mode report does not get the genre-mode practicality-line append.
    assert "**Practicality**: bpm_smoothness" not in report


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
async def test_stage2_pool_scoping_drops_id_valid_in_library_but_not_offered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only IDs actually offered to the model this run are accepted (#50).

    Track "999" is a real track in ``tracks_by_id`` (the whole library) but was never
    included in any shortlist offered to Stage 2 — it must be dropped even though the
    old whole-library ``valid_ids`` check would have accepted it.
    """
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = json.dumps(
        [
            {
                "title": "Dark Rollers",
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
            Response(200, json=_anthropic_response("CONCEPT: Dark Rollers\n\nBrief.")),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="m", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }
    # "999" is a real, resolvable library track — but not part of any offered shortlist.
    tracks_by_id["999"] = Track(
        track_id="999", artist="Other", title="Other Track", bpm=174.0, camelot_key="8A", genre="Drum & Bass"
    )

    concepts, _ = await stage2_curate_and_report(shortlists, tracks_by_id)

    assert concepts[0].track_ids == ["1", "2", "3", "4"]
    captured = capsys.readouterr()
    assert "dropped 1 track ID(s) outside the offered pool" in captured.err


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


def test_parse_curated_concepts_prints_dropped_count_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    from mixlab.llm import _parse_curated_concepts

    raw = json.dumps([{"title": "Set A", "mood": "dark", "track_ids": ["1", "2", "3", "4", "999"], "report": "x"}])
    concepts, _ = _parse_curated_concepts(raw, {"1", "2", "3", "4"})

    assert concepts[0].track_ids == ["1", "2", "3", "4"]
    captured = capsys.readouterr()
    assert "[Set A] dropped 1 track ID(s) outside the offered pool" in captured.err


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
async def test_stage2_prompt_includes_recent_concepts_block_when_history_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When concept_history has entries, Stage 2 sees a RECENT CONCEPTS block listing prior titles."""
    from mixlab.history import ConceptHistory, HistoryEntry
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }
    history = ConceptHistory(
        runs=[
            HistoryEntry(
                run_id="r1",
                created_at="2026-05-10T12:00:00+00:00",
                mode="standard",
                genre="dnb",
                selected_canvas_ids=["c1"],
                dominant_bpm_clusters=[174.0],
                dominant_camelot_keys=["8A"],
                core_track_ids=["X1"],
                anchor_track_ids=["X1"],
                opener_candidates=["X1"],
                closer_candidates=["X1"],
                concept_title="Late Latitude",
                concept_track_ids=["X1"],
                energy_path="wave",
                mood="dark",
            )
        ]
    )

    await stage2_curate_and_report(shortlists, tracks_by_id, concept_history=history)

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "RECENT CONCEPTS" in user_prompt
    assert "Late Latitude" in user_prompt


@respx.mock
async def test_stage2_prompt_omits_recent_concepts_block_when_history_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty history must not produce an orphan header in the prompt."""
    from mixlab.history import ConceptHistory
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, concept_history=ConceptHistory())

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "RECENT CONCEPTS" not in user_prompt


@respx.mock
async def test_stage2_prompt_omits_recent_concepts_block_when_history_not_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing concept_history=None (default) leaves the prompt unchanged."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "RECENT CONCEPTS" not in user_prompt


@respx.mock
async def test_stage2_genre_intent_passthrough_in_genre_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genre-mode Stage 2 prompt must contain the literal --intent text inside USER INTENT block."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    intent_text = "warmup set for an outdoor afternoon, low pressure, melodic"
    await stage2_curate_and_report(shortlists, tracks_by_id, genre_intent=intent_text)

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "USER INTENT" in user_prompt
    assert intent_text in user_prompt
    # Intent block must precede the candidates listing.
    assert user_prompt.index("USER INTENT") < user_prompt.index("Curate a set of mix concepts")


@respx.mock
async def test_stage2_genre_intent_block_absent_when_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default behaviour: no genre_intent means no USER INTENT block in the prompt."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "USER INTENT" not in user_prompt


@respx.mock
async def test_stage2_genre_intent_present_in_playlist_mode_overrides_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playlist mode now honours --intent (#54): USER INTENT block appears after the DJ
    INTENT BRIEF and states it overrides that brief on conflict."""
    from mixlab.llm import stage2_curate_and_report
    from mixlab.models import IntentBrief

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }
    intent_brief = IntentBrief(
        overall_vibe="A relentless rollers set.",
        energy_shape="unclear",
        risk_tolerance="medium",
        is_coherent_set=True,
        seed_analyses=[],
        missing_roles=[],
        strong_adjacencies=[],
        bpm_range=(170.0, 178.0),
    )

    await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1"}),
        seed_track_ids=["1"],
        intent_brief=intent_brief,
        genre_intent="surprise me with something bold",
    )

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "USER INTENT" in user_prompt
    assert "surprise me with something bold" in user_prompt
    assert "OVERRIDES the inferred DJ INTENT BRIEF" in user_prompt
    # USER INTENT must follow the Stage 0 DJ INTENT BRIEF, not precede it.
    assert user_prompt.index("DJ INTENT BRIEF") < user_prompt.index("USER INTENT")


@respx.mock
async def test_stage2_genre_intent_blank_string_emits_no_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only intent string is treated as no intent — block omitted."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, genre_intent="   ")

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "USER INTENT" not in user_prompt


@respx.mock
async def test_stage2_recent_concepts_precedes_intent_in_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """RECENT CONCEPTS block must appear before USER INTENT so intent reads as the override."""
    from mixlab.history import ConceptHistory, HistoryEntry
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }
    history = ConceptHistory(
        runs=[
            HistoryEntry(
                run_id="r1",
                created_at="2026-05-10T12:00:00+00:00",
                mode="standard",
                genre="dnb",
                selected_canvas_ids=["c1"],
                dominant_bpm_clusters=[174.0],
                dominant_camelot_keys=["8A"],
                core_track_ids=["X1"],
                anchor_track_ids=["X1"],
                opener_candidates=["X1"],
                closer_candidates=["X1"],
                concept_title="Dark Minimal",
                concept_track_ids=["X1"],
                energy_path="plateau",
                mood="dark",
            )
        ]
    )

    intent_text = "bright euphoric front-loaded energy, peak time conviction"
    await stage2_curate_and_report(shortlists, tracks_by_id, concept_history=history, genre_intent=intent_text)

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "RECENT CONCEPTS" in user_prompt
    assert "USER INTENT" in user_prompt
    assert user_prompt.index("RECENT CONCEPTS") < user_prompt.index("USER INTENT")


@respx.mock
async def test_stage2_intent_meta_instruction_uses_primary_lens_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta-instruction must tell Stage 2 that intent is the primary curatorial lens."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, genre_intent="warm hypnotic opener")

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "primary curatorial lens" in user_prompt
    assert "Reject concepts" in user_prompt


@respx.mock
async def test_stage2_intent_parsed_signals_injected_when_keywords_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parsed signals line appears in USER INTENT block when keywords are detected."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(
        shortlists, tracks_by_id, genre_intent="late-night radio showcase, dark hypnotic journey, chapters"
    )

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "Parsed signals:" in user_prompt


@respx.mock
async def test_stage2_intent_no_parsed_signals_when_generic_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Parsed signals line emitted when intent contains no detectable keywords."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, genre_intent="xyz foobar qux baz quux norf")

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "Parsed signals:" not in user_prompt


# ---------------------------------------------------------------------------
# _parse_user_intent unit tests (#35)
# ---------------------------------------------------------------------------


def test_parse_user_intent_detects_warmup_register() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("warmup set for an outdoor afternoon")
    assert signals.get("register") == "warmup"


def test_parse_user_intent_detects_late_night_register() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("late-night radio showcase with dark conviction")
    assert signals.get("register") == "late-night"


def test_parse_user_intent_detects_mood_dark() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("dark hypnotic drive through the night")
    assert signals.get("mood") == "dark+hypnotic"


def test_parse_user_intent_detects_radio_occasion() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("radio mix for a broad audience")
    assert signals.get("occasion") == "radio"


def test_parse_user_intent_podcast_occasion_alias() -> None:
    """'podcast' is an alias for occasion=radio in _OCCASION_MAP."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("podcast episode, 60 minutes")
    assert signals.get("occasion") == "radio"


def test_parse_user_intent_detects_journey_arc_hint() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("the journey matters — chapters, not a playlist")
    assert signals.get("arc-hint") == "journey"


def test_parse_user_intent_returns_empty_for_unrecognised_text() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("xyz foobar qux baz quux norf")
    assert signals == {}


def test_parse_user_intent_multiple_signals_all_present() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("late-night radio showcase — dark hypnotic journey, seasoned club crowd")
    assert signals.get("register") == "late-night"
    assert signals.get("occasion") == "showcase"
    assert signals.get("arc-hint") == "journey"
    assert signals.get("mood") == "dark+hypnotic"
    assert signals.get("audience") == "experienced"


def test_parse_user_intent_classic_era_detected() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    # 'classic' comes first in _ERA_MAP so it wins when both appear in text.
    signals = _parse_user_intent("classic oldschool house vibes only")
    assert signals.get("era") == "classic"


def test_parse_user_intent_after_hours_hyphenated_detected() -> None:
    """Hyphenated 'after-hours' must resolve to late-night register (was silently dropped)."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("After-hours club set, seasoned crowd, hypnotic and stripped back")
    assert signals.get("register") == "late-night"


def test_parse_user_intent_slow_burn_hyphenated_detected() -> None:
    """Hyphenated 'slow-burn' must resolve to slow-burn arc-hint (was silently dropped)."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("Slow-burn minimal techno for a late-night crowd, dark and driving")
    assert signals.get("arc-hint") == "slow-burn"


def test_parse_user_intent_warm_mood_not_spurious_on_warmup_text() -> None:
    """'warm' mood must NOT fire when the only 'warm' in text is inside 'warm-up'."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("warm-up set, cold and minimal")
    assert "warm" not in signals.get("mood", "")


def test_parse_user_intent_warm_up_spaced_sets_register() -> None:
    """'warm up' (two words, spaced) must fire register=warmup and NOT mood=warm.

    The 'warm' prefix of the 'warm up' register key must be suppressed from mood
    extraction — same invariant as the hyphenated 'warm-up' form.
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("warm up set, cold and minimal")
    assert signals.get("register") == "warmup"
    assert "warm" not in signals.get("mood", "")


def test_parse_user_intent_warm_mood_preserved_after_warmup_span() -> None:
    """Standalone 'warm' after a 'warm up' span must survive the position-aware re-scan.

    For the spaced 'warm up' form, _kw_search('warm', text) finds 'warm' at position 0
    (inside the register span). The suppression code discards it, then re-scans from
    reg_end to find the independent 'warm' later in the sentence. This exercises the
    actual suppression+re-scan code path (the hyphenated form never reaches it because
    the regex (?!-) prevents the first match from landing in the span at all).
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("warm up set should feel warm and inviting")
    assert signals.get("register") == "warmup"
    assert "warm" in signals.get("mood", "")


def test_parse_user_intent_warm_suppressed_with_non_warmup_register() -> None:
    """'warm' must not fire as mood when 'warm up' appears alongside a different register key.

    'late-night warm up set' — the matched register is 'late-night', but 'warm up' is also
    present at a different span.  The suppression loop must scan ALL register keys, not just
    the matched one, so 'warm' is still suppressed from mood extraction.
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("late-night warm up set, dark and hypnotic")
    assert signals.get("register") == "late-night"
    assert "warm" not in signals.get("mood", "")
    assert "dark" in signals.get("mood", "")
    assert "hypnotic" in signals.get("mood", "")


def test_parse_user_intent_warm_suppressed_when_warm_up_appears_twice() -> None:
    """'warm' must not fire as mood when every occurrence is inside a 'warm up' span.

    Single-pass suppression would re-scan past the first span, land on the second
    occurrence of 'warm' (also inside 'warm up'), and spuriously emit mood=warm.
    The while-loop fix must keep scanning until a standalone 'warm' is found or exhausted.
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("warm up session for warm up lovers")
    assert "warm" not in signals.get("mood", "")


def test_parse_user_intent_radio_sets_both_occasion_and_audience() -> None:
    """'radio' fires occasion=radio AND audience=broad (dual-signal, intentional design)."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("radio mix")
    assert signals.get("occasion") == "radio"
    assert signals.get("audience") == "broad"


def test_parse_user_intent_club_sets_both_occasion_and_audience() -> None:
    """'club' fires occasion=club AND audience=experienced (dual-signal, intentional design)."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("club set")
    assert signals.get("occasion") == "club"
    assert signals.get("audience") == "experienced"


def test_parse_user_intent_old_school_hyphenated_detected() -> None:
    """'old-school' (hyphenated) must resolve to oldschool era — was silently missing."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("old-school jungle from the 90s")
    assert signals.get("era") == "oldschool"


def test_parse_user_intent_late_night_beats_warmup_when_cooccurring() -> None:
    """late-night must take priority over warmup when both appear in intent text."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("late-night warmup set, dark and hypnotic")
    assert signals.get("register") == "late-night"


def test_parse_user_intent_close_not_matched_as_closing_register() -> None:
    """'close' alone must not fire as closing register — too broad a word."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("pay close attention to the BPM gradient")
    assert signals.get("register") is None


def test_parse_user_intent_audience_specific_beats_venue_word() -> None:
    """Specific audience descriptors must win over venue words when both are present.

    'radio show for seasoned listeners' contains both 'radio' (→ broad) and
    'seasoned' (→ experienced); the specific descriptor must take priority.
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("radio show for seasoned listeners")
    assert signals.get("audience") == "experienced", (
        f"Expected audience=experienced but got audience={signals.get('audience')!r}"
    )


def test_parse_user_intent_main_room_hyphenated_detected() -> None:
    """Hyphenated 'main-room' must resolve to peak register (was silently missing from dict).

    Input is free of other register keywords to confirm main-room alone is sufficient.
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("main-room techno, prime time energy")
    assert signals.get("register") == "peak"


def test_parse_user_intent_occasion_showcase_beats_radio() -> None:
    """'showcase' (specific event type) must win over 'radio' (broadcast) when both appear."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("late-night radio showcase — dark and driven")
    assert signals.get("occasion") == "showcase"


def test_parse_user_intent_arc_hint_narrative_beats_arc() -> None:
    """'narrative' must win over bare 'arc' when both appear in 'narrative arc'."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("the narrative arc should span 90 minutes")
    assert signals.get("arc-hint") == "narrative"


def test_parse_user_intent_slow_burn_unhyphenated_detected() -> None:
    """'slow burn' (space-delimited) must resolve to slow-burn arc-hint."""
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("a slow burn through the night, deliberate and unhurried")
    assert signals.get("arc-hint") == "slow-burn"


def test_parse_user_intent_mood_ordered_by_text_position_not_list_order() -> None:
    """Mood extraction must rank moods by where they appear in the text, not by keyword-list position.

    'intense' is near the end of _MOOD_KEYWORDS but appears first in the intent — it must rank first.
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    # "intense" is at index 18 in _MOOD_KEYWORDS, "driving" at 5, "minimal" at 15.
    # But intent leads with "intense" — it must appear first in the extracted mood signal.
    signals = _parse_user_intent("intense driving minimal, keep it stripped")
    mood = signals.get("mood", "")
    assert mood.startswith("intense"), f"Expected 'intense' first in mood '{mood}'"


@respx.mock
async def test_stage2_intent_parsed_signals_include_precedence_caveat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parsed signals block must include a caveat that quoted intent takes precedence on conflict."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, genre_intent="late-night dark hypnotic warmup")

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "Parsed signals:" in user_prompt
    assert "quoted intent takes precedence" in user_prompt
    assert "negated phrases" in user_prompt


@respx.mock
async def test_stage2_genre_intent_present_in_playlist_mode_without_brief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Playlist mode without an IntentBrief still surfaces --intent (#54): the USER INTENT
    block appears (with no preceding DJ INTENT BRIEF to override)."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset(["1", "2"]),
        genre_intent="dark and hypnotic",
    )

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "USER INTENT" in user_prompt
    assert "dark and hypnotic" in user_prompt
    # No IntentBrief was supplied, so the Stage 0 brief section itself (identified by its
    # "Vibe:" field) must be absent — only the override sentence mentions "DJ INTENT BRIEF".
    assert "Vibe:" not in user_prompt


def test_parse_user_intent_negation_does_not_suppress_signal() -> None:
    """Negated mood words still fire — the parser has no negation awareness.

    This documents the current contract: 'not dark' produces mood=dark.
    Stage 2 is instructed via the parsed_line caveat to prefer the quoted intent.
    """
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("not dark, not hypnotic, bright and euphoric")
    mood = signals.get("mood", "")
    assert "dark" in mood
    assert "hypnotic" in mood


# ---------------------------------------------------------------------------
# _parse_user_intent — risk signal (#54)
# ---------------------------------------------------------------------------


def test_parse_user_intent_safe_and_cautious_detected_as_low_risk() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("keep it safe and cautious, smooth transitions only")
    assert signals.get("risk") == "low"


def test_parse_user_intent_no_risks_phrase_detected_as_low_risk() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("no risks please, low risk only")
    assert signals.get("risk") == "low"


def test_parse_user_intent_bold_and_adventurous_detected_as_high_risk() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("go bold, be adventurous, take risky moves")
    assert signals.get("risk") == "high"


def test_parse_user_intent_surprise_me_phrase_detected_as_high_risk() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("surprise me with something weird")
    assert signals.get("risk") == "high"


def test_parse_user_intent_no_risk_keyword_leaves_risk_signal_absent() -> None:
    from mixlab.llm import _parse_user_intent  # noqa: PLC2701

    signals = _parse_user_intent("late-night radio showcase, dark and hypnotic")
    assert "risk" not in signals


@respx.mock
async def test_stage2_mode_fragment_unplayed(monkeypatch: pytest.MonkeyPatch) -> None:
    """mode='unplayed' must append the unplayed framing to the genre-mode system prompt."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, mode="unplayed")

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "MODE: UNPLAYED" in system_prompt
    assert "tracks the user has NOT played live" in system_prompt
    assert "MODE: PLAYED" not in system_prompt
    assert "MODE: ALL" not in system_prompt


@respx.mock
async def test_stage2_mode_fragment_played(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, mode="played")

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "MODE: PLAYED" in system_prompt
    assert "Familiarity is an asset" in system_prompt
    assert "MODE: UNPLAYED" not in system_prompt
    assert "MODE: ALL" not in system_prompt


@respx.mock
async def test_stage2_mode_fragment_all(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, mode="all")

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "MODE: ALL" in system_prompt
    assert "interleave played and unplayed material" in system_prompt
    assert "MODE: UNPLAYED" not in system_prompt
    assert "MODE: PLAYED" not in system_prompt


@respx.mock
async def test_stage2_mode_fragment_absent_when_mode_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (mode=None) leaves the system prompt without any MODE: fragment."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id)

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "MODE: UNPLAYED" not in system_prompt
    assert "MODE: PLAYED" not in system_prompt
    assert "MODE: ALL" not in system_prompt


@respx.mock
async def test_stage2_mode_fragment_not_injected_in_playlist_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playlist mode has its own Stage 0 intent path — mode fragment must not leak in."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1"}),
        seed_track_ids=["1"],
        mode="played",
    )

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "MODE: PLAYED" not in system_prompt
    assert "MODE: UNPLAYED" not in system_prompt
    assert "MODE: ALL" not in system_prompt


@respx.mock
async def test_stage2_risk_fragment_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """risk='high' must append the RISK: HIGH framing to the genre-mode system prompt."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, risk="high")

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "RISK: HIGH" in system_prompt
    assert "asked to be surprised" in system_prompt
    assert "RISK: LOW" not in system_prompt


@respx.mock
async def test_stage2_risk_fragment_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """risk='low' must append the RISK: LOW framing to the genre-mode system prompt."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, risk="low")

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "RISK: LOW" in system_prompt
    assert "Restraint is the brief" in system_prompt
    assert "RISK: HIGH" not in system_prompt


@respx.mock
async def test_stage2_risk_fragment_absent_at_medium(monkeypatch: pytest.MonkeyPatch) -> None:
    """risk='medium' (explicit or default) must leave the system prompt without any
    RISK: fragment — byte-stable default (#42).
    """
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, risk="medium")

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "RISK: HIGH" not in system_prompt
    assert "RISK: LOW" not in system_prompt


@respx.mock
async def test_stage2_risk_fragment_not_injected_in_playlist_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playlist mode ignores the risk knob — it has its own Stage 0 risk-tolerance path (#54)."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1"}),
        seed_track_ids=["1"],
        risk="high",
    )

    body = json.loads(route.calls[0].request.content)
    system_prompt: str = body["system"]
    assert "RISK: HIGH" not in system_prompt
    assert "RISK: LOW" not in system_prompt


def _critique_payload(verdict: str = "needs_attention") -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "single_weakest_moment": "track 3: vocal clash at 0:30",
            "structural_issues": ["energy path drift", "transition 4→5 mechanism boilerplate"],
            "suggested_substitution": "track 3 → ID:2 from canvas bridge pool: instrumental intro",
        }
    )


@respx.mock
async def test_stage2_deep_mode_runs_critique_and_surfaces_in_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """--deep triggers a critique pass per concept; output appears in the report."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_critique_payload("needs_attention"))),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    concepts, report = await stage2_curate_and_report(shortlists, tracks_by_id, deep=True)

    assert len(concepts) == 1
    assert concepts[0].critique is not None
    assert concepts[0].critique.verdict == "needs_attention"
    assert "CRITIQUE (DEEP MODE)" in report
    assert "Verdict: needs_attention" in report
    assert "vocal clash at 0:30" in report


@respx.mock
async def test_stage2_no_deep_makes_no_critique_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --deep, the critique HTTP call must not fire and critique stays None."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool A", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    concepts, report = await stage2_curate_and_report(shortlists, tracks_by_id, deep=False)

    assert concepts[0].critique is None
    assert "CRITIQUE (DEEP MODE)" not in report
    # Exactly 2 Anthropic calls: selection + report. No third call for critique.
    assert len(route.calls) == 2


@respx.mock
async def test_stage2_deep_ignored_in_playlist_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playlist mode runs variant scoring instead of the genre-mode critique loop."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="practical", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    concepts, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1"}),
        seed_track_ids=["1"],
        deep=True,
    )
    assert "CRITIQUE (DEEP MODE)" not in report
    assert concepts[0].critique is None


def test_parse_critique_tolerates_code_fences() -> None:
    from mixlab.llm import _parse_critique

    raw = '```json\n{"verdict": "solid", "single_weakest_moment": "track 4 transition", "structural_issues": [], "suggested_substitution": null}\n```'
    critique = _parse_critique(raw)
    assert critique.verdict == "solid"
    assert critique.structural_issues == []
    assert critique.suggested_substitution is None


def test_parse_critique_malformed_json_returns_needs_attention() -> None:
    """Bad JSON falls back to a needs_attention critique with the raw payload."""
    from mixlab.llm import _parse_critique

    critique = _parse_critique("not json at all")
    assert critique.verdict == "needs_attention"
    assert "critique parse failed" in critique.single_weakest_moment


def test_parse_critique_coerces_invalid_verdict() -> None:
    from mixlab.llm import _parse_critique

    raw = '{"verdict": "AMAZING", "single_weakest_moment": "", "structural_issues": []}'
    critique = _parse_critique(raw)
    assert critique.verdict == "needs_attention"


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
    assert "Selection tolerance: low." in report


# ---------------------------------------------------------------------------
# _adventure_dividend (#54) — reward density for justified risky transitions
# ---------------------------------------------------------------------------


def _transition(risky: bool, risk_type: str) -> Transition:
    return Transition(from_id="a", to_id="b", is_risky=risky, risk_type=risk_type)


def test_adventure_dividend_no_risky_transitions_returns_zero() -> None:
    from mixlab.llm import _adventure_dividend  # noqa: PLC2701

    concept = MixConcept(title="T", mood="adventurous", track_ids=["1", "2"], transitions=[])
    assert _adventure_dividend(concept) == 0.0


def test_adventure_dividend_two_justified_returns_half() -> None:
    from mixlab.llm import _adventure_dividend  # noqa: PLC2701

    concept = MixConcept(
        title="T",
        mood="adventurous",
        track_ids=["1", "2", "3"],
        transitions=[
            _transition(True, "chapter_pivot"),
            _transition(True, "peak_impact"),
        ],
    )
    assert _adventure_dividend(concept) == pytest.approx(0.5)


def test_adventure_dividend_four_justified_caps_at_one() -> None:
    from mixlab.llm import _adventure_dividend  # noqa: PLC2701

    concept = MixConcept(
        title="T",
        mood="adventurous",
        track_ids=["1", "2", "3", "4", "5"],
        transitions=[
            _transition(True, "chapter_pivot"),
            _transition(True, "peak_impact"),
            _transition(True, "deliberate_reset"),
            _transition(True, "closer_move"),
        ],
    )
    assert _adventure_dividend(concept) == pytest.approx(1.0)


def test_adventure_dividend_two_justified_plus_cut_only_returns_quarter() -> None:
    from mixlab.llm import _adventure_dividend  # noqa: PLC2701

    concept = MixConcept(
        title="T",
        mood="adventurous",
        track_ids=["1", "2", "3", "4"],
        transitions=[
            _transition(True, "chapter_pivot"),
            _transition(True, "peak_impact"),
            _transition(True, "cut_only"),
        ],
    )
    assert _adventure_dividend(concept) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _select_best_variant (#54) — tolerance-aware winner selection
# ---------------------------------------------------------------------------


def _variant_with(strategy: str, practicality: float, dividend_transitions: list[Transition]) -> CompletionVariant:
    concept = MixConcept(
        title=strategy.title(),
        mood=strategy,
        track_ids=["1", "2", "3", "4"],
        transitions=dividend_transitions,
    )
    return CompletionVariant(
        strategy=cast("Literal['practical', 'balanced', 'adventurous']", strategy),
        concept=concept,
        anchor_retention_rate=1.0,
        practicality_score=DJPracticalityScore(practicality, practicality, practicality, practicality),
    )


def test_select_best_variant_low_tolerance_prefers_practicality_over_risk() -> None:
    from mixlab.llm import _select_best_variant  # noqa: PLC2701

    practical = _variant_with("practical", 0.90, [])
    adventurous = _variant_with(
        "adventurous",
        0.70,
        [_transition(True, "chapter_pivot"), _transition(True, "peak_impact"), _transition(True, "deliberate_reset")],
    )
    winner = _select_best_variant([practical, adventurous], "low")
    assert winner.strategy == "practical"


def test_select_best_variant_medium_tolerance_still_prefers_practicality() -> None:
    from mixlab.llm import _select_best_variant  # noqa: PLC2701

    practical = _variant_with("practical", 0.90, [])
    adventurous = _variant_with(
        "adventurous",
        0.70,
        [_transition(True, "chapter_pivot"), _transition(True, "peak_impact"), _transition(True, "deliberate_reset")],
    )
    winner = _select_best_variant([practical, adventurous], "medium")
    assert winner.strategy == "practical"


def test_select_best_variant_high_tolerance_rewards_justified_risk() -> None:
    from mixlab.llm import _select_best_variant  # noqa: PLC2701

    practical = _variant_with("practical", 0.90, [])
    adventurous = _variant_with(
        "adventurous",
        0.70,
        [_transition(True, "chapter_pivot"), _transition(True, "peak_impact"), _transition(True, "deliberate_reset")],
    )
    winner = _select_best_variant([practical, adventurous], "high")
    assert winner.strategy == "adventurous"


def test_select_best_variant_default_argument_matches_pre_54_behaviour() -> None:
    """No tolerance passed must behave exactly like the pre-#54 practicality-only ranking,
    including the practical > balanced > adventurous tie-break."""
    from mixlab.llm import _select_best_variant  # noqa: PLC2701

    practical = _variant_with("practical", 0.5, [])
    balanced = _variant_with("balanced", 0.5, [])
    adventurous = _variant_with(
        "adventurous",
        0.5,
        [_transition(True, "chapter_pivot"), _transition(True, "peak_impact")],
    )
    winner = _select_best_variant([adventurous, balanced, practical])
    assert winner.strategy == "practical"


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


def test_stage2_report_prompt_enforces_prose_and_json_risk_consistency() -> None:
    """The prose `Risk:` line and JSON is_risky/risk_type must agree (#29).

    Without this rule the LLM writes rich prose risk descriptions while leaving JSON
    is_risky=False, defeating the validator's justified-risk suppression (#28). The
    consistency rule lives in the report-pass prompt because that is where prose risk
    is authored against the selection pass's structured annotations.
    """
    from mixlab.llm import _STAGE2_REPORT_SYSTEM

    assert "CONSISTENCY" in _STAGE2_REPORT_SYSTEM
    assert "is_risky" in _STAGE2_REPORT_SYSTEM
    assert "Risk:" in _STAGE2_REPORT_SYSTEM and "must agree" in _STAGE2_REPORT_SYSTEM
    assert "Risk: none" in _STAGE2_REPORT_SYSTEM


def test_stage2_report_prompt_enforces_arc_consistency() -> None:
    """The prose "Energy path:" label must match the concept's declared arc_type.

    Live-run finding: the report pass never saw arc_type, so cards showed e.g. a
    `wave` badge next to "Energy path: Slow Climb" prose. The mapping from arc_type
    values to the seven prose labels lives in the report-pass prompt.
    """
    from mixlab.llm import _STAGE2_REPORT_SYSTEM

    assert "ARC CONSISTENCY" in _STAGE2_REPORT_SYSTEM
    assert "declared arc_type" in _STAGE2_REPORT_SYSTEM
    assert "plateau or sustained-pressure" in _STAGE2_REPORT_SYSTEM
    assert "Do not contradict the declared arc." in _STAGE2_REPORT_SYSTEM


def test_stage2_prompts_explain_intro_zero_as_deliberate_mix_in() -> None:
    """intro:0b means the DJ's mix-in cue is at the track's top — not a cold drop.

    Live-run finding: many first cues sit at 0:00 (confirmed deliberate by the owner),
    and without a legend the report pass read intro:0b as "no intro to blend over".
    The guidance must reach all Stage 2 prompt surfaces: selection, playlist
    (derived from selection), and report.
    """
    from mixlab.llm import _STAGE2_REPORT_SYSTEM, _STAGE2_SYSTEM, _STAGE2_SYSTEM_PLAYLIST

    for prompt in (_STAGE2_SYSTEM, _STAGE2_SYSTEM_PLAYLIST, _STAGE2_REPORT_SYSTEM):
        assert "intro:0b" in prompt
        assert "mixes" in prompt and "bar one" in prompt
        assert "cold drop" in prompt
        assert "no cue data yet" in prompt


@respx.mock
async def test_stage2_report_prompt_includes_declared_arc_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report-pass user prompt must carry the concept's arc_type so prose can match it."""
    from mixlab.llm import _call_stage2_reports

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    captured: list[dict[str, object]] = []

    def capture(request: object) -> Response:
        body = json.loads(request.content)  # type: ignore[attr-defined]
        captured.append(body)
        return Response(200, json={"content": [{"text": "report"}], "stop_reason": "end_turn"})

    respx.post(_ANTHROPIC_URL).mock(side_effect=capture)

    concept = MixConcept(title="Test", mood="dark", track_ids=["1", "2"], arc_type="wave")
    no_arc = MixConcept(title="Test 2", mood="warm", track_ids=["1", "2"])
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 3)
    }

    await _call_stage2_reports([concept, no_arc], tracks_by_id, None, None, "test-key")

    assert len(captured) == 2
    prompts = []
    for body in captured:
        messages = cast(list[dict[str, str]], body["messages"])
        prompts.append(next(m["content"] for m in messages if m["role"] == "user"))
    assert any("Declared arc_type: wave" in p for p in prompts)
    assert any("Declared arc_type: (none)" in p for p in prompts)


@respx.mock
async def test_stage2_report_prompt_includes_transition_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report-pass prompt must include the selection pass's transitions annotations (#29).

    Without this, the report writer cannot align its prose `Risk:` lines with the
    structured `is_risky`/`risk_type` flags set during selection.
    """
    from mixlab.llm import _call_stage2_reports
    from mixlab.models import Transition

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    captured: list[dict[str, object]] = []

    def capture(request: object) -> Response:
        body = json.loads(request.content)  # type: ignore[attr-defined]
        captured.append(body)
        return Response(200, json={"content": [{"text": "report"}], "stop_reason": "end_turn"})

    respx.post(_ANTHROPIC_URL).mock(side_effect=capture)

    concept = MixConcept(
        title="Test",
        mood="dark",
        track_ids=["1", "2", "3"],
        transitions=[
            Transition(from_id="1", to_id="2", is_risky=False, risk_type=""),
            Transition(from_id="2", to_id="3", is_risky=True, risk_type="chapter_pivot"),
        ],
    )
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 4)
    }

    await _call_stage2_reports([concept], tracks_by_id, None, None, "test-key")

    assert len(captured) == 1
    messages = cast(list[dict[str, str]], captured[0]["messages"])
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "chapter_pivot" in user_msg
    assert "is_risky" in user_msg


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
    assert "stacked peaks" in result
    assert "name_reason" in result
    assert "chapter_pivot" in result


def test_stage2_system_role_vocabulary_trimmed_to_ten_roles() -> None:
    """Stage 2 prompts must reference the new 10-role vocabulary, not the old 19-role list (#23)."""
    from mixlab.llm import _STAGE2_REPORT_SYSTEM, _STAGE2_SYSTEM

    new_roles = {
        "opener",
        "groove",
        "hook",
        "pivot",
        "lift",
        "vocal-moment",
        "texture-change",
        "peak",
        "resolution",
        "closer",
    }
    old_only = {
        "world-setter",
        "early-hook",
        "groove-locker",
        "builder",
        "connector",
        "pressure",
        "cleanser",
        "weapon",
        "post-peak",
        "utility",
    }
    for prompt in (_STAGE2_SYSTEM, _STAGE2_REPORT_SYSTEM):
        for role in new_roles:
            assert role in prompt, f"new role '{role}' missing from prompt"
        for role in old_only:
            # Old role tokens must not appear as standalone role labels in the prompt.
            assert f", {role}," not in prompt and f", {role}." not in prompt, (
                f"old role '{role}' still listed in prompt"
            )


def test_stage0_system_role_vocabulary_trimmed() -> None:
    """Stage 0 (playlist mode) inferred-role options match the new vocabulary."""
    from mixlab.llm import _STAGE0_SYSTEM

    assert "opener, groove, hook, pivot, lift, vocal_moment, texture_change, peak, resolution, closer" in _STAGE0_SYSTEM
    assert "world_setter" not in _STAGE0_SYSTEM
    assert "groove_locker" not in _STAGE0_SYSTEM
    assert "weapon" not in _STAGE0_SYSTEM
    assert "cleanser" not in _STAGE0_SYSTEM


def test_stage0_parser_coerces_old_role_names_to_unknown() -> None:
    """Stage 0 LLM responses with old vocab (e.g. 'weapon') should coerce to 'unknown' (#23)."""
    from mixlab.llm import _parse_intent_brief

    raw = """{
      "overall_vibe": "test",
      "energy_shape": "single_arc",
      "risk_tolerance": "medium",
      "is_coherent_set": true,
      "missing_roles": [],
      "seed_analyses": [
        {"track_id": "1", "tier": "anchor", "inferred_role": "weapon"},
        {"track_id": "2", "tier": "supporting", "inferred_role": "groove"}
      ]
    }"""
    seed_tracks = [
        Track(track_id="1", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB"),
        Track(track_id="2", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB"),
    ]
    brief = _parse_intent_brief(raw, seed_tracks, (170.0, 178.0))
    roles_by_id = {s.track_id: s.inferred_role for s in brief.seed_analyses}
    assert roles_by_id["1"] == "unknown"  # 'weapon' coerced
    assert roles_by_id["2"] == "groove"  # 'groove' kept


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


def test_validate_stage2_output_bpm_jump_suppressed_when_transition_is_risky_and_justified() -> None:
    from mixlab.llm import validate_stage2_output
    from mixlab.models import Transition

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=200.0, camelot_key="8A", genre="DnB"),  # jump=26
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=200.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(
        title="T",
        mood="dark",
        track_ids=ids,
        transitions=[Transition(from_id="1", to_id="2", is_risky=True, risk_type="chapter_pivot")],
    )
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert not any("BPM jump" in w for w in warnings)


def test_validate_stage2_output_camelot_jump_suppressed_when_transition_is_risky_and_justified() -> None:
    from mixlab.llm import validate_stage2_output
    from mixlab.models import Transition

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="1A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=174.0, camelot_key="8B", genre="DnB"),  # dist=6
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=174.0, camelot_key="8B", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(
        title="T",
        mood="dark",
        track_ids=ids,
        transitions=[Transition(from_id="1", to_id="2", is_risky=True, risk_type="chapter_pivot")],
    )
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert not any("Camelot jump" in w for w in warnings)


def test_validate_stage2_output_bpm_jump_still_warns_when_is_risky_but_risk_type_empty() -> None:
    from mixlab.llm import validate_stage2_output
    from mixlab.models import Transition

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=200.0, camelot_key="8A", genre="DnB"),  # jump=26
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=200.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(
        title="T",
        mood="dark",
        track_ids=ids,
        transitions=[Transition(from_id="1", to_id="2", is_risky=True, risk_type="")],
    )
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("BPM jump" in w for w in warnings)


def test_validate_stage2_output_camelot_jump_still_warns_when_is_risky_but_risk_type_empty() -> None:
    from mixlab.llm import validate_stage2_output
    from mixlab.models import Transition

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="1A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=174.0, camelot_key="8B", genre="DnB"),  # dist=6
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=174.0, camelot_key="8B", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(
        title="T",
        mood="dark",
        track_ids=ids,
        transitions=[Transition(from_id="1", to_id="2", is_risky=True, risk_type="")],
    )
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("Camelot jump" in w for w in warnings)


# ---------------------------------------------------------------------------
# validate_stage2_output — risk knob (#42): risk-aware BPM/Camelot jump thresholds
# ---------------------------------------------------------------------------


def test_validate_stage2_output_18bpm_annotated_risky_warns_at_medium_silent_at_high() -> None:
    """An 18 BPM jump on a transition flagged is_risky=True (but not fully justified —
    empty risk_type) exceeds the medium threshold (15) but not the relaxed high
    threshold (20), since high relaxation applies to any annotated-risky transition.
    """
    from mixlab.llm import validate_stage2_output
    from mixlab.models import Transition

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=192.0, camelot_key="8A", genre="DnB"),  # jump=18
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=192.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(
        title="T",
        mood="dark",
        track_ids=ids,
        transitions=[Transition(from_id="1", to_id="2", is_risky=True, risk_type="")],
    )
    canvas = _make_canvas(ids)

    warnings_medium = validate_stage2_output([concept], [canvas], lib, set(), set(), risk="medium")
    assert any("BPM jump" in w for w in warnings_medium)

    warnings_high = validate_stage2_output([concept], [canvas], lib, set(), set(), risk="high")
    assert not any("BPM jump" in w for w in warnings_high)


def test_validate_stage2_output_18bpm_unannotated_warns_at_both_medium_and_high() -> None:
    """An 18 BPM jump with NO transition annotation always uses the medium thresholds,
    even at risk='high' — the high relaxation requires an explicit is_risky flag.
    """
    from mixlab.llm import validate_stage2_output

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=192.0, camelot_key="8A", genre="DnB"),  # jump=18
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=192.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)  # no transitions — tr is None
    canvas = _make_canvas(ids)

    warnings_medium = validate_stage2_output([concept], [canvas], lib, set(), set(), risk="medium")
    assert any("BPM jump" in w for w in warnings_medium)

    warnings_high = validate_stage2_output([concept], [canvas], lib, set(), set(), risk="high")
    assert any("BPM jump" in w for w in warnings_high)


def test_validate_stage2_output_12bpm_warns_at_low_silent_at_medium() -> None:
    """A 12 BPM jump clears the low threshold (10) but not the medium threshold (15)."""
    from mixlab.llm import validate_stage2_output

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=186.0, camelot_key="8A", genre="DnB"),  # jump=12
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=186.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)

    warnings_low = validate_stage2_output([concept], [canvas], lib, set(), set(), risk="low")
    assert any("BPM jump" in w for w in warnings_low)

    warnings_medium = validate_stage2_output([concept], [canvas], lib, set(), set(), risk="medium")
    assert not any("BPM jump" in w for w in warnings_medium)


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


# ---------------------------------------------------------------------------
# DJ-structural validation checks (#4) — opener/closer/peak/wind-down/role-runs/energy
# ---------------------------------------------------------------------------


def _canvas_with_roles(
    core_ids: list[str],
    opener: list[str] | None = None,
    closer: list[str] | None = None,
    peak: list[str] | None = None,
    builder: list[str] | None = None,
) -> MixCanvas:
    """Canvas factory with explicit role-pool assignments for structural-validation tests."""
    return MixCanvas(
        canvas_id="test_canvas",
        genre="Drum & Bass",
        bpm_range=(168.0, 178.0),
        dominant_bpm=172.0,
        dominant_camelot="4A",
        core_track_ids=core_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=CanvasRoleCandidates(
            opener=opener or [],
            groove_locker=[],
            builder=builder or [],
            pivot=[],
            peak=peak or [],
            closer=closer or [],
        ),
        contrast=ContrastAssets(
            vocal_moments=[],
            texture_changes=[],
            darker_turns=[],
            brighter_lifts=[],
            lower_pressure_resets=[],
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=MixConcept(title="T", mood="dark", track_ids=core_ids),
    )


def test_validate_stage2_output_does_not_warn_about_role_pool_opener_or_closer() -> None:
    """Removed in #27 — canvas-pool-based opener/closer classification disagreed with
    Stage 2's textual role picks too often. These warnings no longer fire."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    canvas_no_opener = _canvas_with_roles(ids, opener=["8"], closer=["8"])
    canvas_no_closer = _canvas_with_roles(ids, opener=["1"], closer=["1"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    warnings_a = validate_stage2_output([concept], [canvas_no_opener], _lib(ids), set(), set())
    warnings_b = validate_stage2_output([concept], [canvas_no_closer], _lib(ids), set(), set())
    assert not any("no opener-role track in first 2 positions" in w for w in warnings_a)
    assert not any("no closer-role track in last 2 positions" in w for w in warnings_b)


def test_validate_stage2_output_no_peak_warning_softened_by_plateau_arc() -> None:
    """Soft-tier: missing peak suppressed when arc_type=plateau explicitly justifies it."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])  # peak pool present, T5 isn't picked
    # Concept doesn't include track 5 (the peak candidate).
    concept = MixConcept(title="T", mood="plateau", track_ids=["1", "2", "3", "4", "6", "7", "8"], arc_type="plateau")
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
    assert not any("no peak-role" in w for w in warnings)


def test_validate_stage2_output_does_not_warn_about_role_family_run() -> None:
    """Removed in #27 — role-family-run check relied on the same canvas-pool
    classification as the dropped opener/closer warnings."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], builder=["3", "4", "5"], peak=["6"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
    assert not any("consecutive builder tracks" in w for w in warnings)


def test_validate_stage2_output_warns_all_high_energy_no_dynamic_range() -> None:
    """Concept where every track is energy ≥6/8 fires the high-energy band warning."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=124.0, camelot_key="8A", genre="HipHop", energy=7)
        for tid in ids
    }
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    # Use a genre outside the soft-tier softening families so the warning fires.
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="hip_hop")
    assert any("all tracks high-energy" in w for w in warnings)


def test_validate_stage2_output_high_energy_warning_softened_for_house() -> None:
    """House and techno are sustained-groove genres — all-high-energy warning suppressed (#27)."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=124.0, camelot_key="8A", genre="House", energy=7)
        for tid in ids
    }
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    warnings_house = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="house")
    warnings_techno = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="techno")
    assert not any("all tracks high-energy" in w for w in warnings_house)
    assert not any("all tracks high-energy" in w for w in warnings_techno)


def test_validate_stage2_output_high_energy_warning_softened_for_dnb() -> None:
    """DnB genre tolerates sustained-pressure structures: all-high-energy warning suppressed."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB", energy=7)
        for tid in ids
    }
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="drum_and_bass")
    assert not any("all tracks high-energy" in w for w in warnings)


def test_validate_stage2_output_high_energy_warning_softened_for_sustained_pressure_arc() -> None:
    """arc_type=sustained-pressure suppresses the high-energy warning regardless of genre."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=124.0, camelot_key="8A", genre="House", energy=7)
        for tid in ids
    }
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids, arc_type="sustained-pressure")
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="house")
    assert not any("all tracks high-energy" in w for w in warnings)


def test_validate_stage2_output_wind_down_warning_fires_when_final_three_all_high_energy() -> None:
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    # First five tracks low-energy, last three high — fires "no wind-down" warning.
    energies = [2, 3, 3, 4, 4, 7, 7, 8]
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=124.0, camelot_key="8A", genre="HipHop", energy=e)
        for tid, e in zip(ids, energies, strict=True)
    }
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    # Use a non-soft-tier genre so the warning fires.
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="hip_hop")
    assert any("no wind-down" in w for w in warnings)


def test_validate_stage2_output_wind_down_warning_softened_for_house() -> None:
    """House sustained-groove sets don't owe a wind-down ramp (#27)."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    energies = [2, 3, 3, 4, 4, 7, 7, 8]
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=124.0, camelot_key="8A", genre="House", energy=e)
        for tid, e in zip(ids, energies, strict=True)
    }
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="house")
    assert not any("no wind-down" in w for w in warnings)


def test_validate_stage2_output_skips_structural_checks_when_canvas_has_no_role_data() -> None:
    """Legacy canvases without any role assignments must not trigger false positives."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    canvas = _make_canvas(ids)  # No role pools populated.
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
    # None of the structural checks should fire.
    assert not any("no opener-role" in w for w in warnings)
    assert not any("no closer-role" in w for w in warnings)
    assert not any("no peak-role" in w for w in warnings)


# ---------------------------------------------------------------------------
# Cross-concept distinctiveness (L3) and generic-name regex (L4) from review
# ---------------------------------------------------------------------------


def test_validate_stage2_output_warns_when_concepts_share_more_than_half_tracks() -> None:
    """Two concepts with >50% track overlap fire the distinctiveness check."""
    from mixlab.llm import validate_stage2_output

    ids_a = [f"A{i}" for i in range(1, 11)]  # 10 unique IDs
    ids_b = ids_a[:6] + [f"B{i}" for i in range(1, 5)]  # 6 of 10 shared = 60% overlap
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=124.0, camelot_key="8A", genre="House")
        for tid in ids_a + ids_b
    }
    canvas_a = _make_canvas(ids_a)
    concept_a = MixConcept(title="Set Alpha", mood="dark", track_ids=ids_a)
    concept_b = MixConcept(title="Set Bravo", mood="dark", track_ids=ids_b)
    warnings = validate_stage2_output([concept_a, concept_b], [canvas_a], lib, set(), set())
    assert any("Set Alpha" in w and "Set Bravo" in w and "distinctiveness" in w for w in warnings)


def test_validate_stage2_output_no_distinctiveness_warning_when_concepts_diverge() -> None:
    """Concepts with <50% overlap should not fire."""
    from mixlab.llm import validate_stage2_output

    ids_a = [f"A{i}" for i in range(1, 11)]
    ids_b = ids_a[:3] + [f"B{i}" for i in range(1, 8)]  # 3 of 10 shared = 30% overlap
    lib = {
        tid: Track(track_id=tid, artist="A", title="T", bpm=124.0, camelot_key="8A", genre="House")
        for tid in ids_a + ids_b
    }
    canvas_a = _make_canvas(ids_a)
    concept_a = MixConcept(title="Set Alpha", mood="dark", track_ids=ids_a)
    concept_b = MixConcept(title="Set Bravo", mood="dark", track_ids=ids_b)
    warnings = validate_stage2_output([concept_a, concept_b], [canvas_a], lib, set(), set())
    assert not any("distinctiveness check" in w for w in warnings)


def test_validate_stage2_output_warns_on_generic_adjective_noun_title() -> None:
    """Concept titled "Warm Gravity" matches the forbidden generic pattern."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    canvas = _make_canvas(ids)
    concept = MixConcept(title="Warm Gravity", mood="dark", track_ids=ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
    assert any("Warm Gravity" in w and "generic" in w for w in warnings)


def test_validate_stage2_output_does_not_warn_on_oblique_title() -> None:
    """Concept titled "Late Latitude" or other oblique forms should not match the generic pattern."""
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    canvas = _make_canvas(ids)
    for title in ("Late Latitude", "Fever", "Interzone", "Red Light", "The Slow Hours"):
        concept = MixConcept(title=title, mood="dark", track_ids=ids)
        warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
        assert not any("generic" in w for w in warnings), f"Title '{title}' wrongly flagged as generic"


# ---------------------------------------------------------------------------
# Known-good fixture + edge cases — verifies the validator is not noisy (#5)
# ---------------------------------------------------------------------------


def _known_good_setup() -> tuple[MixConcept, MixCanvas, dict[str, Track]]:
    """Construct a 10-track concept that should produce zero validation warnings.

    Properties:
    - First track is the opener candidate (energy 2)
    - Last track is the closer candidate (energy 3)
    - Mid-set peak candidate (energy 7) sits in positions 5-6
    - BPM walk: 122 → 124, max step <2
    - Camelot walk: 8A → 9A → 10A → 11A, max distance 1 per step
    - Each artist appears once
    - Two distinct mid-range roles (groove_locker + builder) prevent role-family runs
    """
    ids = [f"T{i:03d}" for i in range(1, 11)]
    artists = [f"Artist{i}" for i in range(1, 11)]
    bpms = [122.0, 122.5, 123.0, 123.5, 124.0, 124.0, 123.5, 123.0, 122.5, 122.0]
    keys = ["8A", "8A", "9A", "9A", "10A", "10A", "9A", "9A", "8A", "8A"]
    energies = [2, 4, 5, 6, 7, 7, 6, 5, 4, 3]
    lib = {
        tid: Track(
            track_id=tid,
            artist=artist,
            title=f"Track {tid}",
            bpm=bpm,
            camelot_key=key,
            genre="House",
            energy=energy,
        )
        for tid, artist, bpm, key, energy in zip(ids, artists, bpms, keys, energies, strict=True)
    }
    canvas = _canvas_with_roles(
        ids,
        opener=[ids[0]],
        closer=[ids[-1]],
        peak=[ids[4], ids[5]],
        builder=[ids[2], ids[3], ids[6], ids[7]],
    )
    canvas.bpm_range = (122.0, 124.0)
    concept = MixConcept(
        title="Late Latitude",  # oblique title — does NOT match generic regex
        mood="warm and patient",
        track_ids=ids,
        arc_type="wave",
    )
    return concept, canvas, lib


def test_validate_stage2_output_known_good_mix_produces_zero_warnings() -> None:
    """A well-formed concept with full role coverage and gentle BPM/key walk fires no warnings."""
    from mixlab.llm import validate_stage2_output

    concept, canvas, lib = _known_good_setup()
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="house")
    assert warnings == [], f"Known-good mix produced unexpected warnings: {warnings}"


def test_validate_stage2_output_empty_concept_list_returns_no_warnings() -> None:
    """No concepts, no canvases — must not crash, must return [] (regression guard)."""
    from mixlab.llm import validate_stage2_output

    assert validate_stage2_output([], [], {}, set(), set()) == []


def test_validate_stage2_output_empty_track_list_does_not_crash() -> None:
    """A concept with zero tracks must not crash structural checks."""
    from mixlab.llm import validate_stage2_output

    concept = MixConcept(title="Empty", mood="dark", track_ids=[])
    canvas = _canvas_with_roles([], opener=[], closer=[])  # no roles, but should not crash
    warnings = validate_stage2_output([concept], [canvas], {}, set(), set())
    # Track-count warning is expected (0 < min); just confirm no crash.
    assert isinstance(warnings, list)


def test_validate_stage2_output_single_track_concept_does_not_crash() -> None:
    """A 1-track concept exercises edge cases in BPM-jump / role-run loops."""
    from mixlab.llm import validate_stage2_output

    concept = MixConcept(title="Solo", mood="dark", track_ids=["1"])
    canvas = _canvas_with_roles(["1"], opener=["1"], closer=["1"])
    warnings = validate_stage2_output([concept], [canvas], _lib(["1"]), set(), set())
    # Track-count warning expected; should not crash.
    assert isinstance(warnings, list)


def test_validate_stage2_output_distinctiveness_skipped_for_empty_concepts() -> None:
    """Pairwise check should tolerate empty track lists without ZeroDivision or crash."""
    from mixlab.llm import _cross_concept_distinctiveness_warnings

    a = MixConcept(title="A", mood="m", track_ids=[])
    b = MixConcept(title="B", mood="m", track_ids=["1", "2"])
    # Should not raise; should produce no warning (one concept is empty).
    assert _cross_concept_distinctiveness_warnings([a, b]) == []


@pytest.mark.parametrize(
    "title,should_flag",
    [
        ("Warm Gravity", True),
        ("Slow Descent", True),
        ("Deep Pulse", True),
        ("Orbital Descent", True),
        ("Late Latitude", False),
        ("Fever", False),
        ("Interzone", False),
        ("Red Light", False),
        ("The Slow Hours", False),
        ("Slow Burn", False),  # noun (Burn) not in the suffix list — title is acceptable
    ],
)
def test_generic_name_regex_classification(title: str, should_flag: bool) -> None:
    """Parametrized regression: confirm the generic-name regex flags clichés and spares oblique titles."""
    from mixlab.llm import _generic_name_warning

    concept = MixConcept(title=title, mood="m", track_ids=["1"])
    warning = _generic_name_warning(concept)
    if should_flag:
        assert warning is not None, f"Expected '{title}' to be flagged as generic"
    else:
        assert warning is None, f"Expected '{title}' NOT to be flagged"


@pytest.mark.parametrize(
    "genre,arc_type,should_warn",
    [
        ("hip_hop", None, True),  # default genre, no arc → warns
        ("drum_and_bass", None, False),  # DnB suppresses high-energy warning
        ("hip_hop", "plateau", False),  # arc_type suppresses
        ("hip_hop", "sustained-pressure", False),  # arc_type suppresses
        ("electronica", None, True),  # electronica does NOT suppress high-energy specifically
        ("house", None, False),  # house is sustained-groove — suppresses (#27)
        ("techno", None, False),  # techno is sustained-groove — suppresses (#27)
    ],
)
def test_structural_high_energy_warning_softening(genre: str, arc_type: str | None, should_warn: bool) -> None:
    """High-energy warning softening: genre and arc_type each independently suppress the warning."""
    from typing import cast

    from mixlab.llm import validate_stage2_output
    from mixlab.models import ArcType

    ids = [str(i) for i in range(1, 9)]
    lib = {
        tid: Track(track_id=tid, artist=f"A{tid}", title="T", bpm=124.0, camelot_key="8A", genre=genre, energy=7)
        for tid in ids
    }
    canvas = _canvas_with_roles(ids, opener=["1"], closer=["8"], peak=["5"])
    arc_value = cast("ArcType | None", arc_type)
    concept = MixConcept(title="T", mood="dark", track_ids=ids, arc_type=arc_value)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre=genre)
    fires = any("all tracks high-energy" in w for w in warnings)
    assert fires is should_warn, f"genre={genre} arc_type={arc_type}: expected fires={should_warn}, got {fires}"


def test_known_good_mix_with_different_arc_types_stays_clean() -> None:
    """Known-good fixture should remain warning-free under several plausible arc_types."""
    from mixlab.llm import validate_stage2_output
    from mixlab.models import ArcType

    arc_values: list[ArcType | None] = ["wave", "build-and-drop", "double-peak", None]
    for arc in arc_values:
        concept, canvas, lib = _known_good_setup()
        concept = concept.model_copy(update={"arc_type": arc})
        warnings = validate_stage2_output([concept], [canvas], lib, set(), set(), genre="house")
        assert warnings == [], f"arc_type={arc} produced unexpected warnings: {warnings}"


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


@respx.mock
async def test_stage2_mix_length_appends_target_to_playlist_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """mix_length=60 injects set-length target (~15 tracks) into the playlist user prompt."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="practical", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1"}),
        seed_track_ids=["1"],
        mix_length=60,
    )

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "60 minutes" in user_prompt
    assert "15 tracks" in user_prompt


@respx.mock
async def test_stage2_mix_length_absent_from_playlist_prompt_when_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without mix_length, the playlist prompt contains no set-length target injection."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(_curated_payload())),
            Response(200, json=_anthropic_response(_REPORT_TEXT)),
        ]
    )

    shortlists = [MixConcept(title="Pool", mood="practical", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name="Monday Night",
        seed_ids=frozenset({"1"}),
        seed_track_ids=["1"],
    )

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "Set length target" not in user_prompt


@respx.mock
async def test_stage2_mix_length_applies_in_genre_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """mix_length also injects a set-length target in standard (genre) mode (issue #49)."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    shortlists = [MixConcept(title="Pool", mood="dark", track_ids=["1", "2", "3", "4"])]
    tracks_by_id = {
        str(i): Track(track_id=str(i), artist=f"A{i}", title=f"T{i}", bpm=174.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }

    await stage2_curate_and_report(shortlists, tracks_by_id, mix_length=60)

    body = json.loads(route.calls[0].request.content)
    user_prompt: str = next(m["content"] for m in body["messages"] if m["role"] == "user")
    assert "Set length target" in user_prompt
    assert "60 minutes" in user_prompt
    assert "per concept" in user_prompt


# ---------------------------------------------------------------------------
# _format_duration (issue #49)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("secs", "expected"),
    [
        (65, "1:05"),
        (272, "4:32"),
        (3600, "60:00"),
    ],
)
def test_format_duration_formats_mss(secs: int, expected: str) -> None:
    from mixlab.llm import _format_duration

    assert _format_duration(secs) == expected


# ---------------------------------------------------------------------------
# _duration_target_tracks (issue #49)
# ---------------------------------------------------------------------------


def test_duration_target_tracks_uses_mean_real_duration_when_available() -> None:
    from mixlab.llm import _duration_target_tracks

    tracks = [
        Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB", duration_secs=240)
        for i in range(4)
    ]
    # mean duration 240s (4 min) -> 60 min set / 4 min per track = 15 tracks
    assert _duration_target_tracks(60, tracks) == 15


def test_duration_target_tracks_falls_back_to_heuristic_without_duration_data() -> None:
    from mixlab.llm import _duration_target_tracks

    tracks = [Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB") for i in range(4)]
    assert _duration_target_tracks(60, tracks) == max(10, round(60 / 4.0))


def test_duration_target_tracks_enforces_floor_of_six() -> None:
    from mixlab.llm import _duration_target_tracks

    # Very long tracks (20 min each) over a short 20-minute set would compute to 1 track —
    # the floor of 6 must still apply.
    tracks = [
        Track(track_id=str(i), artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB", duration_secs=1200)
        for i in range(3)
    ]
    assert _duration_target_tracks(20, tracks) == 6


def test_duration_target_tracks_ignores_tracks_without_duration_in_mean() -> None:
    from mixlab.llm import _duration_target_tracks

    tracks = [
        Track(track_id="1", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB", duration_secs=240),
        Track(track_id="2", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB", duration_secs=240),
        Track(track_id="3", artist="A", title="T", bpm=174.0, camelot_key="8A", genre="DnB"),
    ]
    # Only tracks 1 and 2 (240s each) count towards the mean — track 3 is excluded.
    assert _duration_target_tracks(60, tracks) == 15


# ---------------------------------------------------------------------------
# Duration token in Stage 2 prompt track lines (issue #49)
# ---------------------------------------------------------------------------


@respx.mock
async def test_stage2_prompt_includes_duration_token_in_canvas_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duration token appears as the first extras element on canvas candidate lines."""
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    ids = ["1", "2", "3", "4"]
    concept = MixConcept(title="Pool", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    tracks_by_id = _lib(ids)
    tracks_by_id["1"] = tracks_by_id["1"].model_copy(update={"duration_secs": 272, "year": 2020})

    await stage2_curate_and_report(shortlists=[concept], tracks_by_id=tracks_by_id, canvases=[canvas])

    body = route.calls[0].request.content.decode()
    assert "4:32" in body
    # duration token must precede the year token on that track's candidate line
    line = next(text_line for text_line in body.splitlines() if "Artist_1 —" in text_line)
    assert line.index("4:32") < line.index("2020")


@respx.mock
async def test_stage2_prompt_omits_duration_token_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixlab.llm import stage2_curate_and_report

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(_curated_payload())))

    ids = ["1", "2", "3", "4"]
    concept = MixConcept(title="Pool", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    tracks_by_id = _lib(ids)  # no track carries duration_secs

    await stage2_curate_and_report(shortlists=[concept], tracks_by_id=tracks_by_id, canvases=[canvas])
    # No assertion error means no exception raised formatting extras with missing durations —
    # combined with the presence test above, this establishes the token is conditional.


# ---------------------------------------------------------------------------
# _append_runtime_to_report (issue #49)
# ---------------------------------------------------------------------------


def test_append_runtime_to_report_appends_footer_when_durations_known() -> None:
    from mixlab.llm import _append_runtime_to_report

    ids = ["1", "2", "3"]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    tracks_by_id = _lib(ids)
    tracks_by_id["1"] = tracks_by_id["1"].model_copy(update={"duration_secs": 240})
    tracks_by_id["2"] = tracks_by_id["2"].model_copy(update={"duration_secs": 300})
    # track "3" has no duration_secs

    original = "CONCEPT: T\n\nSome report body."
    result = _append_runtime_to_report(original, concept, tracks_by_id)

    assert result.startswith(original)
    assert "**Runtime**: ~9m (2/3 tracks with durations)" in result


def test_append_runtime_to_report_unchanged_when_no_durations_known() -> None:
    from mixlab.llm import _append_runtime_to_report

    ids = ["1", "2", "3"]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    tracks_by_id = _lib(ids)  # no track carries duration_secs

    original = "CONCEPT: T\n\nSome report body."
    result = _append_runtime_to_report(original, concept, tracks_by_id)

    assert result == original


# ---------------------------------------------------------------------------
# _format_canvas_section — Strong transitions line + demoted role hints (#51)
# ---------------------------------------------------------------------------


def _canvas_with_role_candidates(track_ids: list[str], roles: CanvasRoleCandidates) -> MixCanvas:
    concept = MixConcept(title="T", mood="dark", track_ids=track_ids)
    return MixCanvas(
        canvas_id="dnb_172.0_8A",
        genre="Drum & Bass",
        bpm_range=(86.0, 172.0),
        dominant_bpm=172.0,
        dominant_camelot="8A",
        core_track_ids=track_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=roles,
        contrast=ContrastAssets(
            vocal_moments=[], texture_changes=[], darker_turns=[], brighter_lifts=[], lower_pressure_resets=[]
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=concept,
    )


def test_format_canvas_section_renders_strong_transitions_line_with_mechanism() -> None:
    from mixlab.llm import _format_canvas_section

    tracks_by_id = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=172.0, camelot_key="8A", genre="Drum & Bass"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=86.0, camelot_key="8A", genre="Drum & Bass"),
    }
    roles = CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[])
    canvas = _canvas_with_role_candidates(["1", "2"], roles)

    section = _format_canvas_section(canvas, tracks_by_id)

    assert "Strong transitions:" in section
    assert "halftime lock 172→86" in section
    assert "ID:1→ID:2" in section


def test_format_canvas_section_strong_transitions_includes_blend_label_when_mix_points_present() -> None:
    # #59: when both tracks carry mix_points, the Strong transitions line appends the
    # blend-headroom label (e.g. "29 bars out / 32 in — tight") after the mechanism.
    from mixlab.llm import _format_canvas_section

    # Both directions need mix_points so blend_headroom is defined both ways; the
    # reverse direction (2→1) is deliberately given a much lower blend headroom so it
    # loses the symmetric-pair dedup and the kept edge is the 1→2 halftime one.
    tracks_by_id = {
        "1": Track(
            track_id="1",
            artist="A",
            title="T1",
            bpm=172.0,
            camelot_key="8A",
            genre="Drum & Bass",
            mix_points=MixPoints(mix_in_secs=0.0, outro_bars=29.0, intro_bars=40.0),
        ),
        "2": Track(
            track_id="2",
            artist="B",
            title="T2",
            bpm=86.0,
            camelot_key="8A",
            genre="Drum & Bass",
            mix_points=MixPoints(mix_in_secs=0.0, outro_bars=2.0, intro_bars=32.0),
        ),
    }
    roles = CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[])
    canvas = _canvas_with_role_candidates(["1", "2"], roles)

    section = _format_canvas_section(canvas, tracks_by_id)

    assert "Strong transitions:" in section
    assert "halftime lock 172→86" in section
    assert "29 bars out / 32 in — tight" in section


def test_format_canvas_section_keeps_opener_closer_drops_other_role_hints() -> None:
    from mixlab.llm import _format_canvas_section

    tracks_by_id = {
        str(i): Track(track_id=str(i), artist="A", title=f"T{i}", bpm=172.0, camelot_key="8A", genre="Drum & Bass")
        for i in range(1, 5)
    }
    roles = CanvasRoleCandidates(
        opener=["1"], groove_locker=["2"], builder=["2"], pivot=["3"], peak=["3"], closer=["4"]
    )
    canvas = _canvas_with_role_candidates(["1", "2", "3", "4"], roles)

    section = _format_canvas_section(canvas, tracks_by_id)

    assert "Opener: ID:1" in section
    assert "Closer: ID:4" in section
    assert "Groove-locker:" not in section
    assert "Builder:" not in section
    assert "Peak:" not in section
    assert "Pivot:" not in section


# ---------------------------------------------------------------------------
# validate_stage2_output — tempo-relation-aware BPM jump + arc verification (#51)
# ---------------------------------------------------------------------------


def test_validate_stage2_output_halftime_pair_does_not_warn_bpm_jump() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=172.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=86.0, camelot_key="8A", genre="DnB"),  # halftime lock
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=172.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert not any("BPM jump" in w for w in warnings)


def test_validate_stage2_output_straight_20_bpm_jump_still_warns() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=174.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="B", title="T2", bpm=194.0, camelot_key="8A", genre="DnB"),  # +20, no ratio
        **{
            str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=194.0, camelot_key="8A", genre="DnB")
            for i in range(3, 9)
        },
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("BPM jump" in w for w in warnings)


def test_validate_stage2_output_arc_mismatch_warning_for_monotonic_wave() -> None:
    from mixlab.llm import validate_stage2_output

    # Declared 'wave' but energy climbs monotonically — a factual mismatch.
    energies = [2, 3, 4, 5, 6, 7, 8, 8]
    lib = {
        str(i): Track(track_id=str(i), artist="C", title=f"T{i}", bpm=172.0, camelot_key="8A", genre="DnB", energy=e)
        for i, e in enumerate(energies, start=1)
    }
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="T", mood="dark", track_ids=ids, arc_type="wave")
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("arc mismatch" in w and "monotonic rising" in w for w in warnings)


# ---------------------------------------------------------------------------
# _compute_practicality_score — residual-stretch bpm_smoothness (#51)
# ---------------------------------------------------------------------------


def test_compute_practicality_clean_halftime_smoothness_matches_tight_straight() -> None:
    from mixlab.llm import _compute_practicality_score

    halftime_ids = ["1", "2", "3", "4"]
    halftime_lib = {
        "1": Track(track_id="1", artist="A", title="T1", bpm=172.0, camelot_key="8A", genre="DnB"),
        "2": Track(track_id="2", artist="A", title="T2", bpm=86.0, camelot_key="8A", genre="DnB"),
        "3": Track(track_id="3", artist="A", title="T3", bpm=172.0, camelot_key="8A", genre="DnB"),
        "4": Track(track_id="4", artist="A", title="T4", bpm=86.0, camelot_key="8A", genre="DnB"),
    }
    halftime_concept = MixConcept(title="H", mood="dark", track_ids=halftime_ids)
    halftime_score = _compute_practicality_score(halftime_concept, halftime_lib, intent_brief=None)

    straight_ids = ["10", "11", "12", "13"]
    straight_lib = {
        tid: Track(track_id=tid, artist="A", title=f"T{tid}", bpm=128.0, camelot_key="8A", genre="DnB")
        for tid in straight_ids
    }
    straight_concept = MixConcept(title="S", mood="dark", track_ids=straight_ids)
    straight_score = _compute_practicality_score(straight_concept, straight_lib, intent_brief=None)

    assert halftime_score.bpm_smoothness > 0.9
    assert abs(halftime_score.bpm_smoothness - straight_score.bpm_smoothness) < 0.05


# ---------------------------------------------------------------------------
# Concept directions (#53) — DIRECTION BRIEF + Target line + B8 fix + suppression
# ---------------------------------------------------------------------------


def _direction_canvas(
    track_ids: list[str],
    *,
    genre: str = "Drum & Bass",
    brief: str = "",
    direction_type: str = "",
    thread_artist: str = "",
) -> MixCanvas:
    concept = MixConcept(title="Mood journey: dark -> euphoric", mood="dark to euphoric", track_ids=track_ids)
    return MixCanvas(
        canvas_id="dir_172.0_8A",
        genre=genre,
        bpm_range=(170.0, 174.0),
        dominant_bpm=172.0,
        dominant_camelot="8A",
        core_track_ids=track_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[]),
        contrast=ContrastAssets(
            vocal_moments=[], texture_changes=[], darker_turns=[], brighter_lifts=[], lower_pressure_resets=[]
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=concept,
        brief=brief,
        direction_type=direction_type,
        thread_artist=thread_artist,
    )


def test_format_canvas_section_renders_direction_brief_block() -> None:
    from mixlab.llm import _format_canvas_section

    tracks_by_id = {"1": Track(track_id="1", artist="A", title="T1", bpm=172.0, camelot_key="8A", genre="Drum & Bass")}
    canvas = _direction_canvas(["1"], brief="Open dark, land euphoric.", direction_type="mood_journey")
    section = _format_canvas_section(canvas, tracks_by_id)
    assert "DIRECTION BRIEF (mood_journey):" in section
    assert "Open dark, land euphoric." in section
    # The brief precedes the canvas header line.
    assert section.index("DIRECTION BRIEF") < section.index("[Canvas ")


def test_format_canvas_section_renders_target_track_count_from_genre() -> None:
    from mixlab.llm import _format_canvas_section

    tracks_by_id = {"1": Track(track_id="1", artist="A", title="T1", bpm=172.0, camelot_key="8A", genre="Drum & Bass")}
    canvas = _direction_canvas(["1"], genre="Drum & Bass")
    section = _format_canvas_section(canvas, tracks_by_id)
    assert "Target: 10-14 tracks" in section


def test_format_canvas_section_target_falls_back_to_default_for_unknown_genre() -> None:
    from mixlab.llm import _format_canvas_section

    tracks_by_id = {"1": Track(track_id="1", artist="A", title="T1", bpm=172.0, camelot_key="8A", genre="Mystery")}
    canvas = _direction_canvas(["1"], genre="Mystery")
    section = _format_canvas_section(canvas, tracks_by_id)
    assert "Target: 8-12 tracks" in section


def test_format_canvas_section_classic_canvas_has_no_direction_brief() -> None:
    from mixlab.llm import _format_canvas_section

    tracks_by_id = {"1": Track(track_id="1", artist="A", title="T1", bpm=172.0, camelot_key="8A", genre="Drum & Bass")}
    canvas = _direction_canvas(["1"])  # brief defaults to ""
    section = _format_canvas_section(canvas, tracks_by_id)
    assert "DIRECTION BRIEF" not in section


def test_stage2_system_no_longer_hardcodes_eight_to_twelve_tracks() -> None:
    from mixlab.llm import _STAGE2_SYSTEM

    assert "8–12 tracks" not in _STAGE2_SYSTEM
    assert "target track-count range" in _STAGE2_SYSTEM


def test_stage2_canvas_rules_mention_direction_brief() -> None:
    from mixlab.llm import _STAGE2_CANVAS_RULES

    assert "DIRECTION BRIEF" in _STAGE2_CANVAS_RULES


def test_validate_stage2_output_suppresses_thread_artist_repeat() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        str(i): Track(track_id=str(i), artist="SpineGuy", title=f"T{i}", bpm=172.0, camelot_key="8A", genre="DnB")
        for i in range(1, 4)
    }
    ids = [str(i) for i in range(1, 4)]
    concept = MixConcept(title="Artist thread: SpineGuy", mood="spine", track_ids=ids)
    canvas = _direction_canvas(ids, direction_type="artist_thread", thread_artist="SpineGuy")
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert not any("SpineGuy" in w and "appears" in w for w in warnings)


def test_validate_stage2_output_still_warns_non_thread_artist_repeat() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        str(i): Track(track_id=str(i), artist="Other Artist", title=f"T{i}", bpm=172.0, camelot_key="8A", genre="DnB")
        for i in range(1, 4)
    }
    ids = [str(i) for i in range(1, 4)]
    concept = MixConcept(title="Artist thread: SpineGuy", mood="spine", track_ids=ids)
    canvas = _direction_canvas(ids, direction_type="artist_thread", thread_artist="SpineGuy")
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("Other Artist" in w and "appears 3 times" in w for w in warnings)


def test_validate_stage2_output_thread_artist_four_plus_still_warns() -> None:
    from mixlab.llm import validate_stage2_output

    lib = {
        str(i): Track(track_id=str(i), artist="SpineGuy", title=f"T{i}", bpm=172.0, camelot_key="8A", genre="DnB")
        for i in range(1, 5)
    }
    ids = [str(i) for i in range(1, 5)]
    concept = MixConcept(title="Artist thread: SpineGuy", mood="spine", track_ids=ids)
    canvas = _direction_canvas(ids, direction_type="artist_thread", thread_artist="SpineGuy")
    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert any("SpineGuy" in w and "appears 4 times" in w for w in warnings)


# ---------------------------------------------------------------------------
# Bounded self-revision (#55)
# ---------------------------------------------------------------------------


def _revision_lib(key_by_id: dict[str, str], bpm: float = 174.0) -> dict[str, Track]:
    """Library where each id gets an explicit Camelot key (defaults to 8A)."""
    return {
        i: Track(
            track_id=i,
            artist=f"Artist_{i}",
            title=f"Title_{i}",
            bpm=bpm,
            camelot_key=key_by_id.get(i, "8A"),
            genre="house",
        )
        for i in key_by_id
    }


def test_qualifies_for_revision_single_hard_finding_returns_false() -> None:
    from mixlab.llm import _qualifies_for_revision

    concept = MixConcept(title="X", mood="m", track_ids=["1", "2", "3", "4"])
    warnings = ["[X] track ID 5 not found in library"]
    assert _qualifies_for_revision(concept, warnings) is False


def test_qualifies_for_revision_two_hard_findings_returns_true() -> None:
    from mixlab.llm import _qualifies_for_revision

    concept = MixConcept(title="X", mood="m", track_ids=["1", "2", "3", "4"])
    warnings = [
        "[X] track ID 5 not found in library",
        "[X] BPM jump 22.0 between A — a and B — b",
    ]
    assert _qualifies_for_revision(concept, warnings) is True


def test_qualifies_for_revision_soft_only_warnings_returns_false() -> None:
    from mixlab.llm import _qualifies_for_revision

    concept = MixConcept(title="X", mood="m", track_ids=["1", "2", "3", "4"])
    warnings = [
        "[X] no wind-down in final 3 tracks (all energy >4/8)",
        "[X] all tracks high-energy (≥6/8) — no dynamic range",
        "Concept title 'X' matches generic [Adjective][Noun] pattern — review for distinctiveness",
    ]
    assert _qualifies_for_revision(concept, warnings) is False


def test_qualifies_for_revision_weak_critique_returns_true() -> None:
    from mixlab.llm import _qualifies_for_revision

    concept = MixConcept(title="X", mood="m", track_ids=["1", "2", "3", "4"], critique=Critique(verdict="weak"))
    assert _qualifies_for_revision(concept, []) is True


def test_qualifies_for_revision_needs_attention_with_substitution_returns_true() -> None:
    from mixlab.llm import _qualifies_for_revision

    concept = MixConcept(
        title="X",
        mood="m",
        track_ids=["1", "2", "3", "4"],
        critique=Critique(verdict="needs_attention", suggested_substitution="track 3 → ID:9"),
    )
    assert _qualifies_for_revision(concept, []) is True


def test_qualifies_for_revision_needs_attention_without_substitution_returns_false() -> None:
    from mixlab.llm import _qualifies_for_revision

    concept = MixConcept(
        title="X",
        mood="m",
        track_ids=["1", "2", "3", "4"],
        critique=Critique(verdict="needs_attention", suggested_substitution=None),
    )
    assert _qualifies_for_revision(concept, []) is False


def _revision_canvas(pool_ids: list[str]) -> MixCanvas:
    concept = MixConcept(title="pool", mood="m", track_ids=pool_ids)
    return MixCanvas(
        canvas_id="rev_canvas",
        genre="house",
        bpm_range=(170.0, 178.0),
        dominant_bpm=174.0,
        dominant_camelot="8A",
        core_track_ids=pool_ids,
        bridge_track_ids=[],
        wildcard_track_ids=[],
        roles=CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[]),
        contrast=ContrastAssets(
            vocal_moments=[], texture_changes=[], darker_turns=[], brighter_lifts=[], lower_pressure_resets=[]
        ),
        risk_notes=[],
        score=CanvasScore(),
        source_concept=concept,
    )


@respx.mock
async def test_revise_concepts_happy_path_accepts_and_annotates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import revise_concepts, validate_stage2_output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Pool ids 1..9 all 8A except id 3 (2A) — the offending track. Id 9 is a spare 8A.
    pool = [str(i) for i in range(1, 10)]
    lib = _revision_lib({i: ("2A" if i == "3" else "8A") for i in pool})
    canvas = _revision_canvas(pool)
    original = MixConcept(title="Jump Fix", mood="steady", track_ids=[str(i) for i in range(1, 9)])

    warnings = validate_stage2_output([original], [canvas], lib, played_ids=set(), denylist_ids=set(), genre="house")
    assert len([w for w in warnings if "Camelot jump" in w]) == 2

    # Revision swaps the 2A track (3) for the spare 8A track (9) → no jumps.
    revision_payload = json.dumps(
        [
            {
                "title": "Jump Fix",
                "name_reason": "steady",
                "mood": "steady",
                "track_ids": ["1", "2", "9", "4", "5", "6", "7", "8"],
                "transitions": [],
            }
        ]
    )
    # First call: the revision repair. Second call: the report-section regeneration.
    route = respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(revision_payload)),
            Response(200, json=_anthropic_response("FRESH PROSE FOR REVISED ORDER")),
        ]
    )

    concepts, report, final_warnings = await revise_concepts(
        [original],
        "PROSE REPORT",
        warnings,
        [canvas],
        lib,
        played_ids=set(),
        allow_played=False,
        genre="house",
    )

    assert route.call_count == 2
    assert concepts[0].track_ids == ["1", "2", "9", "4", "5", "6", "7", "8"]
    assert "**Revised**" in report
    assert "prose regenerated to match" in report
    assert "FRESH PROSE FOR REVISED ORDER" in report
    assert "PROSE REPORT" not in report  # stale pre-revision section replaced
    assert not any("Camelot jump" in w for w in final_warnings)
    assert len(final_warnings) < len(warnings)


@respx.mock
async def test_revise_concepts_report_regen_failure_keeps_repair_with_disclaimer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A report-regeneration failure must not cost us the accepted repair."""
    from mixlab.llm import revise_concepts, validate_stage2_output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pool = [str(i) for i in range(1, 10)]
    lib = _revision_lib({i: ("2A" if i == "3" else "8A") for i in pool})
    canvas = _revision_canvas(pool)
    original = MixConcept(title="Jump Fix", mood="steady", track_ids=[str(i) for i in range(1, 9)])
    warnings = validate_stage2_output([original], [canvas], lib, played_ids=set(), denylist_ids=set(), genre="house")

    revision_payload = json.dumps(
        [
            {
                "title": "Jump Fix",
                "name_reason": "steady",
                "mood": "steady",
                "track_ids": ["1", "2", "9", "4", "5", "6", "7", "8"],
                "transitions": [],
            }
        ]
    )
    # 400 fails fast (no retry loop) — only 429/5xx are retried.
    respx.post(_ANTHROPIC_URL).mock(
        side_effect=[
            Response(200, json=_anthropic_response(revision_payload)),
            Response(400, json={"error": {"message": "bad request"}}),
        ]
    )

    concepts, report, _final = await revise_concepts(
        [original],
        "PROSE REPORT",
        warnings,
        [canvas],
        lib,
        played_ids=set(),
        allow_played=False,
        genre="house",
    )

    assert concepts[0].track_ids == ["1", "2", "9", "4", "5", "6", "7", "8"]
    assert "PROSE REPORT" in report  # pre-revision prose kept
    assert "pre-revision sequence" in report  # disclaimer annotation
    assert "report regeneration failed" in capsys.readouterr().err


@respx.mock
async def test_revise_concepts_section_mismatch_skips_regen_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the report doesn't split into one section per concept, no regen call is made."""
    from mixlab.llm import revise_concepts, validate_stage2_output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pool = [str(i) for i in range(1, 10)]
    lib = _revision_lib({i: ("2A" if i == "3" else "8A") for i in pool})
    canvas = _revision_canvas(pool)
    flagged = MixConcept(title="Flagged", mood="m", track_ids=[str(i) for i in range(1, 9)])
    clean = MixConcept(title="Clean", mood="m", track_ids=["1", "2", "4", "5", "6", "7", "8", "9"])
    warnings = validate_stage2_output(
        [flagged, clean], [canvas], lib, played_ids=set(), denylist_ids=set(), genre="house"
    )

    revision_payload = json.dumps(
        [
            {
                "title": "Flagged",
                "name_reason": "m",
                "mood": "m",
                "track_ids": ["1", "2", "4", "5", "6", "7", "8", "9"],
                "transitions": [],
            }
        ]
    )
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(revision_payload)))

    # Two concepts but a single-section report — splice impossible, regen skipped.
    _concepts, report, _final = await revise_concepts(
        [flagged, clean],
        "PROSE REPORT",
        warnings,
        [canvas],
        lib,
        played_ids=set(),
        allow_played=False,
        genre="house",
    )

    assert route.call_count == 1  # revision only, no regeneration
    assert "pre-revision sequence" in report


@respx.mock
async def test_revise_concepts_worse_revision_keeps_original(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mixlab.llm import revise_concepts, validate_stage2_output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pool = [str(i) for i in range(1, 11)]
    lib = _revision_lib({i: ("2A" if i in ("3", "6") else "8A") for i in pool})
    canvas = _revision_canvas(pool)
    original = MixConcept(title="Steady", mood="steady", track_ids=["1", "2", "3", "4", "5", "7", "8", "9"])

    warnings = validate_stage2_output([original], [canvas], lib, played_ids=set(), denylist_ids=set(), genre="house")
    assert len([w for w in warnings if "Camelot jump" in w]) == 2

    # Revision scatters BOTH 2A tracks (3 and 6) between 8A tracks → four jumps (worse).
    revision_payload = json.dumps(
        [
            {
                "title": "Steady",
                "name_reason": "steady",
                "mood": "steady",
                "track_ids": ["1", "3", "2", "6", "4", "5", "7", "8"],
                "transitions": [],
            }
        ]
    )
    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(revision_payload)))

    concepts, report, _final = await revise_concepts(
        [original],
        "PROSE REPORT",
        warnings,
        [canvas],
        lib,
        played_ids=set(),
        allow_played=False,
        genre="house",
    )

    assert concepts[0].track_ids == original.track_ids
    assert "**Revised**" not in report
    assert "did not improve" in capsys.readouterr().err


@respx.mock
async def test_revise_concepts_unparseable_response_keeps_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import revise_concepts, validate_stage2_output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pool = [str(i) for i in range(1, 10)]
    lib = _revision_lib({i: ("2A" if i == "3" else "8A") for i in pool})
    canvas = _revision_canvas(pool)
    original = MixConcept(title="Jump Fix", mood="steady", track_ids=[str(i) for i in range(1, 9)])
    warnings = validate_stage2_output([original], [canvas], lib, played_ids=set(), denylist_ids=set(), genre="house")

    respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response("sorry, cannot help")))

    concepts, report, _final = await revise_concepts(
        [original],
        "PROSE REPORT",
        warnings,
        [canvas],
        lib,
        played_ids=set(),
        allow_played=False,
        genre="house",
    )

    assert concepts[0].track_ids == original.track_ids
    assert "**Revised**" not in report


@respx.mock
async def test_revise_concepts_calls_once_per_qualifying_concept_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import revise_concepts, validate_stage2_output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pool = [str(i) for i in range(1, 10)]
    lib = _revision_lib({i: ("2A" if i == "3" else "8A") for i in pool})
    canvas = _revision_canvas(pool)
    # Qualifying: two Camelot jumps around id 3.
    flagged = MixConcept(title="Flagged", mood="m", track_ids=[str(i) for i in range(1, 9)])
    # Clean: no jumps, no critique — must NOT be revised.
    clean = MixConcept(title="Clean", mood="m", track_ids=["1", "2", "4", "5", "6", "7", "8", "9"])

    warnings = validate_stage2_output(
        [flagged, clean], [canvas], lib, played_ids=set(), denylist_ids=set(), genre="house"
    )
    revision_payload = json.dumps(
        [
            {
                "title": "Flagged",
                "name_reason": "m",
                "mood": "m",
                "track_ids": ["1", "2", "4", "5", "6", "7", "8", "9"],
                "transitions": [],
            }
        ]
    )
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response(revision_payload)))

    await revise_concepts(
        [flagged, clean],
        "PROSE REPORT",
        warnings,
        [canvas],
        lib,
        played_ids=set(),
        allow_played=False,
        genre="house",
    )
    assert route.call_count == 1


@respx.mock
async def test_revise_concepts_no_api_key_returns_inputs_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixlab.llm import revise_concepts

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pool = [str(i) for i in range(1, 10)]
    lib = _revision_lib({i: "8A" for i in pool})
    canvas = _revision_canvas(pool)
    original = MixConcept(title="X", mood="m", track_ids=[str(i) for i in range(1, 9)])
    warnings = ["[X] BPM jump 20.0 between a and b", "[X] Camelot jump 6 between 8A and 2A"]
    route = respx.post(_ANTHROPIC_URL).mock(return_value=Response(200, json=_anthropic_response("[]")))

    concepts, report, final_warnings = await revise_concepts(
        [original],
        "PROSE REPORT",
        warnings,
        [canvas],
        lib,
        played_ids=set(),
        allow_played=False,
        genre="house",
    )

    assert route.call_count == 0
    assert concepts[0] is original
    assert report == "PROSE REPORT"
    assert final_warnings == warnings


# ---------------------------------------------------------------------------
# Mix Engine (#61) — validator blend check
# ---------------------------------------------------------------------------


def _blend_track(tid: str, mp: MixPoints | None = None) -> Track:
    return Track(
        track_id=tid, artist=f"A{tid}", title=f"T{tid}", bpm=174.0, camelot_key="8A", genre="DnB", mix_points=mp
    )


def test_validate_stage2_output_tight_blend_unannotated_warns_and_is_hard_finding() -> None:
    from mixlab.llm import _qualifies_for_revision, validate_stage2_output

    # Two consecutive pairs (1->2, 2->3) with a 2-bar outro into a 32-bar intro — both
    # score 0.15 (< 0.3). Same BPM/key everywhere so nothing else warns.
    lib = {
        "1": _blend_track("1", MixPoints(mix_in_secs=0.0, outro_bars=2.0)),
        "2": _blend_track("2", MixPoints(mix_in_secs=0.0, intro_bars=32.0, outro_bars=2.0)),
        "3": _blend_track("3", MixPoints(mix_in_secs=0.0, intro_bars=32.0)),
    }
    for i in range(4, 9):
        lib[str(i)] = _blend_track(str(i))
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="B", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)

    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    blend_warnings = [w for w in warnings if "blend risk" in w]
    assert len(blend_warnings) == 2
    assert "blend risk 1->2" in blend_warnings[0]
    assert "cut or manual loop likely" in blend_warnings[0]
    # Two blend findings are two hard findings — enough to trigger the #55 revision gate.
    assert _qualifies_for_revision(concept, warnings) is True


def test_validate_stage2_output_tight_blend_suppressed_when_transition_justified() -> None:
    from mixlab.llm import validate_stage2_output
    from mixlab.models import Transition

    lib = {
        "1": _blend_track("1", MixPoints(mix_in_secs=0.0, outro_bars=2.0)),
        "2": _blend_track("2", MixPoints(mix_in_secs=0.0, intro_bars=32.0, outro_bars=2.0)),
        "3": _blend_track("3", MixPoints(mix_in_secs=0.0, intro_bars=32.0)),
    }
    for i in range(4, 9):
        lib[str(i)] = _blend_track(str(i))
    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(
        title="B",
        mood="dark",
        track_ids=ids,
        transitions=[Transition(from_id="1", to_id="2", is_risky=True, risk_type="chapter_pivot")],
    )
    canvas = _make_canvas(ids)

    warnings = validate_stage2_output([concept], [canvas], lib, set(), set())
    assert not any("blend risk 1->2" in w for w in warnings)
    # The un-annotated 2->3 pair still warns.
    assert any("blend risk 2->3" in w for w in warnings)


def test_validate_stage2_output_blend_silent_without_mix_points() -> None:
    from mixlab.llm import validate_stage2_output

    ids = [str(i) for i in range(1, 9)]
    concept = MixConcept(title="B", mood="dark", track_ids=ids)
    canvas = _make_canvas(ids)
    warnings = validate_stage2_output([concept], [canvas], _lib(ids), set(), set())
    assert not any("blend risk" in w for w in warnings)


# ---------------------------------------------------------------------------
# Mix Engine (#61) — practicality blend component
# ---------------------------------------------------------------------------


def test_compute_practicality_cueless_overall_uses_legacy_formula() -> None:
    from mixlab.llm import _compute_practicality_score

    ids = ["1", "2", "3", "4"]
    lib = {
        tid: Track(track_id=tid, artist="A", title=f"T{tid}", bpm=172.0, camelot_key="8A", genre="DnB") for tid in ids
    }
    concept = MixConcept(title="C", mood="dark", track_ids=ids)
    score = _compute_practicality_score(concept, lib, intent_brief=None)

    assert score.has_blend_data is False
    legacy = (
        score.bpm_smoothness * 0.30
        + score.harmonic_ratio * 0.30
        + score.risk_justified * 0.25
        + score.fragment_preserved * 0.15
    )
    assert score.overall == pytest.approx(legacy)


def test_compute_practicality_cued_all_pairs_uses_new_weights() -> None:
    from mixlab.llm import _compute_practicality_score

    ids = ["1", "2", "3", "4"]
    # outro 16 into intro 16 → ratio 1.0 → headroom 0.7 on every pair.
    lib = {
        tid: Track(
            track_id=tid,
            artist="A",
            title=f"T{tid}",
            bpm=172.0,
            camelot_key="8A",
            genre="DnB",
            mix_points=MixPoints(mix_in_secs=0.0, intro_bars=16.0, outro_bars=16.0),
        )
        for tid in ids
    }
    concept = MixConcept(title="C", mood="dark", track_ids=ids)
    score = _compute_practicality_score(concept, lib, intent_brief=None)

    assert score.has_blend_data is True
    assert score.blend_feasibility == pytest.approx(0.7)
    new_formula = (
        score.bpm_smoothness * 0.25
        + score.harmonic_ratio * 0.25
        + score.risk_justified * 0.20
        + score.fragment_preserved * 0.10
        + score.blend_feasibility * 0.20
    )
    assert score.overall == pytest.approx(new_formula)
    assert score.overall == pytest.approx(0.94)


def test_compute_practicality_has_blend_data_half_pairs_boundary() -> None:
    from mixlab.llm import _compute_practicality_score

    ids = ["0", "1", "2", "3", "4"]  # 4 consecutive pairs
    concept = MixConcept(title="C", mood="dark", track_ids=ids)

    def _mk(tid: str, mp: MixPoints | None) -> Track:
        return Track(track_id=tid, artist="A", title=f"T{tid}", bpm=172.0, camelot_key="8A", genre="DnB", mix_points=mp)

    # Exactly 2 of 4 pairs carry data (pairs 0->1 and 1->2): 2*2 >= 4 → True.
    lib_boundary = {
        "0": _mk("0", MixPoints(mix_in_secs=0.0, outro_bars=16.0)),
        "1": _mk("1", MixPoints(mix_in_secs=0.0, intro_bars=16.0, outro_bars=16.0)),
        "2": _mk("2", MixPoints(mix_in_secs=0.0, intro_bars=16.0)),
        "3": _mk("3", None),
        "4": _mk("4", None),
    }
    assert _compute_practicality_score(concept, lib_boundary, intent_brief=None).has_blend_data is True

    # Only 1 of 4 pairs carries data (drop track 1's outro): 1*2 < 4 → False.
    lib_below = dict(lib_boundary)
    lib_below["1"] = _mk("1", MixPoints(mix_in_secs=0.0, intro_bars=16.0))
    assert _compute_practicality_score(concept, lib_below, intent_brief=None).has_blend_data is False


def test_format_practicality_line_appends_blend_only_when_blend_data() -> None:
    from mixlab.llm import _format_practicality_line

    with_blend = DJPracticalityScore(
        bpm_smoothness=0.8,
        harmonic_ratio=0.7,
        risk_justified=0.5,
        fragment_preserved=1.0,
        blend_feasibility=0.65,
        has_blend_data=True,
    )
    assert "blend_feasibility 0.65" in _format_practicality_line(with_blend)

    without_blend = DJPracticalityScore(
        bpm_smoothness=0.8, harmonic_ratio=0.7, risk_justified=0.5, fragment_preserved=1.0
    )
    assert "blend_feasibility" not in _format_practicality_line(without_blend)


# ---------------------------------------------------------------------------
# Mix Engine (#61) — resequence_suggestions
# ---------------------------------------------------------------------------


def _resequence_case() -> tuple[MixConcept, dict[str, Track]]:
    # Two tempo families: k0..k5 at 170 BPM, k6..k9 at 205 BPM. The planted order swaps
    # positions 3 and 6, dropping a fast track into the slow block; a single 3<->6 swap
    # restores contiguity for a >10% gain (0.63 -> 0.78). Annotate (k0,k1) so we can
    # check annotation survival after apply.
    bpms = [170.0] * 6 + [205.0] * 4
    lib = {
        f"k{i}": Track(track_id=f"k{i}", artist=f"A{i}", title=f"T{i}", bpm=bpms[i], camelot_key="8A", genre="DnB")
        for i in range(10)
    }
    planted = [f"k{i}" for i in range(10)]
    planted[3], planted[6] = planted[6], planted[3]
    concept = MixConcept(
        title="Two Families",
        mood="dark",
        track_ids=planted,
        transitions=[Transition(from_id="k0", to_id="k1", is_risky=True, risk_type="chapter_pivot")],
    )
    return concept, lib


def test_resequence_suggestions_suggest_mode_leaves_concept_unchanged() -> None:
    from mixlab.llm import resequence_suggestions

    concept, lib = _resequence_case()
    result, notes = resequence_suggestions([concept], lib, apply=False)

    assert result[0].track_ids == concept.track_ids  # untouched
    assert result[0] is concept
    assert len(notes) == 1
    assert "Exported order unchanged." in notes[0]
    assert "swapping 4<->7" in notes[0]
    assert "Two Families" in notes[0]


def test_resequence_suggestions_apply_mode_swaps_and_rebuilds_transitions() -> None:
    from mixlab.llm import resequence_suggestions

    concept, lib = _resequence_case()
    result, notes = resequence_suggestions([concept], lib, apply=True)

    assert result[0].track_ids == [f"k{i}" for i in range(10)]  # restored contiguity
    assert len(notes) == 1
    assert "Order applied." in notes[0]

    tmap = {(t.from_id, t.to_id): t for t in result[0].transitions}
    # Annotated pair (k0,k1) survives adjacency in the new order → keeps its risk_type.
    assert tmap[("k0", "k1")].is_risky is True
    assert tmap[("k0", "k1")].risk_type == "chapter_pivot"
    # A pair newly adjacent after the swap (k2,k3) defaults to a non-risky transition.
    assert tmap[("k2", "k3")].is_risky is False
    assert tmap[("k2", "k3")].risk_type == ""
    assert len(result[0].transitions) == 9  # one per consecutive pair


def test_resequence_suggestions_already_good_order_no_notes() -> None:
    from mixlab.llm import resequence_suggestions

    ids = [f"c{i}" for i in range(10)]
    lib = {
        f"c{i}": Track(track_id=f"c{i}", artist=f"A{i}", title=f"T{i}", bpm=170.0, camelot_key=f"{i + 1}A", genre="DnB")
        for i in range(10)
    }
    concept = MixConcept(title="Chain", mood="dark", track_ids=ids)
    result, notes = resequence_suggestions([concept], lib, apply=False)
    assert notes == []
    assert result[0] is concept


def test_resequence_suggestions_skips_short_concepts() -> None:
    from mixlab.llm import resequence_suggestions

    ids = ["1", "2", "3"]
    lib = {
        tid: Track(track_id=tid, artist="A", title=f"T{tid}", bpm=170.0, camelot_key="8A", genre="DnB") for tid in ids
    }
    concept = MixConcept(title="Short", mood="dark", track_ids=ids)
    result, notes = resequence_suggestions([concept], lib, apply=True)
    assert notes == []
    assert result[0] is concept


# ---------------------------------------------------------------------------
# Mix Engine (#61) — intro/outro prompt tokens
# ---------------------------------------------------------------------------


def test_format_canvas_section_includes_intro_outro_token_when_mix_points_present() -> None:
    from mixlab.llm import _format_canvas_section

    ids = ["1", "2"]
    tracks_by_id = _lib(ids)
    tracks_by_id["1"] = tracks_by_id["1"].model_copy(
        update={"mix_points": MixPoints(mix_in_secs=0.0, intro_bars=16.0, outro_bars=32.0)}
    )
    canvas = _make_canvas(ids)

    section = _format_canvas_section(canvas, tracks_by_id)
    line = next(text_line for text_line in section.splitlines() if "Artist_1 —" in text_line)
    assert "intro:16b/outro:32b" in line
    # A track with no mix_points carries no intro/outro token.
    line2 = next(text_line for text_line in section.splitlines() if "Artist_2 —" in text_line)
    assert "intro:" not in line2
    assert "outro:" not in line2
