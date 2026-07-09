from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import respx
from httpx import Response

from mixlab.history import ConceptHistory, ConceptRecord, HistoryEntry, load_history, save_history
from mixlab.remote import FeedbackEvent, MixLabRemote, RemoteConfig, RemoteConflictError, RemoteError
from mixlab.sync import SyncReport, _compose_notes, sync_down, sync_up

_BASE = "https://mixlab-api.example.com"
_SECRET = "test-secret"

_HISTORY_PATH = "/api/mixlab/history"
_FEEDBACK_PENDING_PATH = "/api/mixlab/feedback/pending"
_FEEDBACK_ACK_PATH = "/api/mixlab/feedback/ack"


def _remote() -> MixLabRemote:
    return MixLabRemote(RemoteConfig(base_url=_BASE, api_secret=_SECRET))


def _entry(run_id: str, genre: str = "house") -> HistoryEntry:
    return HistoryEntry(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        mode="standard",
        genre=genre,
        selected_canvas_ids=[],
        dominant_bpm_clusters=[],
        dominant_camelot_keys=[],
        core_track_ids=[],
        anchor_track_ids=[],
        opener_candidates=[],
        closer_candidates=[],
        concept_title="",
        concept_track_ids=[],
        energy_path="",
        mood="",
    )


def _serialize(entries: list[HistoryEntry]) -> str:
    return json.dumps({"runs": [asdict(e) for e in entries]}, indent=2)


def _etag_path(history_path: Path) -> Path:
    return Path(str(history_path) + ".etag")


def _feedback_event(
    event_id: str = "f_1",
    *,
    concept_id: str = "c1",
    verdict: str | None = "played",
    rating: int | None = None,
    notes: str | None = None,
    published_mix_slug: str | None = None,
) -> FeedbackEvent:
    return FeedbackEvent(
        event_id=event_id,
        run_id="r1",
        concept_id=concept_id,
        verdict=verdict,
        rating=rating,
        notes=notes,
        published_mix_slug=published_mix_slug,
        recorded_at="2026-07-08T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# SyncReport
# ---------------------------------------------------------------------------


def test_sync_report_str_single_line_summary() -> None:
    report = SyncReport(applied=2, skipped=1, acked=3)
    assert str(report) == "sync: applied 2 verdict(s), skipped 1, acked 3"


def test_sync_report_defaults_all_zero() -> None:
    report = SyncReport()
    assert (report.applied, report.skipped, report.acked) == (0, 0, 0)


# ---------------------------------------------------------------------------
# _compose_notes
# ---------------------------------------------------------------------------


def test_compose_notes_all_parts_present_joined_in_order() -> None:
    event = _feedback_event(rating=4, published_mix_slug="my-mix", notes="great opener")
    assert _compose_notes(event) == "rating=4; mix=my-mix; great opener"


def test_compose_notes_rating_only() -> None:
    event = _feedback_event(rating=5)
    assert _compose_notes(event) == "rating=5"


def test_compose_notes_notes_only() -> None:
    event = _feedback_event(notes="loved it")
    assert _compose_notes(event) == "loved it"


def test_compose_notes_nothing_present_returns_empty_string() -> None:
    event = _feedback_event()
    assert _compose_notes(event) == ""


def test_compose_notes_slug_and_notes_no_rating() -> None:
    event = _feedback_event(published_mix_slug="my-mix", notes="loved it")
    assert _compose_notes(event) == "mix=my-mix; loved it"


# ---------------------------------------------------------------------------
# sync_down
# ---------------------------------------------------------------------------


@respx.mock
def test_sync_down_remote_history_adopted_and_etag_cached(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    remote_body = _serialize([_entry("r_remote")])
    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200, text=remote_body, headers={"ETag": '"e1"'}))
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(return_value=Response(200, json=[]))

    report = sync_down(_remote(), history_path)

    assert history_path.read_text() == remote_body
    assert _etag_path(history_path).read_text() == '"e1"'
    assert report == SyncReport()


@respx.mock
def test_sync_down_remote_404_keeps_local_history_asis(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    local_entry = _entry("r_local")
    save_history(ConceptHistory(runs=[local_entry]), history_path)
    original_text = history_path.read_text()

    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(404))
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(return_value=Response(200, json=[]))

    report = sync_down(_remote(), history_path)

    assert history_path.read_text() == original_text
    assert not _etag_path(history_path).exists()
    assert report == SyncReport()


@respx.mock
def test_sync_down_matched_event_applied_through_shared_helper_sets_feedback(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    entry = _entry("r1")
    entry.concepts = [ConceptRecord(concept_id="c1", title="Midnight Run", mood="dark", track_ids=["T1"], arc_type="")]
    save_history(ConceptHistory(runs=[entry]), history_path)

    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(404))
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(
        return_value=Response(200, json=[{"eventId": "f_1", "runId": "r1", "conceptId": "c1", "verdict": "played"}])
    )
    respx.put(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200, headers={"ETag": '"e2"'}))
    ack_route = respx.post(f"{_BASE}{_FEEDBACK_ACK_PATH}").mock(return_value=Response(200))

    report = sync_down(_remote(), history_path)

    assert report == SyncReport(applied=1, skipped=0, acked=1)
    reloaded = load_history(history_path)
    assert reloaded.runs[0].concepts[0].feedback == "played"
    assert json.loads(ack_route.calls.last.request.content) == {"eventIds": ["f_1"]}


@respx.mock
def test_sync_down_unmatched_event_counted_skipped_and_acked(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    entry = _entry("r1")
    entry.concepts = [ConceptRecord(concept_id="c1", title="Midnight Run", mood="dark", track_ids=["T1"], arc_type="")]
    save_history(ConceptHistory(runs=[entry]), history_path)

    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(404))
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(
        return_value=Response(
            200, json=[{"eventId": "f_1", "runId": "r1", "conceptId": "no-such-id", "verdict": "played"}]
        )
    )
    put_route = respx.put(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200))
    ack_route = respx.post(f"{_BASE}{_FEEDBACK_ACK_PATH}").mock(return_value=Response(200))

    report = sync_down(_remote(), history_path)

    assert report == SyncReport(applied=0, skipped=1, acked=1)
    assert put_route.call_count == 0
    assert json.loads(ack_route.calls.last.request.content) == {"eventIds": ["f_1"]}


@respx.mock
def test_sync_down_put_failure_skips_ack_and_raises(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    entry = _entry("r1")
    entry.concepts = [ConceptRecord(concept_id="c1", title="Midnight Run", mood="dark", track_ids=["T1"], arc_type="")]
    save_history(ConceptHistory(runs=[entry]), history_path)

    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(404))
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(
        return_value=Response(200, json=[{"eventId": "f_1", "runId": "r1", "conceptId": "c1", "verdict": "played"}])
    )
    respx.put(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(500, text="down"))
    ack_route = respx.post(f"{_BASE}{_FEEDBACK_ACK_PATH}").mock(return_value=Response(200))

    with pytest.raises(RemoteError):
        sync_down(_remote(), history_path)

    assert ack_route.call_count == 0


@respx.mock
def test_sync_down_conflict_retries_once_then_applies_and_acks(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    entry = _entry("r1")
    entry.concepts = [ConceptRecord(concept_id="c1", title="Midnight Run", mood="dark", track_ids=["T1"], arc_type="")]
    save_history(ConceptHistory(runs=[entry]), history_path)

    fresh_remote_body = _serialize([entry])
    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(
        side_effect=[
            Response(404),  # initial pull (step 1) — keep local as seed
            Response(200, text=fresh_remote_body, headers={"ETag": '"fresh"'}),  # conflict retry re-pull
        ]
    )
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(
        return_value=Response(200, json=[{"eventId": "f_1", "runId": "r1", "conceptId": "c1", "verdict": "played"}])
    )
    put_route = respx.put(f"{_BASE}{_HISTORY_PATH}").mock(
        side_effect=[
            Response(412, text="etag mismatch"),
            Response(200, headers={"ETag": '"final"'}),
        ]
    )
    ack_route = respx.post(f"{_BASE}{_FEEDBACK_ACK_PATH}").mock(return_value=Response(200))

    report = sync_down(_remote(), history_path)

    assert report == SyncReport(applied=1, skipped=0, acked=1)
    assert put_route.call_count == 2
    second_put_request = put_route.calls[1].request
    assert second_put_request.headers.get("If-Match") == '"fresh"'
    reloaded = load_history(history_path)
    assert reloaded.runs[0].concepts[0].feedback == "played"
    assert _etag_path(history_path).read_text() == '"final"'
    assert json.loads(ack_route.calls.last.request.content) == {"eventIds": ["f_1"]}


@respx.mock
def test_sync_down_second_conflict_raises(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    entry = _entry("r1")
    entry.concepts = [ConceptRecord(concept_id="c1", title="Midnight Run", mood="dark", track_ids=["T1"], arc_type="")]
    save_history(ConceptHistory(runs=[entry]), history_path)

    fresh_remote_body = _serialize([entry])
    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(
        side_effect=[
            Response(404),
            Response(200, text=fresh_remote_body, headers={"ETag": '"fresh"'}),
        ]
    )
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(
        return_value=Response(200, json=[{"eventId": "f_1", "runId": "r1", "conceptId": "c1", "verdict": "played"}])
    )
    respx.put(f"{_BASE}{_HISTORY_PATH}").mock(
        side_effect=[
            Response(412, text="etag mismatch"),
            Response(412, text="etag mismatch again"),
        ]
    )
    ack_route = respx.post(f"{_BASE}{_FEEDBACK_ACK_PATH}").mock(return_value=Response(200))

    with pytest.raises(RemoteConflictError):
        sync_down(_remote(), history_path)

    assert ack_route.call_count == 0


@respx.mock
def test_sync_down_no_pending_events_short_circuits(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(404))
    respx.get(f"{_BASE}{_FEEDBACK_PENDING_PATH}").mock(return_value=Response(200, json=[]))
    put_route = respx.put(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200))
    ack_route = respx.post(f"{_BASE}{_FEEDBACK_ACK_PATH}").mock(return_value=Response(200))

    report = sync_down(_remote(), history_path)

    assert report == SyncReport()
    assert put_route.call_count == 0
    assert ack_route.call_count == 0


# ---------------------------------------------------------------------------
# sync_up
# ---------------------------------------------------------------------------


@respx.mock
def test_sync_up_happy_path_sends_if_match_and_updates_etag(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    save_history(ConceptHistory(runs=[_entry("r1")]), history_path)
    _etag_path(history_path).write_text('"cached"')

    route = respx.put(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200, headers={"ETag": '"new"'}))

    sync_up(_remote(), history_path)

    request = route.calls.last.request
    assert request.headers.get("If-Match") == '"cached"'
    assert _etag_path(history_path).read_text() == '"new"'


@respx.mock
def test_sync_up_no_cached_etag_omits_if_match(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    save_history(ConceptHistory(runs=[_entry("r1")]), history_path)

    route = respx.put(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200, headers={"ETag": '"new"'}))

    sync_up(_remote(), history_path)

    assert "If-Match" not in route.calls.last.request.headers


@respx.mock
def test_sync_up_conflict_merges_unique_run_ids_from_both_sides(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    save_history(ConceptHistory(runs=[_entry("B"), _entry("C")]), history_path)
    _etag_path(history_path).write_text('"cached"')

    remote_body = _serialize([_entry("A"), _entry("B")])
    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200, text=remote_body, headers={"ETag": '"fresh"'}))
    put_route = respx.put(f"{_BASE}{_HISTORY_PATH}").mock(
        side_effect=[
            Response(412, text="etag mismatch"),
            Response(200, headers={"ETag": '"merged"'}),
        ]
    )

    sync_up(_remote(), history_path)

    assert put_route.call_count == 2
    second_request = put_route.calls[1].request
    assert second_request.headers.get("If-Match") == '"fresh"'
    merged_body = json.loads(second_request.content)
    run_ids = {run["run_id"] for run in merged_body["runs"]}
    assert run_ids == {"A", "B", "C"}
    assert _etag_path(history_path).read_text() == '"merged"'


@respx.mock
def test_sync_up_second_conflict_raises(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    save_history(ConceptHistory(runs=[_entry("B")]), history_path)

    remote_body = _serialize([_entry("A")])
    respx.get(f"{_BASE}{_HISTORY_PATH}").mock(return_value=Response(200, text=remote_body, headers={"ETag": '"fresh"'}))
    respx.put(f"{_BASE}{_HISTORY_PATH}").mock(
        side_effect=[
            Response(412, text="etag mismatch"),
            Response(412, text="etag mismatch again"),
        ]
    )

    with pytest.raises(RemoteConflictError):
        sync_up(_remote(), history_path)
