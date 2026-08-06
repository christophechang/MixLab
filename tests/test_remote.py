from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import respx
from httpx import Request, Response

from mixlab.remote import (
    ConceptSummary,
    FeedbackEvent,
    MixLabRemote,
    RemoteConfig,
    RemoteConfigError,
    RemoteConflictError,
    RemoteError,
    RunManifest,
)

_BASE = "https://mixlab-api.example.com"
_SECRET = "test-secret"


def _remote() -> MixLabRemote:
    return MixLabRemote(RemoteConfig(base_url=_BASE, api_secret=_SECRET))


def _bearer(request: Request) -> str | None:
    value = request.headers.get("Authorization")
    return str(value) if value is not None else None


# --- RemoteConfig -----------------------------------------------------------------


def test_remote_config_trailing_slash_present_stripped() -> None:
    config = RemoteConfig(base_url="https://example.com/", api_secret="x")
    assert config.base_url == "https://example.com"


def test_remote_config_from_env_missing_url_raises_naming_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIXLAB_API_URL", raising=False)
    monkeypatch.setenv("MIXLAB_API_SECRET", "secret")
    with pytest.raises(RemoteConfigError, match="MIXLAB_API_URL"):
        RemoteConfig.from_env()


def test_remote_config_from_env_missing_secret_raises_naming_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_URL", "https://example.com")
    monkeypatch.delenv("MIXLAB_API_SECRET", raising=False)
    with pytest.raises(RemoteConfigError, match="MIXLAB_API_SECRET"):
        RemoteConfig.from_env()


def test_remote_config_from_env_present_builds_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_URL", "https://example.com/")
    monkeypatch.setenv("MIXLAB_API_SECRET", "secret")
    config = RemoteConfig.from_env()
    assert config.base_url == "https://example.com"
    assert config.api_secret == "secret"


# --- RunManifest / FeedbackEvent parsing -------------------------------------------


def test_run_manifest_from_json_camel_case_full() -> None:
    data = {
        "runId": "r_1",
        "createdAt": "2026-07-08T00:00:00Z",
        "status": "succeeded",
        "flags": {"genre": "house"},
        "uploadId": "u_1",
        "claimedAt": "2026-07-08T00:01:00Z",
        "workerId": "w_1",
        "concepts": [{"conceptId": "c_1", "title": "107 Degrees"}],
    }
    manifest = RunManifest.from_json(data)
    assert manifest.run_id == "r_1"
    assert manifest.claimed_at == "2026-07-08T00:01:00Z"
    assert manifest.worker_id == "w_1"
    assert manifest.concepts == [ConceptSummary(concept_id="c_1", title="107 Degrees")]


def test_run_manifest_from_json_absent_optionals_tolerated() -> None:
    data = {
        "runId": "r_1",
        "createdAt": "2026-07-08T00:00:00Z",
        "status": "queued",
        "uploadId": "u_1",
    }
    manifest = RunManifest.from_json(data)
    assert manifest.claimed_at is None
    assert manifest.worker_id is None
    assert manifest.flags == {}
    assert manifest.concepts == []


def test_feedback_event_from_json_camel_case_full() -> None:
    data = {
        "eventId": "f_1",
        "runId": "r_1",
        "conceptId": "c_1",
        "verdict": "played",
        "rating": 4,
        "notes": "great set",
        "publishedMixSlug": "107-degrees",
        "recordedAt": "2026-07-08T00:02:00Z",
    }
    event = FeedbackEvent.from_json(data)
    assert event.event_id == "f_1"
    assert event.rating == 4
    assert event.published_mix_slug == "107-degrees"


def test_feedback_event_from_json_absent_optionals_tolerated() -> None:
    data = {"eventId": "f_1", "runId": "r_1", "conceptId": "c_1", "recordedAt": "2026-07-08T00:02:00Z"}
    event = FeedbackEvent.from_json(data)
    assert event.verdict is None
    assert event.rating is None
    assert event.notes is None
    assert event.published_mix_slug is None


# --- claim --------------------------------------------------------------------------


@respx.mock
def test_claim_happy_path_returns_manifest() -> None:
    route = respx.post(f"{_BASE}/api/mixlab/worker/claim").mock(
        return_value=Response(
            200,
            json={
                "runId": "r_1",
                "createdAt": "2026-07-08T00:00:00Z",
                "status": "running",
                "flags": {},
                "uploadId": "u_1",
                "claimedAt": "2026-07-08T00:01:00Z",
                "workerId": "w_1",
                "concepts": [],
            },
        )
    )
    manifest = _remote().claim("w_1")
    assert isinstance(manifest, RunManifest)
    assert manifest.run_id == "r_1"
    request = route.calls.last.request
    assert _bearer(request) == f"Bearer {_SECRET}"
    assert json.loads(request.content) == {"workerId": "w_1"}


@respx.mock
def test_claim_empty_queue_returns_none() -> None:
    respx.post(f"{_BASE}/api/mixlab/worker/claim").mock(return_value=Response(204))
    assert _remote().claim("w_1") is None


@respx.mock
def test_claim_server_error_raises_remote_error_with_status_and_body() -> None:
    respx.post(f"{_BASE}/api/mixlab/worker/claim").mock(return_value=Response(500, text="boom"))
    with pytest.raises(RemoteError) as exc_info:
        _remote().claim("w_1")
    assert exc_info.value.status == 500
    assert "boom" in str(exc_info.value)


# --- download_collection --------------------------------------------------------------


@respx.mock
def test_download_collection_gzip_round_trip_writes_decompressed_file(tmp_path: Path) -> None:
    xml_bytes = b"<DJ_PLAYLISTS><COLLECTION>hello world</COLLECTION></DJ_PLAYLISTS>"
    route = respx.get(f"{_BASE}/api/mixlab/uploads/u_1").mock(
        return_value=Response(200, content=gzip.compress(xml_bytes))
    )
    dest = tmp_path / "import" / "rekordbox.xml"
    result = _remote().download_collection("u_1", dest)
    assert result == dest
    assert dest.read_bytes() == xml_bytes
    assert _bearer(route.calls.last.request) == f"Bearer {_SECRET}"
    remnants = list(dest.parent.glob("*.tmp"))
    assert remnants == []
    entries = list(dest.parent.iterdir())
    assert entries == [dest]


@respx.mock
def test_download_collection_server_error_raises_remote_error(tmp_path: Path) -> None:
    respx.get(f"{_BASE}/api/mixlab/uploads/u_1").mock(return_value=Response(404, text="not found"))
    dest = tmp_path / "rekordbox.xml"
    with pytest.raises(RemoteError) as exc_info:
        _remote().download_collection("u_1", dest)
    assert exc_info.value.status == 404
    assert not dest.exists()
    assert list(dest.parent.glob("*.tmp")) == []


# --- complete -------------------------------------------------------------------------


@respx.mock
def test_complete_happy_path_sends_multipart_fields(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.html"
    export_path = tmp_path / "export.xml"
    summary_path.write_text('{"schemaVersion": 1}')
    report_path.write_text("<html></html>")
    export_path.write_text("<xml/>")

    route = respx.post(f"{_BASE}/api/mixlab/runs/r_1/complete").mock(return_value=Response(200))
    _remote().complete(
        "r_1",
        summary_path=summary_path,
        report_path=report_path,
        export_path=export_path,
    )
    request = route.calls.last.request
    assert _bearer(request) == f"Bearer {_SECRET}"
    body = request.content
    assert b'name="summary"' in body
    assert b'name="report"' in body
    assert b'name="export"' in body


@respx.mock
def test_complete_export_none_omits_export_field(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.html"
    summary_path.write_text('{"schemaVersion": 1}')
    report_path.write_text("<html></html>")

    route = respx.post(f"{_BASE}/api/mixlab/runs/r_1/complete").mock(return_value=Response(200))
    _remote().complete(
        "r_1",
        summary_path=summary_path,
        report_path=report_path,
        export_path=None,
    )
    body = route.calls.last.request.content
    assert b'name="summary"' in body
    assert b'name="report"' in body
    assert b'name="export"' not in body


@respx.mock
def test_complete_server_error_raises_remote_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.html"
    summary_path.write_text("{}")
    report_path.write_text("<html></html>")
    respx.post(f"{_BASE}/api/mixlab/runs/r_1/complete").mock(return_value=Response(500, text="db down"))
    with pytest.raises(RemoteError) as exc_info:
        _remote().complete("r_1", summary_path=summary_path, report_path=report_path, export_path=None)
    assert exc_info.value.status == 500


# --- fail -----------------------------------------------------------------------------


@respx.mock
def test_fail_happy_path_sends_error_and_log_tail() -> None:
    route = respx.post(f"{_BASE}/api/mixlab/runs/r_1/fail").mock(return_value=Response(200))
    _remote().fail("r_1", error="boom", log_tail="trace...")
    request = route.calls.last.request
    assert _bearer(request) == f"Bearer {_SECRET}"
    assert json.loads(request.content) == {"error": "boom", "logTail": "trace..."}


# --- history --------------------------------------------------------------------------


@respx.mock
def test_get_history_happy_path_returns_etag_and_body() -> None:
    route = respx.get(f"{_BASE}/api/mixlab/history").mock(
        return_value=Response(200, text="[]", headers={"ETag": '"abc"'})
    )
    etag, body = _remote().get_history()
    assert etag == '"abc"'
    assert body == "[]"
    assert _bearer(route.calls.last.request) == f"Bearer {_SECRET}"


@respx.mock
def test_get_history_not_found_returns_none_and_empty_string() -> None:
    respx.get(f"{_BASE}/api/mixlab/history").mock(return_value=Response(404))
    etag, body = _remote().get_history()
    assert etag is None
    assert body == ""


@respx.mock
def test_get_history_server_error_raises_remote_error() -> None:
    respx.get(f"{_BASE}/api/mixlab/history").mock(return_value=Response(500, text="down"))
    with pytest.raises(RemoteError) as exc_info:
        _remote().get_history()
    assert exc_info.value.status == 500
    assert "down" in str(exc_info.value)


@respx.mock
def test_put_history_no_etag_omits_if_match_header() -> None:
    route = respx.put(f"{_BASE}/api/mixlab/history").mock(return_value=Response(200, headers={"ETag": '"new"'}))
    result = _remote().put_history("[]", etag=None)
    assert result == '"new"'
    request = route.calls.last.request
    assert "If-Match" not in request.headers
    assert _bearer(request) == f"Bearer {_SECRET}"


@respx.mock
def test_put_history_with_etag_sends_if_match_header() -> None:
    route = respx.put(f"{_BASE}/api/mixlab/history").mock(return_value=Response(200, headers={"ETag": '"new"'}))
    _remote().put_history("[]", etag='"old"')
    request = route.calls.last.request
    assert request.headers.get("If-Match") == '"old"'


@respx.mock
def test_put_history_conflict_raises_remote_conflict_error() -> None:
    respx.put(f"{_BASE}/api/mixlab/history").mock(return_value=Response(412, text="etag mismatch"))
    with pytest.raises(RemoteConflictError) as exc_info:
        _remote().put_history("[]", etag='"stale"')
    assert exc_info.value.status == 412


# --- feedback ---------------------------------------------------------------------------


@respx.mock
def test_pending_feedback_happy_path_parses_events() -> None:
    route = respx.get(f"{_BASE}/api/mixlab/feedback/pending").mock(
        return_value=Response(
            200,
            json=[
                {
                    "eventId": "f_1",
                    "runId": "r_1",
                    "conceptId": "c_1",
                    "verdict": "played",
                    "rating": 5,
                    "notes": None,
                    "publishedMixSlug": None,
                    "recordedAt": "2026-07-08T00:03:00Z",
                }
            ],
        )
    )
    events = _remote().pending_feedback()
    assert events == [
        FeedbackEvent(
            event_id="f_1",
            run_id="r_1",
            concept_id="c_1",
            verdict="played",
            rating=5,
            notes=None,
            published_mix_slug=None,
            recorded_at="2026-07-08T00:03:00Z",
        )
    ]
    assert _bearer(route.calls.last.request) == f"Bearer {_SECRET}"


@respx.mock
def test_ack_feedback_nonempty_posts_event_ids() -> None:
    route = respx.post(f"{_BASE}/api/mixlab/feedback/ack").mock(return_value=Response(200))
    _remote().ack_feedback(["f_1", "f_2"])
    request = route.calls.last.request
    assert _bearer(request) == f"Bearer {_SECRET}"
    assert json.loads(request.content) == {"eventIds": ["f_1", "f_2"]}


@respx.mock
def test_ack_feedback_empty_list_skips_request() -> None:
    route = respx.post(f"{_BASE}/api/mixlab/feedback/ack").mock(return_value=Response(200))
    _remote().ack_feedback([])
    assert route.call_count == 0


# --- claim_map / complete_map / fail_map -----------------------------------------------


@respx.mock
def test_claim_map_returns_upload_id_on_200() -> None:
    route = respx.post(f"{_BASE}/api/mixlab/maps/claim").mock(
        return_value=Response(200, json={"uploadId": "up-1", "status": "running"})
    )
    upload_id = _remote().claim_map("w_1")
    assert upload_id == "up-1"
    request = route.calls.last.request
    assert _bearer(request) == f"Bearer {_SECRET}"
    assert json.loads(request.content) == {"workerId": "w_1"}


@respx.mock
def test_claim_map_returns_none_on_204() -> None:
    respx.post(f"{_BASE}/api/mixlab/maps/claim").mock(return_value=Response(204))
    assert _remote().claim_map("w_1") is None


@respx.mock
def test_claim_map_missing_upload_id_raises_remote_error() -> None:
    respx.post(f"{_BASE}/api/mixlab/maps/claim").mock(return_value=Response(200, json={}))
    with pytest.raises(RemoteError):
        _remote().claim_map("w_1")


@respx.mock
def test_complete_map_posts_payload_bytes_verbatim() -> None:
    payload = b'{"cells": []}'
    route = respx.post(f"{_BASE}/api/mixlab/maps/up-1/result").mock(return_value=Response(204))
    _remote().complete_map("up-1", payload)
    request = route.calls.last.request
    assert request.content == payload
    assert request.headers.get("Content-Type") == "application/json"
    assert _bearer(request) == f"Bearer {_SECRET}"


@respx.mock
def test_complete_map_non_2xx_raises() -> None:
    respx.post(f"{_BASE}/api/mixlab/maps/up-1/result").mock(return_value=Response(404, text="not found"))
    with pytest.raises(RemoteError) as exc_info:
        _remote().complete_map("up-1", b"{}")
    assert exc_info.value.status == 404


@respx.mock
def test_fail_map_posts_error_body() -> None:
    route = respx.post(f"{_BASE}/api/mixlab/maps/up-1/fail").mock(return_value=Response(204))
    _remote().fail_map("up-1", error="boom")
    request = route.calls.last.request
    assert _bearer(request) == f"Bearer {_SECRET}"
    assert json.loads(request.content) == {"error": "boom"}
