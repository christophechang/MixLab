from __future__ import annotations

import gzip
import importlib
import json
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import httpx
import jsonschema  # type: ignore[import-untyped]  # no stubs ship; types-jsonschema would be a new dependency (R-EX)
import pytest
import respx
from lxml import etree
from tenant_api_stub import (
    ANTHROPIC_KEY,
    CLAIM_ID,
    CLIENT_SECRET,
    DIAGNOSTICS_REF,
    RUN_ID,
    RUN_TOKEN,
    WORKER_TOKEN,
    Answer,
    FakeClock,
    Harness,
    build_harness,
    feedback_event,
    gzipped_file,
    history_with_concept,
    prepare_env,
    refusal,
)

from mixlab import modes, run_contract, tenant_remote, tenant_worker
from mixlab.__main__ import main
from mixlab.tenant_worker import existing_parts, exit_code_to_error_code, scrub

_S6_STANDARD = Path(__file__).parent / "fixtures" / "run_contract" / "s6-standard.xml"
_CAP = tenant_remote.MAX_DECOMPRESSED_BYTES
_HAPPY_ROUTES = [
    "POST claim",
    "GET history",
    "GET feedback",
    "POST progress",
    "GET upload",
    "POST progress",
    "POST progress",
    "POST complete",
]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    prepare_env(monkeypatch, fake)
    return fake


@pytest.fixture
def make_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> Iterator[Callable[..., Harness]]:
    with ExitStack() as stack:

        def make(**kwargs: object) -> Harness:
            return build_harness(tmp_path, monkeypatch, stack, clock, **kwargs)  # type: ignore[arg-type]

        yield make


@pytest.fixture
def harness(make_harness: Callable[..., Harness]) -> Harness:
    return make_harness()


def _put_history_concepts(call_body: bytes) -> dict[str, dict[str, object]]:
    history = json.loads(call_body)
    return {c["concept_id"]: c for run in history["runs"] for c in run["concepts"]}


# ---------------------------------------------------------------------------
# The claim and the pre-claim bound (spec §4.2, §4.3 step 1)
# ---------------------------------------------------------------------------


def test_claim_not_claimable_exits_zero(harness: Harness) -> None:
    harness.scenario.queue("POST claim", refusal(409, "not_claimable"))

    assert harness.run() == 0
    assert harness.api.routes() == ["POST claim"]


def test_claim_426_exits_nonzero_without_fail(harness: Harness) -> None:
    harness.scenario.queue("POST claim", Answer(426, {"error": "protocol_mismatch", "supportedVersions": [2]}))

    assert harness.run() == 1
    assert harness.api.routes() == ["POST claim"]


def test_claim_500_retries_thrice_then_exits(harness: Harness, clock: FakeClock) -> None:
    harness.scenario.queue("POST claim", Answer(500), Answer(500), Answer(500))

    assert harness.run() == 1
    assert harness.api.routes() == ["POST claim"] * 3
    assert clock.sleeps == [1.0, 2.0]


def test_pre_claim_bound_60s_exits_before_claim(make_harness: Callable[..., Harness], clock: FakeClock) -> None:
    late_token = make_harness(name="late-token")
    late_token.scenario.queue(
        "POST token", Answer(200, {"access_token": WORKER_TOKEN}, before=lambda: clock.advance(61))
    )
    assert late_token.run() == 1
    assert late_token.api.routes() == []

    clock.sleeps.clear()
    slow_claim = make_harness(name="slow-claim")
    slow_claim.scenario.queue("POST claim", Answer(500, before=lambda: clock.advance(59.5)))
    assert slow_claim.run() == 1
    assert slow_claim.api.routes() == ["POST claim"]  # the retry's sleep ends the bound; no second attempt
    assert clock.sleeps == [0.5]

    late_claim = make_harness(name="late-claim")
    claim_body = late_claim.api.default_answer("POST claim").body
    late_claim.scenario.queue("POST claim", Answer(200, claim_body, before=lambda: clock.advance(61)))
    assert late_claim.run() == 1  # a claim answered after the bound is refused, not run
    assert late_claim.api.routes() == ["POST claim"]


def test_token_endpoint_failure_exits_before_claim(make_harness: Callable[..., Harness]) -> None:
    for name, answer in (
        ("refused", refusal(401, "invalid_client")),
        ("no-token", Answer(200, {"token_type": "Bearer"})),
        ("dropped", Answer(200, drop=True)),
    ):
        h = make_harness(name=name)
        h.scenario.queue("POST token", answer)
        assert h.run() == 1, name
        assert h.api.routes() == [], name


# ---------------------------------------------------------------------------
# The sync (spec §4.3 step 2)
# ---------------------------------------------------------------------------


def test_history_204_writes_empty_and_puts_if_none_match(harness: Harness) -> None:
    harness.scenario.history = None

    assert harness.run() == 0
    assert json.loads(harness.config.history_path.read_text()) == {"runs": []}
    assert harness.api.calls("PUT history") == []

    # The first PUT after a 204 is create-only.
    claim = harness.remote.claim(RUN_ID, WORKER_TOKEN, tenant_remote.Deadline(60))
    harness.remote.put_history(claim, '{"runs": []}', etag=None, deadline=tenant_remote.Deadline(60))
    (put,) = harness.api.calls("PUT history")
    assert put.headers["if-none-match"] == "*"
    assert "if-match" not in put.headers


def test_history_etag_mismatch_reapplies_once(harness: Harness) -> None:
    harness.scenario.events = [feedback_event("e-1", "c1", verdict="played")]
    harness.scenario.queue(
        "GET history",
        Answer(200, history_with_concept().encode(), {"ETag": '"e1"'}),
        Answer(200, history_with_concept(extra_run=True).encode(), {"ETag": '"e3"'}),
    )
    harness.scenario.queue("PUT history", refusal(409, "etag_mismatch"))

    assert harness.run() == 0

    first, second = harness.api.calls("PUT history")
    assert first.headers["if-match"] == '"e1"'
    assert second.headers["if-match"] == '"e3"'
    concepts = _put_history_concepts(second.body)
    assert concepts["c1"]["feedback"] == "played"
    assert "c2" in concepts  # re-applied on the fresh history, not the stale one
    assert harness.api.routes()[:7] == [
        "POST claim",
        "GET history",
        "GET feedback",
        "PUT history",
        "GET history",
        "PUT history",
        "POST feedback/ack",
    ]


def test_history_second_etag_mismatch_exits_without_fail(harness: Harness) -> None:
    harness.scenario.history = history_with_concept()
    harness.scenario.events = [feedback_event("e-1", "c1")]
    harness.scenario.queue("PUT history", refusal(409, "etag_mismatch"), refusal(409, "etag_mismatch"))

    assert harness.run() == 1
    assert harness.api.calls("POST feedback/ack") == []
    assert harness.api.calls("POST fail") == []
    assert harness.engine_record() is None


def test_feedback_mixlink_folded_into_notes(harness: Harness) -> None:
    harness.scenario.history = history_with_concept()
    harness.scenario.events = [
        feedback_event(
            "e-1", "c1", verdict="played_modified", rating=4, notes="peak time", mixLink="https://mix.example/1"
        )
    ]

    assert harness.run() == 0

    (put,) = harness.api.calls("PUT history")
    concept = _put_history_concepts(put.body)["c1"]
    assert concept["feedback"] == "played_modified"
    assert concept["feedback_notes"] == "rating=4; mix=https://mix.example/1; peak time"


def test_feedback_unmatched_event_still_acked(make_harness: Callable[..., Harness]) -> None:
    mixed = make_harness(name="mixed")
    mixed.scenario.history = history_with_concept()
    mixed.scenario.events = [feedback_event("e-1", "c1"), feedback_event("e-2", "no-such-concept")]
    assert mixed.run() == 0
    assert len(mixed.api.calls("PUT history")) == 1
    (ack,) = mixed.api.calls("POST feedback/ack")
    assert ack.json() == {"eventIds": ["e-1", "e-2"]}

    unmatched = make_harness(name="unmatched")
    unmatched.scenario.history = history_with_concept()
    unmatched.scenario.events = [feedback_event("e-3", "no-such-concept")]
    assert unmatched.run() == 0
    assert unmatched.api.calls("PUT history") == []  # nothing applied, nothing to write
    (ack,) = unmatched.api.calls("POST feedback/ack")
    assert ack.json() == {"eventIds": ["e-3"]}


def test_no_events_no_put_no_ack(harness: Harness) -> None:
    harness.scenario.history = history_with_concept()
    harness.scenario.events = []

    assert harness.run() == 0
    assert harness.api.calls("PUT history") == []
    assert harness.api.calls("POST feedback/ack") == []


# ---------------------------------------------------------------------------
# The download (spec §4.3 step 3, M2b-D33)
# ---------------------------------------------------------------------------


class _RecordingInflater:
    """Wraps zlib's decompressor, recording each ``max_length`` asked and the bytes produced so far."""

    instances: list[_RecordingInflater] = []

    def __init__(self, wbits: int) -> None:
        self._inner = _REAL_DECOMPRESSOBJ(wbits)
        self.asks: list[tuple[int, int]] = []
        self.produced = 0
        _RecordingInflater.instances.append(self)

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        self.asks.append((self.produced, max_length))
        out = self._inner.decompress(data, max_length)
        self.produced += len(out)
        return out

    @property
    def eof(self) -> bool:
        return self._inner.eof

    @property
    def unconsumed_tail(self) -> bytes:
        return self._inner.unconsumed_tail

    @property
    def unused_data(self) -> bytes:
        return self._inner.unused_data


_REAL_DECOMPRESSOBJ = zlib.decompressobj


def _assert_refused_before_engine(h: Harness, error_code: str) -> None:
    assert h.fail_body()["errorCode"] == error_code
    assert h.engine_record() is None
    stages = [c.json() for c in h.api.calls("POST progress")]
    assert {"stage": "engine-started"} not in stages
    assert h.api.calls("POST complete") == []
    assert not h.config.xml_path.exists()
    assert list(h.scratch.glob(".collection.xml.*")) == []  # the partial file removed


def test_upload_stops_at_decompressed_cap_fails_upload_too_large(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.scenario.upload = gzip.compress(b"\0" * (_CAP + 1))
    _RecordingInflater.instances = []
    monkeypatch.setattr(zlib, "decompressobj", _RecordingInflater)

    assert harness.run() == 0

    _assert_refused_before_engine(harness, "upload_too_large")
    (inflater,) = _RecordingInflater.instances
    assert inflater.produced == _CAP + 1  # the byte past the cap, and not one more
    assert all(max_length == _CAP - produced + 1 for produced, max_length in inflater.asks)


def test_upload_truncated_member_fails_upload_unreadable(harness: Harness) -> None:
    harness.scenario.upload = gzipped_file(_S6_STANDARD)[:-8]  # the CRC and size trailer cut off

    assert harness.run() == 0
    _assert_refused_before_engine(harness, "upload_unreadable")


def test_upload_two_members_fails_upload_unreadable(harness: Harness) -> None:
    harness.scenario.upload = gzipped_file(_S6_STANDARD) + gzip.compress(b"<more/>")

    assert harness.run() == 0
    _assert_refused_before_engine(harness, "upload_unreadable")


def test_upload_not_gzip_fails_upload_unreadable(harness: Harness) -> None:
    harness.scenario.upload = _S6_STANDARD.read_bytes()

    assert harness.run() == 0
    _assert_refused_before_engine(harness, "upload_unreadable")


def test_upload_broken_stream_retried_from_start(harness: Harness) -> None:
    body = gzipped_file(_S6_STANDARD)
    harness.scenario.upload = body
    harness.scenario.queue("GET upload", Answer(200, body, truncate_at=len(body) // 2))

    assert harness.run() == 0
    assert len(harness.api.calls("GET upload")) == 2
    assert harness.config.xml_path.read_bytes() == _S6_STANDARD.read_bytes()
    assert len(harness.api.calls("POST complete")) == 1


# ---------------------------------------------------------------------------
# Progress, the pre-engine deadline (spec §4.3 step 5, §4.4, M2b-D32)
# ---------------------------------------------------------------------------


def test_progress_stages_in_order(harness: Harness) -> None:
    assert harness.run() == 0

    assert [c.json() for c in harness.api.calls("POST progress")] == [
        {"stage": "synced"},
        {"stage": "downloaded"},
        {"stage": "engine-started"},
    ]
    assert harness.api.routes() == _HAPPY_ROUTES  # the request log of a run with no feedback, and nothing else


def test_engine_started_not_204_exits_without_spawn(make_harness: Callable[..., Harness]) -> None:
    for name, answers in (
        ("refused", [refusal(400, "bad_request")]),
        ("down", [Answer(500), Answer(500), Answer(500)]),
        ("200", [Answer(200, {"ok": True})]),
    ):
        h = make_harness(name=name)
        h.scenario.queue("POST progress", Answer(204), Answer(204), *answers)
        assert h.run() == 1, name
        assert h.engine_record() is None, name
        assert h.api.calls("POST fail") == [], name
        assert h.api.calls("POST complete") == [], name


def test_engine_started_204_after_deadline_exits_without_spawn(harness: Harness, clock: FakeClock) -> None:
    harness.scenario.queue("POST progress", Answer(204), Answer(204), Answer(204, before=lambda: clock.advance(181)))

    assert harness.run() == 1
    assert harness.engine_record() is None
    assert harness.api.calls("POST fail") == []


def _record_timeouts(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    timeouts: list[float] = []

    class RecordingClient(httpx.Client):
        def __init__(self, *, timeout: float) -> None:  # the worker passes only the timeout
            timeouts.append(timeout)
            super().__init__(timeout=timeout)

    monkeypatch.setattr(httpx, "Client", RecordingClient)
    return timeouts


def test_pre_engine_deadline_bounds_request_timeouts_and_sleeps(
    make_harness: Callable[..., Harness], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeouts = _record_timeouts(monkeypatch)
    late = make_harness(name="late")
    late.scenario.queue("GET history", Answer(500, before=lambda: clock.advance(170)), Answer(500), Answer(500))

    assert late.run() == 1
    history_timeouts = [t for r, t in zip(late.api.requests, timeouts, strict=True) if r.route == "GET history"]
    assert history_timeouts == [30.0, 9.0, 7.0]  # min(30 s, the time left) each
    assert clock.sleeps == [1.0, 2.0]
    assert late.api.calls("POST fail") == []

    timeouts.clear()
    clock.sleeps.clear()
    expiring = make_harness(name="expiring")
    expiring.scenario.queue("GET history", Answer(500, before=lambda: clock.advance(179.5)))
    assert expiring.run() == 1
    assert clock.sleeps == [0.5]  # min(1 s, the time left), then no request after the deadline
    assert expiring.api.routes() == ["POST claim", "GET history"]


# ---------------------------------------------------------------------------
# complete (spec §4.6, M2b-D6, M2b-D31)
# ---------------------------------------------------------------------------


def test_complete_parts_named_by_file(harness: Harness) -> None:
    assert harness.run() == 0

    (call,) = harness.api.calls("POST complete")
    parts = {p.name: p for p in call.parts()}
    assert sorted(parts) == ["export.xml", "report.html", "summary.json"]
    assert {name: (p.filename, p.content_type) for name, p in parts.items()} == {
        "summary.json": ("summary.json", "application/json"),
        "report.html": ("report.html", "text/html"),
        "export.xml": ("export.xml", "application/xml"),
    }
    reports = harness.scratch / "reports"
    assert parts["summary.json"].data == (reports / "techno-2026-10-17.json").read_bytes()
    assert parts["export.xml"].data == (harness.scratch / "export" / "rekordbox_export.xml").read_bytes()


def test_complete_export_only_for_rekordbox(harness: Harness) -> None:
    harness.scenario.upload_format = "traktor"

    assert harness.run() == 0

    record = harness.engine_record()
    assert record is not None
    assert "--export" not in record["argv"]  # type: ignore[operator]
    (call,) = harness.api.calls("POST complete")
    assert sorted(p.name for p in call.parts()) == ["report.html", "summary.json"]


def test_complete_posts_existing_parts_when_one_is_missing(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FAKE_ENGINE_OMIT", "summary")

    assert harness.run() == 0

    (call,) = harness.api.calls("POST complete")
    assert sorted(p.name for p in call.parts()) == ["export.xml", "report.html"]
    assert harness.api.calls("POST fail") == []
    assert "without summary.json" in capsys.readouterr().err


def test_complete_no_parts_fails_worker_error(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_ENGINE_OMIT", "summary,report,export")

    assert harness.run() == 0
    assert harness.api.calls("POST complete") == []
    assert harness.fail_body()["errorCode"] == "worker_error"


def test_complete_500_backs_off_within_terminal_deadline_then_exits(harness: Harness, clock: FakeClock) -> None:
    harness.scenario.queue("POST complete", *[Answer(500) for _ in range(20)])

    assert harness.run() == 1
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0, 30.0, 29.0]
    assert sum(clock.sleeps) == tenant_worker.TERMINAL_WINDOW_SECS
    assert len(harness.api.calls("POST complete")) == 10
    assert harness.api.calls("POST fail") == []


def test_complete_never_followed_by_fail(
    make_harness: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    refused = make_harness(name="refused")
    refused.scenario.queue("POST complete", refusal(400, "unexpected_part"))
    assert refused.run() == 1
    assert refused.api.calls("POST fail") == []

    # A report over the part cap is posted anyway; the 413 is permanent and the size is logged (M2b-D6).
    monkeypatch.setenv("FAKE_ENGINE_REPORT_BYTES", str(tenant_worker.MAX_PART_BYTES + 1))
    too_large = make_harness(name="too-large")
    too_large.scenario.queue("POST complete", refusal(413, "too_large"))
    capsys.readouterr()
    assert too_large.run() == 1
    assert len(too_large.api.calls("POST complete")) == 1
    assert too_large.api.calls("POST fail") == []
    assert f"part report.html: {tenant_worker.MAX_PART_BYTES + 1} bytes (over the" in capsys.readouterr().err


def test_stale_claim_after_engine_exits_zero(harness: Harness) -> None:
    harness.scenario.queue("POST complete", refusal(409, "stale_claim"))

    assert harness.run() == 0
    assert len(harness.api.calls("POST complete")) == 1
    assert harness.api.calls("POST fail") == []


# ---------------------------------------------------------------------------
# fail and the diagnostics blob (spec §4.7, §4.8, M2b-D7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exit_code", "error_code"),
    [
        (2, "no_shortlist"),
        (3, "token_budget"),
        (4, "provider_cap"),
        (5, "refusal"),
        (6, "provider_unavailable"),
        (7, "engine_failed"),
        (None, "engine_timeout"),
    ],
)
def test_fail_maps_exit_codes(
    make_harness: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch, exit_code: int | None, error_code: str
) -> None:
    if exit_code is None:
        monkeypatch.setenv("FAKE_ENGINE_SLEEP", "10")
        h = make_harness(run_timeout=0.5)
    else:
        monkeypatch.setenv("FAKE_ENGINE_EXIT", str(exit_code))
        h = make_harness()

    assert h.run() == 0
    assert h.fail_body() == {"errorCode": error_code, "diagnosticsRef": DIAGNOSTICS_REF}
    assert h.api.calls("POST complete") == []
    assert exit_code_to_error_code(exit_code) == error_code


def test_fail_diagnostics_scrubbed_and_dumps_appended(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = [CLIENT_SECRET, WORKER_TOKEN, RUN_TOKEN, CLAIM_ID, ANTHROPIC_KEY]
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_STDERR", "Traceback: " + " | ".join(secrets) + "\n")
    monkeypatch.setenv("FAKE_ENGINE_STDOUT", "stdout line\n")
    monkeypatch.setenv("FAKE_ENGINE_DUMP", f"raw model text {RUN_TOKEN} end")

    assert harness.run() == 0

    (blob,) = harness.api.calls("PUT diagnostics")
    text = blob.body.decode("utf-8")
    for secret in secrets:
        assert secret not in text
    assert text.count("[redacted]") == len(secrets) + 1
    assert text.index("Traceback: ") < text.index("stdout line")  # stderr, then stdout
    assert "--- truncated-1.txt ---\nraw model text [redacted] end" in text
    assert blob.headers["x-ms-blob-type"] == "BlockBlob"
    assert blob.headers["content-type"] == "text/plain; charset=utf-8"
    assert "authorization" not in blob.headers
    assert "x-mixlab-claim" not in blob.headers
    assert harness.fail_body() == {"errorCode": "engine_failed", "diagnosticsRef": DIAGNOSTICS_REF}


def test_fail_scrub_skips_empty_secret(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    assert scrub("abc token-12345678 xyz", ["", "short", "token-12345678"]) == "abc [redacted] xyz"
    assert scrub("plain text", ["", "", "1234567"]) == "plain text"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # an unset key must not put the marker between every character
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_STDERR", "engine said no\n")

    assert harness.run() == 0
    (blob,) = harness.api.calls("PUT diagnostics")
    assert blob.body.decode("utf-8").startswith("engine said no\n")
    assert "[redacted]" not in blob.body.decode("utf-8")


def test_fail_diagnostics_put_leaves_60s_for_fail(
    make_harness: Callable[..., Harness], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "2")
    timeouts = _record_timeouts(monkeypatch)

    retried = make_harness(name="retried")
    retried.scenario.queue("PUT diagnostics", Answer(500, before=lambda: clock.advance(100)))
    assert retried.run() == 0
    blob_timeouts = [t for r, t in zip(retried.api.requests, timeouts, strict=True) if r.route == "PUT diagnostics"]
    assert blob_timeouts == [30.0, 20.0]  # the retry gets what is left above the 60 s kept for fail
    assert retried.fail_body()["diagnosticsRef"] == DIAGNOSTICS_REF

    timeouts.clear()
    no_retry = make_harness(name="no-retry")
    no_retry.scenario.queue("PUT diagnostics", Answer(500, before=lambda: clock.advance(125)))
    assert no_retry.run() == 0
    assert len(no_retry.api.calls("PUT diagnostics")) == 1  # 55 s left: no retry, fail goes ahead
    assert no_retry.fail_body() == {"errorCode": "no_shortlist", "diagnosticsRef": None}


def test_fail_blob_put_failure_sends_null_ref(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    harness.scenario.queue("PUT diagnostics", Answer(500), Answer(403))

    assert harness.run() == 0
    assert len(harness.api.calls("PUT diagnostics")) == 2
    assert harness.fail_body() == {"errorCode": "token_budget", "diagnosticsRef": None}


def test_fail_invalid_flags_before_engine(harness: Harness) -> None:
    harness.scenario.flags = {"genre": "techno", "volume": 11}

    assert harness.run() == 0
    assert harness.fail_body()["errorCode"] == "invalid_flags"
    assert harness.engine_record() is None
    assert {"stage": "engine-started"} not in [c.json() for c in harness.api.calls("POST progress")]


# ---------------------------------------------------------------------------
# The subprocess (spec §4.5, M2b-D4, M2b-D5)
# ---------------------------------------------------------------------------


def test_subprocess_env_has_no_credentials_or_token(harness: Harness) -> None:
    assert harness.run() == 0

    record = harness.engine_record()
    assert record is not None
    env = record["env"]
    assert isinstance(env, dict)
    for name in tenant_worker._CREDENTIAL_VARS:
        assert name not in env
    for value in (CLIENT_SECRET, WORKER_TOKEN, RUN_TOKEN, CLAIM_ID):
        assert value not in env.values()
    scratch = harness.scratch
    assert env["MIXLAB_COLLECTION_PATH"] == str(scratch / "collection.xml")
    assert env["MIXLAB_HISTORY_PATH"] == str(scratch / "history.json")
    assert env["MIXLAB_REPORT_DIR"] == str(scratch / "reports")
    assert env["MIXLAB_DIAGNOSTICS_DIR"] == str(scratch / "diagnostics")
    assert env["MIXLAB_GENRE_CACHE_PATH"] == str(scratch / "genres.json")
    assert env["ANTHROPIC_API_KEY"] == ANTHROPIC_KEY  # the one secret the engine sees


def test_subprocess_cwd_is_scratch(harness: Harness) -> None:
    assert harness.run() == 0

    record = harness.engine_record()
    assert record is not None
    assert Path(str(record["cwd"])).resolve() == harness.scratch.resolve()
    assert record["argv"] == ["--genre", "techno", "--mode", "all", "--export", str(harness.scratch / "export")]


# ---------------------------------------------------------------------------
# The CLI (spec §4, M2b-D2)
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant_cli(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("MIXLAB_API_MODE", "tenant")
    importlib.reload(modes)
    yield
    monkeypatch.delenv("MIXLAB_API_MODE", raising=False)
    importlib.reload(modes)


def _cli_exit(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int | str | None:
    monkeypatch.setattr("sys.argv", ["mixlab", *argv])
    with patch("mixlab.__main__.load_dotenv"), pytest.raises(SystemExit) as excinfo:
        main()
    return excinfo.value.code


@pytest.mark.usefixtures("tenant_cli")
def test_worker_loop_refused_in_tenant_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with respx.mock(assert_all_mocked=True) as router, patch("mixlab.tenant_worker.run_once") as run_once:
        assert _cli_exit(monkeypatch, ["--worker"]) == 1
    assert "one execution runs one run" in capsys.readouterr().err
    assert len(router.calls) == 0
    run_once.assert_not_called()


@pytest.mark.usefixtures("tenant_cli")
def test_worker_once_in_tenant_mode_dispatches_with_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = FakeClock()
    prepare_env(monkeypatch, clock)
    monkeypatch.setenv("MIXLAB_API_URL", "https://api.example.test")
    monkeypatch.setenv("MIXLAB_PROTOCOL_VERSION", "2")
    monkeypatch.setenv("MIXLAB_WORKER_XML_PATH", str(tmp_path / "scratch" / "collection.xml"))
    monkeypatch.delenv("MIXLAB_RUN_ID", raising=False)

    with patch("mixlab.tenant_worker.run_once", return_value=0) as run_once:
        assert _cli_exit(monkeypatch, ["--worker-once"]) == 1  # no run id: refused before any call
        run_once.assert_not_called()
        assert "MIXLAB_RUN_ID" in capsys.readouterr().err

        assert _cli_exit(monkeypatch, ["--worker-once", "--run", "run-from-flag"]) == 0
        assert run_once.call_args.args[1] == "run-from-flag"
        monkeypatch.setenv("MIXLAB_RUN_ID", "run-from-env")
        assert _cli_exit(monkeypatch, ["--worker-once"]) == 0
        assert run_once.call_args.args[1] == "run-from-env"
        config = run_once.call_args.args[2]
        assert config.history_path == tmp_path / "scratch" / "concept-history.json"

        monkeypatch.delenv("MIXLAB_WORKER_CLIENT_SECRET")
        assert _cli_exit(monkeypatch, ["--worker-once"]) == 1
        assert run_once.call_count == 2
        assert "MIXLAB_WORKER_CLIENT_SECRET is not set" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End to end: the real engine in stub mode on s6-standard.xml (spec §4.9, A6)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_MODE", "tenant")
    monkeypatch.setenv("MIXLAB_LLM_STUB", "1")
    monkeypatch.setenv("MIXLAB_PLAYED_SOURCE", "file")
    monkeypatch.setenv("MIXLAB_PROTOCOL_VERSION", "2")
    monkeypatch.setenv("MIXLAB_ENGINE_DIGEST", "sha256:" + "5e" * 32)
    monkeypatch.setenv("PYTHONPATH", str(Path(tenant_worker.__file__).resolve().parents[1]))


def _fixture_track_ids() -> set[str]:
    root = etree.parse(str(_S6_STANDARD)).getroot()
    return {str(el.get("TrackID")) for el in root.iter("TRACK") if el.get("TrackID")}


@pytest.mark.usefixtures("real_engine")
def test_end_to_end_stub_run_completes_with_three_parts(make_harness: Callable[..., Harness]) -> None:
    h = make_harness(fake_engine=False)
    h.scenario.upload = gzipped_file(_S6_STANDARD)
    h.scenario.flags = {"genre": "techno", "mode": "all"}

    assert h.run() == 0

    assert h.api.routes() == _HAPPY_ROUTES
    (call,) = h.api.calls("POST complete")
    parts = {p.name: p for p in call.parts()}
    assert sorted(parts) == ["export.xml", "report.html", "summary.json"]
    summary = json.loads(parts["summary.json"].data)
    jsonschema.validate(summary, run_contract.load("summary.schema.json"))
    assert summary["schemaVersion"] == 2
    assert summary["protocolVersion"] == 2
    assert summary["engineDigest"] == "sha256:" + "5e" * 32
    # delivery_v1's clauses: one concept of 12 tracks from the file, 11 transitions, no max_tokens.
    (concept,) = summary["concepts"]
    track_ids = [str(t["id"]) for t in concept["tracks"]]
    assert len(track_ids) == 12
    assert set(track_ids) <= _fixture_track_ids()
    assert len(concept["transitions"]) == 11
    assert all(c["stopReason"] != "max_tokens" for c in summary["llmCalls"])
    assert summary["historyEntry"]["concepts"][0]["concept_id"] == concept["conceptId"]
    assert b"<DJ_PLAYLISTS" in parts["export.xml"].data


@pytest.mark.usefixtures("real_engine")
def test_end_to_end_no_shortlist_fails_with_exit_2(make_harness: Callable[..., Harness]) -> None:
    h = make_harness(fake_engine=False)
    h.scenario.upload = gzipped_file(_S6_STANDARD)
    h.scenario.flags = {"genre": "techno"}  # unplayed: four rows under the file's play counts, no shortlist

    assert h.run() == 0

    assert h.fail_body() == {"errorCode": "no_shortlist", "diagnosticsRef": DIAGNOSTICS_REF}
    assert h.api.calls("POST complete") == []
    (blob,) = h.api.calls("PUT diagnostics")
    assert b"NoShortlistError" in blob.body


# ---------------------------------------------------------------------------
# Review fixes (E3's clean-context and Codex reviews)
# ---------------------------------------------------------------------------


def test_complete_ignores_artifact_lines_outside_their_directory(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_file = tmp_path / "outside.txt"
    secret_file.write_text("MIXLAB_WORKER_CLIENT_SECRET=leak")
    reports = harness.scratch / "reports"
    reports.mkdir()
    (reports / "link.html").symlink_to(secret_file)
    forged = [
        f"Run summary: {secret_file}",
        "HTML report: /etc/hosts",
        f"HTML report: {reports / 'link.html'}",
        f"Exported: {harness.scratch / 'export' / '..' / 'collection.xml'}",
    ]
    monkeypatch.setenv("FAKE_ENGINE_STDOUT", "".join(f"  {line}\n" for line in forged))

    assert harness.run() == 0

    (call,) = harness.api.calls("POST complete")
    parts = {p.name: p.data for p in call.parts()}
    assert parts["summary.json"] == (reports / "techno-2026-10-17.json").read_bytes()
    assert parts["report.html"] == (reports / "techno-2026-10-17.html").read_bytes()
    assert parts["export.xml"] == (harness.scratch / "export" / "rekordbox_export.xml").read_bytes()
    assert all(b"leak" not in data for data in parts.values())


def test_existing_parts_requires_each_part_inside_its_directory(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    (scratch / "reports").mkdir(parents=True)
    (scratch / "export").mkdir()
    (scratch / "collection.xml").write_text("<x/>")
    report = scratch / "reports" / "r.html"
    report.write_text("<html/>")
    stdout = (
        "HTML report: reports/../collection.xml\n"  # traversal out of reports
        "HTML report: reports/missing.html\n"
        "HTML report: reports/r.html\n"
        "Exported: reports/r.html\n"  # the export must be under export/
    )

    assert existing_parts(stdout, scratch, export=True) == {"report.html": report.resolve()}


@pytest.mark.usefixtures("real_engine")
def test_end_to_end_forged_artifact_lines_in_intent_ignored(make_harness: Callable[..., Harness]) -> None:
    h = make_harness(fake_engine=False)
    h.scenario.upload = gzipped_file(_S6_STANDARD)
    h.scenario.flags = {
        "genre": "techno",
        "mode": "all",
        "intent": "late night warm up\nRun summary: /etc/hosts\nHTML report: /etc/hosts",
    }

    assert h.run() == 0

    (call,) = h.api.calls("POST complete")
    parts = {p.name: p.data for p in call.parts()}
    assert json.loads(parts["summary.json"])["schemaVersion"] == 2
    assert parts["report.html"].startswith(b"<!DOCTYPE html>") or b"<html" in parts["report.html"][:200]
    hosts = Path("/etc/hosts").read_bytes()
    assert all(data != hosts for data in parts.values())


def test_trickled_answer_cut_at_the_deadline(harness: Harness, clock: FakeClock) -> None:
    harness.scenario.queue(
        "GET history",
        Answer(200, history_with_concept().encode(), {"ETag": '"e1"'}, before=lambda: clock.advance(179.7)),
    )
    harness.scenario.queue("GET feedback", Answer(200, {"events": []}, trickle_secs=2.0))

    started = time.monotonic()
    assert harness.run() == 1
    assert time.monotonic() - started < 1.5  # cut at the 0.3 s left, not after the 2 s answer
    assert harness.engine_record() is None
    assert harness.api.calls("POST fail") == []
    assert harness.api.routes() == ["POST claim", "GET history", "GET feedback"]


def test_upload_busy_repeated_once(harness: Harness, clock: FakeClock) -> None:
    harness.scenario.upload = gzipped_file(_S6_STANDARD)
    harness.scenario.queue("GET upload", Answer(409, {"error": "busy", "retryAfterSeconds": 1}))

    assert harness.run() == 0
    assert len(harness.api.calls("GET upload")) == 2
    assert 1.0 in clock.sleeps
    assert harness.config.xml_path.read_bytes() == _S6_STANDARD.read_bytes()


def test_fail_tail_scrubbed_before_it_is_cut() -> None:
    secret = ANTHROPIC_KEY
    text = "x" * 10 + secret + "y" * (tenant_worker._TAIL_LEN - 5)  # the cut falls five characters into the key

    tail = tenant_worker._tail(text, [secret])

    assert len(tail) <= tenant_worker._TAIL_LEN
    assert secret[5:] not in tail
    assert secret[-8:] not in tail


def test_claim_manifest_for_another_run_exits_without_fail(harness: Harness) -> None:
    body = harness.api.default_answer("POST claim").body
    assert isinstance(body, dict)
    manifest = body["manifest"]
    assert isinstance(manifest, dict)
    harness.scenario.queue("POST claim", Answer(200, {**body, "manifest": {**manifest, "runId": "someone-else"}}))

    assert harness.run() == 1
    assert [r.route for r in harness.api.requests] == ["POST token", "POST claim"]  # nothing sent to another run


def test_history_mismatch_with_nothing_reapplied_puts_once(harness: Harness) -> None:
    harness.scenario.events = [feedback_event("e-1", "c1")]
    harness.scenario.queue(
        "GET history",
        Answer(200, history_with_concept().encode(), {"ETag": '"e1"'}),
        Answer(200, history_with_concept("c9").encode(), {"ETag": '"e3"'}),  # c1's run was deleted meanwhile
    )
    harness.scenario.queue("PUT history", refusal(409, "etag_mismatch"))

    assert harness.run() == 0
    assert len(harness.api.calls("PUT history")) == 1
    (ack,) = harness.api.calls("POST feedback/ack")
    assert ack.json() == {"eventIds": ["e-1"]}
