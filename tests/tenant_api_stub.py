"""An in-process stand-in for the multi-tenant API's worker routes (M2b spec §4.9).

``http.server`` on an ephemeral port, started per test: the token endpoint, the nine routes of
``protocol.md`` and the diagnostics blob URL, each answering from a scripted queue in a
:class:`Scenario` (or its default when the queue is empty), and every request recorded with its
headers, body, JSON and multipart parts.
"""

from __future__ import annotations

import email.parser
import email.policy
import gzip
import json
import os
import stat
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from mixlab import tenant_remote, tenant_worker
from mixlab.tenant_remote import TenantConfig, TenantRemote

RUN_ID = "run-6f1c"
CLIENT_SECRET = "client-secret-7Hq2vLx9"
ANTHROPIC_KEY = "sk-ant-test-3c9d0e1f2a"

TENANT_ID = "tenant-42"
CLAIM_ID = "claim-6f1c2e0a-91b3"
RUN_TOKEN = "run-token-5d2f8a7c9e"
WORKER_TOKEN = "worker-token-b41c7e09d3"

_API_PREFIX = f"/api/worker/runs/{RUN_ID}/"
_BLOB_PATH = f"/blob/tenants/{TENANT_ID}/runs/{RUN_ID}/attempts/{CLAIM_ID}/diagnostics.txt"
DIAGNOSTICS_REF = f"attempts/{CLAIM_ID}/diagnostics.txt"


@dataclass
class Answer:
    """One scripted answer: a status and a body (a dict is JSON, bytes are sent as they are)."""

    status: int
    body: object = None
    headers: dict[str, str] = field(default_factory=dict)
    drop: bool = False  # close the connection without answering: a transport error
    truncate_at: int | None = None  # declare the whole body's length, send this many bytes, close
    before: Callable[[], None] | None = None  # run before answering, e.g. to move a fake clock
    trickle_secs: float | None = None  # send the body a byte at a time, spread over this many real seconds


def busy() -> Answer:
    return Answer(409, {"error": "busy", "detail": "contention", "retryAfterSeconds": 1})


def refusal(status: int, error: str) -> Answer:
    return Answer(status, {"error": error, "detail": "scripted"})


@dataclass(frozen=True)
class Part:
    name: str
    filename: str | None
    content_type: str
    data: bytes


@dataclass(frozen=True)
class Recorded:
    route: str  # "POST claim", "GET history", "POST token", "PUT diagnostics" …
    path: str
    query: str
    headers: dict[str, str]  # lower-cased names
    body: bytes

    def json(self) -> object:
        return json.loads(self.body)

    def parts(self) -> list[Part]:
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + self.headers["content-type"].encode("latin-1") + b"\r\n\r\n" + self.body
        )
        assert isinstance(message, EmailMessage)
        found: list[Part] = []
        for part in message.iter_parts():
            assert isinstance(part, EmailMessage)
            payload = part.get_payload(decode=True)
            found.append(
                Part(
                    name=str(part.get_param("name", header="content-disposition")),
                    filename=part.get_filename(),
                    content_type=part.get_content_type(),
                    data=payload if isinstance(payload, bytes) else b"",
                )
            )
        return found


def gzipped_file(path: Path) -> bytes:
    return gzip.compress(path.read_bytes())


@dataclass
class Scenario:
    """What the API holds for the run and how each route answers."""

    flags: dict[str, object] = field(default_factory=lambda: {"genre": "techno", "mode": "all"})
    upload_format: str = "rekordbox"
    history: str | None = None  # None: GET history answers 204
    etag: str = '"e1"'
    events: list[dict[str, object]] = field(default_factory=list)
    upload: bytes = field(default_factory=lambda: gzip.compress(b'<?xml version="1.0"?><DJ_PLAYLISTS/>'))
    answers: dict[str, deque[Answer]] = field(default_factory=dict)

    def queue(self, route: str, *answers: Answer) -> None:
        self.answers.setdefault(route, deque()).extend(answers)


class StubApi:
    """The server; ``with StubApi(scenario) as api:`` starts and stops it."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.requests: list[Recorded] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.01,), daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    @property
    def token_endpoint(self) -> str:
        return f"{self.url}/token"

    @property
    def diagnostics_url(self) -> str:
        return f"{self.url}{_BLOB_PATH}?sv=2026-01-01&sig=c2lnbmF0dXJl"

    def __enter__(self) -> StubApi:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def record(self, recorded: Recorded) -> None:
        with self._lock:
            self.requests.append(recorded)

    def routes(self) -> list[str]:
        """The API routes called, in order (the token endpoint and the blob left out)."""
        return [r.route for r in self.requests if r.path.startswith(_API_PREFIX)]

    def calls(self, route: str) -> list[Recorded]:
        return [r for r in self.requests if r.route == route]

    def default_answer(self, route: str) -> Answer:
        scenario = self.scenario
        if route == "POST token":
            return Answer(200, {"access_token": WORKER_TOKEN, "token_type": "Bearer", "expires_in": 3599})
        if route == "POST claim":
            manifest = {
                "runId": RUN_ID,
                "uploadId": "upload-1",
                "uploadFormat": scenario.upload_format,
                "createdAt": "2026-10-17T10:00:00Z",
                "flags": scenario.flags,
            }
            body = {
                "claimId": CLAIM_ID,
                "runToken": RUN_TOKEN,
                "leaseExpiresAt": "2026-10-17T10:35:00Z",
                "diagnosticsUploadUrl": self.diagnostics_url,
                "manifest": manifest,
            }
            return Answer(200, body)
        if route == "GET history":
            if scenario.history is None:
                return Answer(204)
            return Answer(200, scenario.history.encode("utf-8"), {"ETag": scenario.etag})
        if route == "PUT history":
            return Answer(204, headers={"ETag": '"e2"'})
        if route == "GET feedback":
            return Answer(200, {"events": scenario.events})
        if route == "GET upload":
            return Answer(200, scenario.upload, {"Content-Type": "application/gzip"})
        if route == "POST complete":
            return Answer(200, {"runId": RUN_ID, "status": "succeeded", "outcome": "Completed"})
        if route == "POST fail":
            return Answer(200, {"runId": RUN_ID, "status": "failed", "outcome": "Failed"})
        if route == "PUT diagnostics":
            return Answer(201)
        return Answer(204)  # feedback/ack, progress

    def answer_for(self, route: str) -> Answer:
        with self._lock:
            queue = self.scenario.answers.get(route)
            if queue:
                return queue.popleft()
        return self.default_answer(route)


def _route_of(method: str, path: str) -> str:
    if path == "/token":
        return f"{method} token"
    if path == _BLOB_PATH:
        return f"{method} diagnostics"
    if path.startswith(_API_PREFIX):
        return f"{method} {path[len(_API_PREFIX) :]}"
    return f"{method} {path}"


def _handler_for(api: StubApi) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _handle(self, method: str) -> None:
            split = urlsplit(self.path)
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            route = _route_of(method, split.path)
            api.record(
                Recorded(
                    route=route,
                    path=split.path,
                    query=split.query,
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=body,
                )
            )
            answer = api.answer_for(route)
            if answer.before is not None:
                answer.before()
            if answer.drop:
                self.close_connection = True
                return
            payload: bytes
            headers = dict(answer.headers)
            if isinstance(answer.body, bytes):
                payload = answer.body
            elif answer.body is None:
                payload = b""
            else:
                payload = json.dumps(answer.body).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            self.send_response(answer.status)
            for name, value in headers.items():
                self.send_header(name, value)
            if answer.status != 204:
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if answer.status == 204:
                return
            if answer.trickle_secs is not None:
                try:
                    for i in range(len(payload)):
                        self.wfile.write(payload[i : i + 1])
                        self.wfile.flush()
                        time.sleep(answer.trickle_secs / max(1, len(payload)))
                except OSError:  # the client gave up: what the test wants
                    pass
                self.close_connection = True
                return
            if answer.truncate_at is not None:
                self.wfile.write(payload[: answer.truncate_at])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 — http.server's name
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802 — http.server's name
            self._handle("POST")

        def do_PUT(self) -> None:  # noqa: N802 — http.server's name
            self._handle("PUT")

    return Handler


def history_with_concept(concept_id: str = "c1", *, extra_run: bool = False) -> str:
    """A history body with one concept (and, with ``extra_run``, a second run added meanwhile)."""

    def entry(run_id: str, cid: str, title: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "created_at": "2026-10-01T00:00:00+00:00",
            "mode": "standard",
            "genre": "techno",
            "selected_canvas_ids": [],
            "dominant_bpm_clusters": [],
            "dominant_camelot_keys": [],
            "core_track_ids": [],
            "anchor_track_ids": [],
            "opener_candidates": [],
            "closer_candidates": [],
            "concept_title": title,
            "concept_track_ids": [],
            "energy_path": "",
            "mood": "",
            "concepts": [{"concept_id": cid, "title": title, "mood": "dark", "track_ids": ["1"], "arc_type": ""}],
        }

    runs = [entry("h1", concept_id, "Night Drive")]
    if extra_run:
        runs.append(entry("h2", "c2", "Added Meanwhile"))
    return json.dumps({"runs": runs})


def feedback_event(event_id: str, concept_id: str, **fields: object) -> dict[str, object]:
    """One pending event in the shape of GET …/feedback."""
    return {
        "eventId": event_id,
        "runId": "earlier-run",
        "conceptId": concept_id,
        "verdict": "played",
        "rating": None,
        "notes": None,
        "mixLink": None,
        "recordedAt": "2026-10-14T21:30:00Z",
        **fields,
    }


def scenario_with_feedback() -> Scenario:
    """A run whose sync writes: a history with concept c1 and one pending event for it."""
    return Scenario(history=history_with_concept(), events=[feedback_event("e-1", "c1")])


class FakeClock:
    """The protocol's clock in tests: ``now`` is set by hand and ``pause`` advances it instantly."""

    def __init__(self) -> None:
        self.t = 1_000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def pause(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self.t += seconds


_FAKE_ENGINE = Path(__file__).parent / "fake_engine.py"
CREDENTIAL_ENV = {
    "MIXLAB_WORKER_CLIENT_ID": "client-id-0f3e",
    "MIXLAB_WORKER_CLIENT_SECRET": CLIENT_SECRET,
    "MIXLAB_WORKER_TOKEN_ENDPOINT": "",  # set per harness to the stub's /token
    "MIXLAB_WORKER_SCOPE": "api://mixlab-api/.default",
}
# What a developer's shell or .env might hold that no test run may use.
_CLEARED_ENV = (
    "DISCORD_BOT_TOKEN",
    "CATALOG_API_URL",
    "CHANGSTA_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "MIXLAB_STAGE0_MODEL",
    "MIXLAB_STAGE2_MODEL",
    "MIXLAB_STAGE2_EFFORT",
    "MIXLAB_FALLBACKS",
    "MIXLAB_TOKEN_BUDGET",
    "MIXLAB_COLLECTION_PATH",
    "MIXLAB_REPORT_DIR",
    "MIXLAB_DIAGNOSTICS_DIR",
    "MIXLAB_GENRE_CACHE_PATH",
    "MIXLAB_HISTORY_PATH",
    "MIXLAB_KEY_NOTATION",
)


@dataclass
class Harness:
    """One worker execution against a running stub API."""

    api: StubApi
    remote: TenantRemote
    config: tenant_worker.WorkerConfig
    clock: FakeClock
    scratch: Path
    record_path: Path

    @property
    def scenario(self) -> Scenario:
        return self.api.scenario

    def run(self) -> int:
        return tenant_worker.run_once(self.remote, RUN_ID, self.config)

    def engine_record(self) -> dict[str, object] | None:
        """What the fake engine saw (argv, cwd, env), or None when it never ran."""
        if not self.record_path.exists():
            return None
        record = json.loads(self.record_path.read_text())
        assert isinstance(record, dict)
        return record

    def fail_body(self) -> dict[str, object]:
        (call,) = self.api.calls("POST fail")
        body = call.json()
        assert isinstance(body, dict)
        return body


def prepare_env(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    """The worker's environment for a test, the fake clock installed, no .env read by any engine."""
    for name in _CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in CREDENTIAL_ENV.items():
        monkeypatch.setenv(name, value or "unset")
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setattr(tenant_remote, "now", clock.now)
    monkeypatch.setattr(tenant_remote, "pause", clock.pause)


def build_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stack: ExitStack,
    clock: FakeClock,
    *,
    name: str = "run",
    fake_engine: bool = True,
    run_timeout: float = 60.0,
    scenario: Scenario | None = None,
) -> Harness:
    base = tmp_path / name
    scratch = base / "scratch"
    scratch.mkdir(parents=True)
    record_path = base / "engine-record.json"
    api = stack.enter_context(StubApi(scenario or Scenario()))
    monkeypatch.setenv("MIXLAB_WORKER_TOKEN_ENDPOINT", api.token_endpoint)
    if fake_engine:
        wrapper = base / "fake-python"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{_FAKE_ENGINE}" "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(sys, "executable", str(wrapper))
        monkeypatch.setenv("FAKE_ENGINE_RECORD", str(record_path))
    config = TenantConfig(
        api_url=api.url,
        protocol_version=2,
        client_id=CREDENTIAL_ENV["MIXLAB_WORKER_CLIENT_ID"],
        client_secret=CLIENT_SECRET,
        token_endpoint=api.token_endpoint,
        scope=CREDENTIAL_ENV["MIXLAB_WORKER_SCOPE"],
    )
    worker_config = tenant_worker.WorkerConfig(
        xml_path=scratch / "collection.xml", history_path=scratch / "history.json", run_timeout=run_timeout
    )
    assert os.environ["MIXLAB_WORKER_TOKEN_ENDPOINT"] == api.token_endpoint
    return Harness(
        api=api,
        remote=TenantRemote(config),
        config=worker_config,
        clock=clock,
        scratch=scratch,
        record_path=record_path,
    )
