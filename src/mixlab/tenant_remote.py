"""Client for the multi-tenant worker protocol (M2b spec §4; ``protocol.md`` in mixlab-multi-tenant).

The tenant counterpart of ``remote.py``, which stays the MixLab Anywhere client, untouched
(M2b-D1). Synchronous, one ``httpx.Client`` per request as ``remote.py`` does. Every request
takes its timeout from a :class:`Deadline`, ``min(30 s, the time left)``, so no wait outlives
the window it belongs to (M2b-D32). The retry policy (three attempts before the engine, the
back-off after it) is the worker's; the one rule kept here is ``409 busy``: one repeat after
1 s, then a :class:`TransportError` (R-AW).

The run token and ``claimId`` live only on the :class:`Claim` object; nothing here puts them,
or the worker credential, in the environment or a log line.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar, cast
from urllib.parse import quote, unquote, urlsplit

import httpx

REQUEST_TIMEOUT_SECS = 30.0
BUSY_REPEAT_SECS = 1.0
# preflight.json maxDecompressedBytes, 96 MiB (R-Q)
MAX_DECOMPRESSED_BYTES = 100_663_296

_GZIP_WINDOW_BITS = 16 + zlib.MAX_WBITS
_STALE_CODES = frozenset({"stale_claim", "not_running", "not_claimable"})

_T = TypeVar("_T")

ProgressStage = Literal["synced", "downloaded", "engine-started"]
UploadFormat = Literal["rekordbox", "traktor"]

_PART_TYPES = {
    "summary.json": "application/json",
    "report.html": "text/html",
    "export.xml": "application/xml",
}


def now() -> float:
    """The protocol's clock, monotonic seconds; tests replace it with a fake clock."""
    return time.monotonic()


def pause(seconds: float) -> None:
    """Every wait the protocol makes; tests replace it with the fake clock's sleep."""
    if seconds > 0:
        time.sleep(seconds)


class Deadline:
    """A monotonic deadline: ``seconds`` from ``start`` (default: now)."""

    def __init__(self, seconds: float, *, start: float | None = None) -> None:
        self._end = (now() if start is None else start) + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - now())

    def timeout(self, cap: float = REQUEST_TIMEOUT_SECS) -> float:
        """A request's timeout: the cap, or the time left when that is less."""
        return min(cap, self.remaining())

    def expired(self) -> bool:
        return now() >= self._end

    def shortened(self, seconds: float) -> Deadline:
        """The same deadline, ``seconds`` earlier: what is left once a reserve is kept."""
        earlier = Deadline(0.0)
        earlier._end = self._end - seconds
        return earlier


class TenantConfigError(RuntimeError):
    """A worker variable is missing or invalid; the message names the variable, never its value."""


class ProtocolError(RuntimeError):
    """A permanent refusal for this execution: ``400``, ``401``, ``403``, ``413``, ``426`` or another status."""

    def __init__(self, status: int, error: str, detail: str = "") -> None:
        self.status = status
        self.error = error
        super().__init__(f"{status} {error}{': ' + detail if detail else ''}")


class StaleClaim(ProtocolError):  # noqa: N818 — the plan's name (M2b plan Task 10)
    """``409`` ``stale_claim``, ``not_running`` or ``not_claimable``: another execution or the janitor owns the run."""


class EtagMismatch(ProtocolError):  # noqa: N818 — the plan's name (M2b plan Task 10)
    """``409 etag_mismatch`` on the history ``PUT``: the history changed since the ``GET``."""


class TransportError(RuntimeError):
    """A ``5xx``, no answer, a second ``409 busy``, or the deadline passing: transient, retried by the worker."""


class UploadTooLarge(RuntimeError):  # noqa: N818 — the plan's name (M2b plan Task 10)
    """The gunzipped upload passed :data:`MAX_DECOMPRESSED_BYTES`."""


class UploadUnreadable(RuntimeError):  # noqa: N818 — the plan's name (M2b plan Task 10)
    """The upload is not one complete gzip member: not gzip, truncated, or followed by more bytes."""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise TenantConfigError(f"{name} is not set")
    return value


@dataclass(frozen=True)
class TenantConfig:
    """The worker's protocol settings, from the environment (M2b spec §4.1, M2b-D3)."""

    api_url: str
    protocol_version: int
    client_id: str
    client_secret: str = field(repr=False)
    token_endpoint: str
    scope: str
    timeout: float = REQUEST_TIMEOUT_SECS

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_url", self.api_url.rstrip("/"))

    @classmethod
    def from_env(cls) -> TenantConfig:
        version = _required("MIXLAB_PROTOCOL_VERSION")
        if not version.isascii() or not version.isdigit() or int(version) < 1:
            raise TenantConfigError("MIXLAB_PROTOCOL_VERSION is not a positive integer")
        return cls(
            api_url=_required("MIXLAB_API_URL"),
            protocol_version=int(version),
            client_id=_required("MIXLAB_WORKER_CLIENT_ID"),
            client_secret=_required("MIXLAB_WORKER_CLIENT_SECRET"),
            token_endpoint=_required("MIXLAB_WORKER_TOKEN_ENDPOINT"),
            scope=_required("MIXLAB_WORKER_SCOPE"),
        )


def _string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


@dataclass(frozen=True)
class TenantManifest:
    """The claim's ``manifest`` (protocol.md, ``POST …/claim``)."""

    run_id: str
    upload_id: str
    upload_format: UploadFormat
    created_at: str
    flags: dict[str, object]

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> TenantManifest:
        upload_format = data.get("uploadFormat")
        if upload_format not in ("rekordbox", "traktor"):
            raise ProtocolError(200, "bad_claim", "manifest.uploadFormat is neither rekordbox nor traktor")
        raw_flags = data.get("flags")
        return cls(
            run_id=_string(data, "runId"),
            upload_id=_string(data, "uploadId"),
            upload_format=upload_format,
            created_at=_string(data, "createdAt"),
            flags=cast("dict[str, object]", raw_flags) if isinstance(raw_flags, dict) else {},
        )


@dataclass(frozen=True)
class Claim:
    """A claim's ``200`` for ``run_id``: the run token, ``claimId`` and manifest, and when it arrived (monotonic).

    Every later URL is built from ``run_id``, the run the worker claimed, never from the manifest.
    """

    run_id: str
    claim_id: str
    run_token: str
    lease_expires_at: str
    diagnostics_upload_url: str
    manifest: TenantManifest
    claimed_at: float

    def __repr__(self) -> str:  # the token and the claim id stay out of any repr or traceback
        return f"Claim(run_id={self.run_id!r})"

    @classmethod
    def from_json(cls, data: Mapping[str, object], *, run_id: str, claimed_at: float) -> Claim:
        claim_id = _string(data, "claimId")
        run_token = _string(data, "runToken")
        manifest = data.get("manifest")
        if not claim_id or not run_token or not isinstance(manifest, dict):
            raise ProtocolError(200, "bad_claim", "claimId, runToken or manifest missing")
        parsed = TenantManifest.from_json(cast("dict[str, object]", manifest))
        if parsed.run_id != run_id:
            raise ProtocolError(200, "bad_claim", "manifest.runId is not the run claimed")
        return cls(
            run_id=run_id,
            claim_id=claim_id,
            run_token=run_token,
            lease_expires_at=_string(data, "leaseExpiresAt"),
            diagnostics_upload_url=_string(data, "diagnosticsUploadUrl"),
            manifest=parsed,
            claimed_at=claimed_at,
        )


@dataclass(frozen=True)
class TenantFeedbackEvent:
    """One pending feedback event (protocol.md, ``GET …/feedback``)."""

    event_id: str
    run_id: str
    concept_id: str
    verdict: str | None
    rating: int | None
    notes: str | None
    mix_link: str | None
    recorded_at: str

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> TenantFeedbackEvent:
        raw_rating = data.get("rating")
        rating = int(raw_rating) if isinstance(raw_rating, (int, float)) and not isinstance(raw_rating, bool) else None
        return cls(
            event_id=_string(data, "eventId"),
            run_id=_string(data, "runId"),
            concept_id=_string(data, "conceptId"),
            verdict=_optional_string(data, "verdict"),
            rating=rating,
            notes=_optional_string(data, "notes"),
            mix_link=_optional_string(data, "mixLink"),
            recorded_at=_string(data, "recordedAt"),
        )


def _error_code(response: httpx.Response) -> str:
    """The refusal body's ``error``, or empty when the body is not the protocol's."""
    try:
        body = response.json()
    except ValueError:
        return ""
    return _string(body, "error") if isinstance(body, dict) else ""


def _raise_for(response: httpx.Response, method: str, route: str) -> None:
    """Map a non-2xx answer onto the protocol's errors (protocol.md, the status table)."""
    status = response.status_code
    if 200 <= status < 300:
        return
    error = _error_code(response)
    if status >= 500:
        raise TransportError(f"{method} {route} answered {status}")
    if status == 409 and error in _STALE_CODES:
        raise StaleClaim(status, error)
    if status == 409 and error == "etag_mismatch":
        raise EtagMismatch(status, error)
    raise ProtocolError(status, error or "unexpected", f"{method} {route}")


def diagnostics_ref(url: str, run_id: str) -> str | None:
    """The blob's path under ``tenants/{tenantId}/runs/{runId}/``, read from the URL's path (never its SAS)."""
    path = unquote(urlsplit(url).path)
    marker = f"/runs/{run_id}/"
    index = path.find(marker)
    if not run_id or index < 0 or not path[index + len(marker) :]:
        return None
    return path[index + len(marker) :]


class _Busy(Exception):  # noqa: N818 — internal signal, not an error the caller sees
    """Internal: a streamed request answered ``409 busy``."""


def _within(deadline: Deadline, label: str, call: Callable[[], _T]) -> _T:
    """Run one whole request inside the deadline, its answer read to the end.

    ``httpx``'s timeouts bound each read, not the request, so an answer trickled a byte at a
    time could outlive its window. The request runs on a daemon thread that is abandoned at the
    deadline with a :class:`TransportError`, and an answer that arrives after the deadline is
    refused the same way (M2b-D32). An abandoned request dies with the process, which exits or
    moves to its next window at once.
    """
    if deadline.expired():
        raise TransportError(f"{label}: deadline passed")
    outcome: list[_T] = []
    failure: list[BaseException] = []

    def target() -> None:
        try:
            outcome.append(call())
        except BaseException as exc:  # handed to the caller's thread below
            failure.append(exc)

    thread = threading.Thread(target=target, name=f"tenant-{label}", daemon=True)
    thread.start()
    thread.join(deadline.remaining())
    if thread.is_alive():
        raise TransportError(f"{label}: no whole answer inside the deadline")
    if failure:
        raise failure[0]
    if deadline.expired():
        raise TransportError(f"{label}: answered after the deadline")
    return outcome[0]


class TenantRemote:
    """The worker's side of ``protocol.md``: the token, the claim and the eight run-token routes."""

    def __init__(self, config: TenantConfig) -> None:
        self._config = config

    @property
    def config(self) -> TenantConfig:
        return self._config

    def _url(self, run_id: str, route: str) -> str:
        return f"{self._config.api_url}/api/worker/runs/{quote(run_id, safe='')}/{route}"

    @staticmethod
    def _run_headers(claim: Claim) -> dict[str, str]:
        return {"Authorization": f"Bearer {claim.run_token}", "X-MixLab-Claim": claim.claim_id}

    def _send(
        self,
        method: str,
        url: str,
        route: str,
        deadline: Deadline,
        *,
        headers: dict[str, str],
        json_body: object = None,
        content: bytes | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> httpx.Response:
        """One request, repeated once after 1 s on ``409 busy``; no request starts once the deadline has passed."""
        label = f"{method} {route}"

        def request() -> httpx.Response:
            try:
                with httpx.Client(timeout=deadline.timeout(self._config.timeout)) as client:
                    return client.request(method, url, headers=headers, json=json_body, content=content, files=files)
            except httpx.HTTPError as exc:
                raise TransportError(f"{label}: {type(exc).__name__}") from exc

        for attempt in (1, 2):
            response = _within(deadline, label, request)
            if response.status_code == 409 and _error_code(response) == "busy":
                if attempt == 1:
                    pause(min(BUSY_REPEAT_SECS, deadline.remaining()))
                    continue
                raise TransportError(f"{method} {route}: 409 busy twice")
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def acquire_worker_token(self, deadline: Deadline) -> str:
        """The client-credentials token for the claim (M2b spec §4.2): one form ``POST``."""
        form = {
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "scope": self._config.scope,
            "grant_type": "client_credentials",
        }

        def request() -> httpx.Response:
            try:
                with httpx.Client(timeout=deadline.timeout(self._config.timeout)) as client:
                    return client.post(self._config.token_endpoint, data=form)
            except httpx.HTTPError as exc:
                raise TransportError(f"token: {type(exc).__name__}") from exc

        response = _within(deadline, "token", request)
        if response.status_code != 200:
            raise ProtocolError(response.status_code, "token_refused")
        try:
            body = response.json()
        except ValueError:
            body = None
        token = _string(body, "access_token") if isinstance(body, dict) else ""
        if not token:
            raise ProtocolError(200, "token_refused", "no access_token in the answer")
        return token

    def claim(self, run_id: str, worker_token: str, deadline: Deadline) -> Claim:
        response = self._send(
            "POST",
            self._url(run_id, "claim"),
            "claim",
            deadline,
            headers={"Authorization": f"Bearer {worker_token}"},
            json_body={"protocolVersion": self._config.protocol_version},
        )
        _raise_for(response, "POST", "claim")
        try:
            body = response.json()
        except ValueError as exc:
            raise ProtocolError(200, "bad_claim", "the answer is not JSON") from exc
        if not isinstance(body, dict):
            raise ProtocolError(200, "bad_claim", "the answer is not an object")
        return Claim.from_json(body, run_id=run_id, claimed_at=now())

    def get_history(self, claim: Claim, deadline: Deadline) -> tuple[str | None, str | None]:
        """``(etag, body)`` on ``200``; ``(None, None)`` on ``204``, the tenant having no history yet."""
        response = self._send(
            "GET", self._url(claim.run_id, "history"), "history", deadline, headers=self._run_headers(claim)
        )
        _raise_for(response, "GET", "history")
        if response.status_code == 204:
            return None, None
        return response.headers.get("ETag"), response.text

    def put_history(self, claim: Claim, body: str, *, etag: str | None, deadline: Deadline) -> str:
        """Write the history under ``If-Match`` (after a ``200``) or ``If-None-Match: *`` (after a ``204``)."""
        headers = {**self._run_headers(claim), "Content-Type": "application/json"}
        if etag is None:
            headers["If-None-Match"] = "*"
        else:
            headers["If-Match"] = etag
        response = self._send(
            "PUT",
            self._url(claim.run_id, "history"),
            "history",
            deadline,
            headers=headers,
            content=body.encode("utf-8"),
        )
        _raise_for(response, "PUT", "history")
        return str(response.headers.get("ETag", ""))

    def get_feedback(self, claim: Claim, deadline: Deadline) -> list[TenantFeedbackEvent]:
        response = self._send(
            "GET", self._url(claim.run_id, "feedback"), "feedback", deadline, headers=self._run_headers(claim)
        )
        _raise_for(response, "GET", "feedback")
        try:
            body = response.json()
        except ValueError as exc:
            raise ProtocolError(response.status_code, "bad_feedback", "the answer is not JSON") from exc
        events = body.get("events") if isinstance(body, dict) else None
        if not isinstance(events, list):
            raise ProtocolError(response.status_code, "bad_feedback", "no events array")
        return [TenantFeedbackEvent.from_json(item) for item in events if isinstance(item, dict)]

    def ack_feedback(self, claim: Claim, event_ids: list[str], deadline: Deadline) -> None:
        response = self._send(
            "POST",
            self._url(claim.run_id, "feedback/ack"),
            "feedback/ack",
            deadline,
            headers=self._run_headers(claim),
            json_body={"eventIds": event_ids},
        )
        _raise_for(response, "POST", "feedback/ack")

    def download_upload(
        self, claim: Claim, dest: Path, deadline: Deadline, *, max_decompressed: int = MAX_DECOMPRESSED_BYTES
    ) -> None:
        """Stream the upload, gunzip it beside ``dest`` and move it into place (M2b spec §4.3 step 3, M2b-D33).

        The inflater is never asked for more than ``max_decompressed + 1`` bytes in all: the byte
        past the cap raises :class:`UploadTooLarge`. At the end the gzip member must be complete
        with nothing after it, else :class:`UploadUnreadable`. A break in the stream, or the
        deadline passing during it, is a :class:`TransportError`; the partial file is removed. A
        ``409 busy`` is repeated once after 1 s, as on every route.
        """
        for attempt in (1, 2):
            try:
                _within(deadline, "GET upload", lambda: self._stream_upload(claim, dest, deadline, max_decompressed))
                return
            except _Busy:
                if attempt == 2:
                    raise TransportError("GET upload: 409 busy twice") from None
                pause(min(BUSY_REPEAT_SECS, deadline.remaining()))

    def _stream_upload(self, claim: Claim, dest: Path, deadline: Deadline, max_decompressed: int) -> None:
        route = "upload"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with (
                os.fdopen(fd, "wb") as tmp_file,
                httpx.Client(timeout=deadline.timeout(self._config.timeout)) as client,
                client.stream("GET", self._url(claim.run_id, route), headers=self._run_headers(claim)) as response,
            ):
                if not (200 <= response.status_code < 300):
                    response.read()
                    if response.status_code == 409 and _error_code(response) == "busy":
                        raise _Busy
                    _raise_for(response, "GET", route)
                decompressor = zlib.decompressobj(_GZIP_WINDOW_BITS)
                written = 0
                for chunk in response.iter_raw():  # raw: a Content-Encoding is never undone for us
                    if deadline.expired():
                        raise TransportError(f"GET {route}: deadline passed during the stream")
                    data = chunk
                    while data:
                        if decompressor.eof:
                            raise UploadUnreadable("bytes follow the end of the gzip member")
                        out = decompressor.decompress(data, max_decompressed - written + 1)
                        written += len(out)
                        if written > max_decompressed:
                            raise UploadTooLarge(f"the upload inflates past {max_decompressed} bytes")
                        tmp_file.write(out)
                        data = decompressor.unconsumed_tail
                    if decompressor.unused_data:
                        raise UploadUnreadable("bytes follow the end of the gzip member")
                if not decompressor.eof:
                    raise UploadUnreadable("the gzip stream ends before its member does")
            os.replace(tmp_path, dest)
        except httpx.HTTPError as exc:
            tmp_path.unlink(missing_ok=True)
            raise TransportError(f"GET {route}: {type(exc).__name__}") from exc
        except zlib.error as exc:
            tmp_path.unlink(missing_ok=True)
            raise UploadUnreadable(f"not a gzip stream: {exc}") from exc
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def progress(self, claim: Claim, stage: ProgressStage, deadline: Deadline) -> None:
        """Record a stage; only ``204`` counts (``engine-started``'s is load-bearing, M2b spec §4.4)."""
        response = self._send(
            "POST",
            self._url(claim.run_id, "progress"),
            "progress",
            deadline,
            headers=self._run_headers(claim),
            json_body={"stage": stage},
        )
        _raise_for(response, "POST", "progress")
        if response.status_code != 204:
            raise ProtocolError(response.status_code, "unexpected", "POST progress answered other than 204")

    def complete(self, claim: Claim, parts: dict[str, Path], deadline: Deadline) -> None:
        """One multipart ``POST`` with a part per file, named by file (M2b spec §4.6)."""
        files = {name: (name, path.read_bytes(), _PART_TYPES[name]) for name, path in parts.items()}
        response = self._send(
            "POST",
            self._url(claim.run_id, "complete"),
            "complete",
            deadline,
            headers=self._run_headers(claim),
            files=files,
        )
        _raise_for(response, "POST", "complete")

    def fail(self, claim: Claim, *, error_code: str, diagnostics_ref: str | None, deadline: Deadline) -> None:
        response = self._send(
            "POST",
            self._url(claim.run_id, "fail"),
            "fail",
            deadline,
            headers=self._run_headers(claim),
            json_body={"errorCode": error_code, "diagnosticsRef": diagnostics_ref},
        )
        _raise_for(response, "POST", "fail")

    def put_diagnostics(self, claim: Claim, text: str, deadline: Deadline) -> str | None:
        """Write the (already scrubbed) text to the claim's blob URL: one attempt and one retry.

        Returns the blob's ``diagnosticsRef``, or ``None`` when the write failed, the deadline
        passed, or the URL does not name the run's blob path.
        """
        ref = diagnostics_ref(claim.diagnostics_upload_url, claim.run_id)
        if ref is None:
            return None
        headers = {"x-ms-blob-type": "BlockBlob", "Content-Type": "text/plain; charset=utf-8"}
        body = text.encode("utf-8")
        for _attempt in (1, 2):

            def request() -> httpx.Response:
                try:
                    with httpx.Client(timeout=deadline.timeout(self._config.timeout)) as client:
                        return client.put(claim.diagnostics_upload_url, headers=headers, content=body)
                except httpx.HTTPError as exc:
                    raise TransportError(f"PUT diagnostics: {type(exc).__name__}") from exc

            try:
                response = _within(deadline, "PUT diagnostics", request)
            except TransportError:
                continue
            if 200 <= response.status_code < 300:
                return ref
        return None
