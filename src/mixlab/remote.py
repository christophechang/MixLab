"""Typed client for the MixLab Anywhere API surface (#90 / M2).

See ``docs/architecture/mixlab-anywhere.md`` §4 (endpoints), §5 (contracts), and §6
(how the worker consumes this client). This module is synchronous and bearer-authed —
unlike ``client.py`` (async, catalog API), it is designed to be called from the
worker's blocking poll loop. Every method opens its own ``httpx.Client`` as a context
manager; there is no retry policy here — that is the worker's responsibility (§6).
"""

from __future__ import annotations

import os
import tempfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

_GZIP_WINDOW_BITS = 16 + zlib.MAX_WBITS

_ERROR_BODY_EXCERPT_LEN = 200


class RemoteConfigError(RuntimeError):
    """Raised when required remote configuration is missing."""


class RemoteError(RuntimeError):
    """Raised for any non-2xx response not covered by a more specific error."""

    def __init__(self, method: str, path: str, status: int | None, body: str) -> None:
        self.status = status
        excerpt = body[:_ERROR_BODY_EXCERPT_LEN]
        super().__init__(f"{method} {path} returned {status}: {excerpt}")


class RemoteConflictError(RemoteError):
    """Raised on a 412 Precondition Failed (ETag mismatch)."""


@dataclass(frozen=True)
class RemoteConfig:
    """Connection settings for the MixLab Anywhere API."""

    base_url: str
    api_secret: str
    timeout: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def from_env(cls) -> RemoteConfig:
        base_url = os.environ.get("MIXLAB_API_URL", "")
        if not base_url:
            raise RemoteConfigError("MIXLAB_API_URL is not set")
        api_secret = os.environ.get("MIXLAB_API_SECRET", "")
        if not api_secret:
            raise RemoteConfigError("MIXLAB_API_SECRET is not set")
        return cls(base_url=base_url, api_secret=api_secret)


@dataclass(frozen=True)
class ConceptSummary:
    """Minimal concept identity carried on a run manifest (id + title only)."""

    concept_id: str
    title: str

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> ConceptSummary:
        return cls(
            concept_id=str(data.get("conceptId", "")),
            title=str(data.get("title", "")),
        )


@dataclass(frozen=True)
class RunManifest:
    """``run.json`` (docs/architecture/mixlab-anywhere.md §5.1)."""

    run_id: str
    created_at: str
    status: str
    flags: dict[str, object]
    upload_id: str
    claimed_at: str | None
    worker_id: str | None
    concepts: list[ConceptSummary]

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> RunManifest:
        raw_flags = data.get("flags", {})
        flags: dict[str, object] = raw_flags if isinstance(raw_flags, dict) else {}
        raw_concepts = data.get("concepts", [])
        concept_items = raw_concepts if isinstance(raw_concepts, list) else []
        concepts = [ConceptSummary.from_json(c) for c in concept_items if isinstance(c, dict)]
        return cls(
            run_id=str(data.get("runId", "")),
            created_at=str(data.get("createdAt", "")),
            status=str(data.get("status", "")),
            flags=flags,
            upload_id=str(data.get("uploadId", "")),
            claimed_at=_optional_str(data.get("claimedAt")),
            worker_id=_optional_str(data.get("workerId")),
            concepts=concepts,
        )


@dataclass(frozen=True)
class FeedbackEvent:
    """``feedback/pending.json`` entry (docs/architecture/mixlab-anywhere.md §5.3)."""

    event_id: str
    run_id: str
    concept_id: str
    verdict: str | None
    rating: int | None
    notes: str | None
    published_mix_slug: str | None
    recorded_at: str

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> FeedbackEvent:
        raw_rating = data.get("rating")
        rating = int(raw_rating) if isinstance(raw_rating, (int, float)) else None
        return cls(
            event_id=str(data.get("eventId", "")),
            run_id=str(data.get("runId", "")),
            concept_id=str(data.get("conceptId", "")),
            verdict=_optional_str(data.get("verdict")),
            rating=rating,
            notes=_optional_str(data.get("notes")),
            published_mix_slug=_optional_str(data.get("publishedMixSlug")),
            recorded_at=str(data.get("recordedAt", "")),
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


_CLAIM_PATH = "/api/mixlab/worker/claim"
_HISTORY_PATH = "/api/mixlab/history"
_FEEDBACK_PENDING_PATH = "/api/mixlab/feedback/pending"
_FEEDBACK_ACK_PATH = "/api/mixlab/feedback/ack"


class MixLabRemote:
    """Synchronous, bearer-authed client for ``/api/mixlab/*``."""

    def __init__(self, config: RemoteConfig) -> None:
        self._config = config

    @property
    def base_url(self) -> str:
        """Base URL of the configured API (read-only; used by the worker's startup log)."""
        return self._config.base_url

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.api_secret}"}

    def _url(self, path: str) -> str:
        return f"{self._config.base_url}{path}"

    def _raise_for_status(self, method: str, path: str, response: httpx.Response) -> None:
        if response.status_code == 412:
            raise RemoteConflictError(method, path, response.status_code, response.text)
        if not (200 <= response.status_code < 300):
            raise RemoteError(method, path, response.status_code, response.text)

    def claim(self, worker_id: str) -> RunManifest | None:
        with httpx.Client(timeout=self._config.timeout) as client:
            response = client.post(
                self._url(_CLAIM_PATH),
                headers=self._headers(),
                json={"workerId": worker_id},
            )
        if response.status_code == 204:
            return None
        self._raise_for_status("POST", _CLAIM_PATH, response)
        return RunManifest.from_json(response.json())

    def download_collection(self, upload_id: str, dest: Path) -> Path:
        path = f"/api/mixlab/uploads/{upload_id}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with (
                httpx.Client(timeout=self._config.timeout) as client,
                client.stream("GET", self._url(path), headers=self._headers()) as response,
            ):
                if not (200 <= response.status_code < 300):
                    response.read()
                    self._raise_for_status("GET", path, response)
                decompressor = zlib.decompressobj(_GZIP_WINDOW_BITS)
                with os.fdopen(fd, "wb") as tmp_file:
                    for chunk in response.iter_bytes():
                        tmp_file.write(decompressor.decompress(chunk))
                    tmp_file.write(decompressor.flush())
            os.replace(tmp_path, dest)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        return dest

    def complete(
        self,
        run_id: str,
        *,
        summary_path: Path,
        report_path: Path,
        export_path: Path | None,
    ) -> None:
        path = f"/api/mixlab/runs/{run_id}/complete"
        summary_bytes = summary_path.read_bytes()
        report_bytes = report_path.read_bytes()
        files: dict[str, tuple[str, bytes, str]] = {
            "summary": (summary_path.name, summary_bytes, "application/json"),
            "report": (report_path.name, report_bytes, "text/html"),
        }
        if export_path is not None:
            export_bytes = export_path.read_bytes()
            files["export"] = (export_path.name, export_bytes, "application/xml")
        with httpx.Client(timeout=self._config.timeout) as client:
            response = client.post(self._url(path), headers=self._headers(), files=files)
        self._raise_for_status("POST", path, response)

    def fail(self, run_id: str, *, error: str, log_tail: str) -> None:
        path = f"/api/mixlab/runs/{run_id}/fail"
        with httpx.Client(timeout=self._config.timeout) as client:
            response = client.post(
                self._url(path),
                headers=self._headers(),
                json={"error": error, "logTail": log_tail},
            )
        self._raise_for_status("POST", path, response)

    def get_history(self) -> tuple[str | None, str]:
        with httpx.Client(timeout=self._config.timeout) as client:
            response = client.get(self._url(_HISTORY_PATH), headers=self._headers())
        if response.status_code == 404:
            return None, ""
        self._raise_for_status("GET", _HISTORY_PATH, response)
        return response.headers.get("ETag"), response.text

    def put_history(self, body: str, *, etag: str | None) -> str:
        headers = self._headers()
        if etag is not None:
            headers["If-Match"] = etag
        with httpx.Client(timeout=self._config.timeout) as client:
            response = client.put(
                self._url(_HISTORY_PATH),
                headers=headers,
                content=body.encode("utf-8"),
            )
        self._raise_for_status("PUT", _HISTORY_PATH, response)
        return str(response.headers.get("ETag", ""))

    def pending_feedback(self) -> list[FeedbackEvent]:
        with httpx.Client(timeout=self._config.timeout) as client:
            response = client.get(self._url(_FEEDBACK_PENDING_PATH), headers=self._headers())
        self._raise_for_status("GET", _FEEDBACK_PENDING_PATH, response)
        data: list[object] = response.json()
        return [FeedbackEvent.from_json(item) for item in data if isinstance(item, dict)]

    def ack_feedback(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        with httpx.Client(timeout=self._config.timeout) as client:
            response = client.post(
                self._url(_FEEDBACK_ACK_PATH),
                headers=self._headers(),
                json={"eventIds": event_ids},
            )
        self._raise_for_status("POST", _FEEDBACK_ACK_PATH, response)
