"""One tenant run per execution: claim, sync, download, spawn, ``complete`` or ``fail`` (M2b spec §4).

The tenant counterpart of ``worker.py``, which stays the MixLab Anywhere loop, untouched
(M2b-D1); the two share only :func:`mixlab.worker.build_argv`. The protocol is ``protocol.md``
in mixlab-multi-tenant, spoken through :mod:`mixlab.tenant_remote`.

Three windows bound every wait (M2b-D32): the token request and the claim share 60 s from
the start; one pre-engine deadline of 180 s from the claim's ``200`` bounds the sync, the
download and ``engine-started``; one terminal deadline of 180 s from the engine's end bounds
the diagnostics write and ``complete`` or ``fail``. ``fail`` never follows a ``complete``
attempt. The engine runs as a subprocess in the scratch directory, with the worker's
credentials removed from its environment (M2b-D5); its output reaches nothing but the
scrubbed diagnostics blob of a failed run (R-FI).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from mixlab import tenant_remote
from mixlab.history import apply_feedback_verdict, load_history, save_history
from mixlab.tenant_remote import (
    Claim,
    Deadline,
    EtagMismatch,
    ProtocolError,
    StaleClaim,
    TenantFeedbackEvent,
    TenantRemote,
    TransportError,
    UploadTooLarge,
    UploadUnreadable,
)
from mixlab.worker import WorkerFlagError, build_argv

PRE_CLAIM_BOUND_SECS = 60
PRE_ENGINE_DEADLINE_SECS = 180
TERMINAL_WINDOW_SECS = 180
FAIL_RESERVE_SECS = 60
MAX_PART_BYTES = 4_194_304
MIN_SECRET_LEN = 8

# Before the engine: three attempts, 1 s then 2 s apart. After it: 1, 2, 4, 8, 16, then 30 s apart.
_PRE_ENGINE_BACKOFFS = (1.0, 2.0)
_TERMINAL_BACKOFFS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_TAIL_LEN = 4000
_REDACTED = "[redacted]"
_EMPTY_HISTORY = '{"runs": []}'

# The worker's own credential variables: never on the engine subprocess (M2b-D5). A new worker
# secret joins this list.
_CREDENTIAL_VARS = (
    "MIXLAB_WORKER_CLIENT_ID",
    "MIXLAB_WORKER_CLIENT_SECRET",
    "MIXLAB_WORKER_TOKEN_ENDPOINT",
    "MIXLAB_WORKER_SCOPE",
)

# The engine's stdout lines naming its artifacts (worker.py's markers), the part each one is, and the
# scratch subdirectory the file must resolve into.
_PART_LINES = (
    ("Run summary:", "summary.json", "reports"),
    ("HTML report:", "report.html", "reports"),
    ("Exported:", "export.xml", "export"),
)

_ERROR_CODES = {2: "no_shortlist", 3: "token_budget", 4: "provider_cap", 5: "refusal", 6: "provider_unavailable"}

_T = TypeVar("_T")


def _log(message: str) -> None:
    """A worker log line on stderr. Ids, stages, codes and sizes only: nothing from the DJ's file."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tenant-worker: {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class WorkerConfig:
    """The tenant worker's paths and engine timeout (M2b spec §4.1, M2b-D4)."""

    xml_path: Path
    history_path: Path
    run_timeout: float

    @property
    def scratch(self) -> Path:
        return self.xml_path.parent

    @classmethod
    def from_env(cls) -> WorkerConfig:
        """``MIXLAB_WORKER_XML_PATH`` (required; its directory is the scratch), ``MIXLAB_HISTORY_PATH``
        (default ``<scratch>/concept-history.json``) and ``MIXLAB_WORKER_RUN_TIMEOUT`` (default 1200).
        """
        raw_xml = os.environ.get("MIXLAB_WORKER_XML_PATH", "").strip()
        if not raw_xml:
            raise tenant_remote.TenantConfigError("MIXLAB_WORKER_XML_PATH is not set")
        xml_path = Path(raw_xml).absolute()
        raw_history = os.environ.get("MIXLAB_HISTORY_PATH", "").strip()
        history_path = Path(raw_history).absolute() if raw_history else xml_path.parent / "concept-history.json"
        try:
            run_timeout = float(os.environ.get("MIXLAB_WORKER_RUN_TIMEOUT", "1200"))
        except ValueError as exc:
            raise tenant_remote.TenantConfigError("MIXLAB_WORKER_RUN_TIMEOUT is not a number") from exc
        return cls(xml_path=xml_path, history_path=history_path, run_timeout=run_timeout)


@dataclass(frozen=True)
class RunPaths:
    """Everything one run writes, under the scratch (M2b-D4)."""

    scratch: Path
    collection: Path
    history: Path
    reports: Path
    diagnostics: Path
    export: Path
    genre_cache: Path

    @classmethod
    def for_config(cls, config: WorkerConfig) -> RunPaths:
        scratch = config.scratch
        return cls(
            scratch=scratch,
            collection=config.xml_path,
            history=config.history_path,
            reports=scratch / "reports",
            diagnostics=scratch / "diagnostics",
            export=scratch / "export",
            genre_cache=scratch / "genres.json",
        )


class _PreEngineRefusalError(Exception):
    """A refusal after the claim and before the engine that is reported with ``fail`` (M2b spec §4.8)."""

    def __init__(self, error_code: str, reason: str) -> None:
        self.error_code = error_code
        self.reason = reason
        super().__init__(f"{error_code}: {reason}")


def scrub(text: str, secrets: Sequence[str]) -> str:
    """Replace every secret by ``[redacted]``; an empty value or one under 8 characters is skipped (M2b-D7)."""
    usable = sorted({s for s in secrets if len(s) >= MIN_SECRET_LEN}, key=len, reverse=True)
    for secret in usable:
        text = text.replace(secret, _REDACTED)
    return text


def exit_code_to_error_code(returncode: int | None) -> str:
    """The ``fail`` code for an engine exit (M2b spec §4.8); ``None`` is the subprocess timeout."""
    if returncode is None:
        return "engine_timeout"
    return _ERROR_CODES.get(returncode, "engine_failed")


def existing_parts(stdout: str, scratch: Path, export: bool) -> dict[str, Path]:
    """The ``complete`` parts whose line the engine printed and whose file exists (M2b spec §4.6, M2b-D31).

    The engine's stdout also carries text from the DJ's file and the model (track names in
    warnings, the report itself), so a line that looks like a marker proves nothing: a part is
    taken only from a line whose path resolves, symlinks followed, to a file inside its own
    scratch subdirectory (``reports`` or ``export``). The first such line wins.
    """
    parts: dict[str, Path] = {}
    for prefix, name, subdir in _PART_LINES:
        if name == "export.xml" and not export:
            continue
        allowed = (scratch / subdir).resolve()
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            path = Path(stripped[len(prefix) :].strip())
            if not path.is_absolute():
                path = scratch / path
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved.is_relative_to(allowed) and resolved.is_file():
                parts[name] = resolved
                break
    return parts


def _subprocess_env(run_paths: RunPaths) -> dict[str, str]:
    """The worker's environment minus its credentials, plus this run's five paths (M2b spec §4.5)."""
    env = {key: value for key, value in os.environ.items() if key not in _CREDENTIAL_VARS}
    env.update(
        {
            "MIXLAB_COLLECTION_PATH": str(run_paths.collection),
            "MIXLAB_HISTORY_PATH": str(run_paths.history),
            "MIXLAB_REPORT_DIR": str(run_paths.reports),
            "MIXLAB_DIAGNOSTICS_DIR": str(run_paths.diagnostics),
            "MIXLAB_GENRE_CACHE_PATH": str(run_paths.genre_cache),
        }
    )
    return env


def _pre_engine(call: Callable[[], _T], deadline: Deadline) -> _T:
    """Up to three attempts on a transport error, 1 s then 2 s apart, every sleep inside the deadline."""
    attempt = 0
    while True:
        try:
            return call()
        except TransportError:
            if attempt >= len(_PRE_ENGINE_BACKOFFS) or deadline.expired():
                raise
            tenant_remote.pause(min(_PRE_ENGINE_BACKOFFS[attempt], deadline.remaining()))
            attempt += 1


def _terminal(call: Callable[[], _T], deadline: Deadline) -> _T:
    """Back off on a transport error (1, 2, 4, 8, 16, 30, 30 … s) until the terminal deadline passes."""
    attempt = 0
    while True:
        try:
            return call()
        except TransportError:
            if deadline.expired():
                raise
            backoff = _TERMINAL_BACKOFFS[min(attempt, len(_TERMINAL_BACKOFFS) - 1)]
            tenant_remote.pause(min(backoff, deadline.remaining()))
            attempt += 1


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _compose_notes(event: TenantFeedbackEvent) -> str:
    """Fold rating, mix link and notes into one text as the Anywhere sync does, ``mixLink`` in the slug's slot."""
    parts: list[str] = []
    if event.rating is not None:
        parts.append(f"rating={event.rating}")
    if event.mix_link:
        parts.append(f"mix={event.mix_link}")
    if event.notes:
        parts.append(event.notes)
    return "; ".join(parts)


def _write_history(path: Path, body: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if body is not None else _EMPTY_HISTORY, encoding="utf-8")


def _apply_events(path: Path, events: list[TenantFeedbackEvent]) -> int:
    """Apply every event to the history file; returns how many matched a concept."""
    history = load_history(path)
    applied = 0
    for event in events:
        if apply_feedback_verdict(history, event.concept_id, event.verdict, _compose_notes(event)) is None:
            _log(f"feedback event {event.event_id} matches no concept; acknowledged unapplied")
        else:
            applied += 1
    if applied:
        save_history(history, path)
    return applied


def _sync(remote: TenantRemote, claim: Claim, history_path: Path, deadline: Deadline) -> None:
    """Step 2: the history down, the pending feedback applied, the history up, the events acknowledged."""
    etag, body = _pre_engine(lambda: remote.get_history(claim, deadline), deadline)
    _write_history(history_path, body)
    events = _pre_engine(lambda: remote.get_feedback(claim, deadline), deadline)
    if not events:
        return
    if _apply_events(history_path, events):

        def put(tag: str | None) -> str:
            text = history_path.read_text(encoding="utf-8")
            return _pre_engine(lambda: remote.put_history(claim, text, etag=tag, deadline=deadline), deadline)

        try:
            put(etag)
        except EtagMismatch:
            etag, body = _pre_engine(lambda: remote.get_history(claim, deadline), deadline)
            _write_history(history_path, body)
            if _apply_events(history_path, events):  # the run an event named may have gone meanwhile (R-K)
                try:
                    put(etag)
                except EtagMismatch as exc:
                    raise TransportError("PUT history: etag_mismatch twice") from exc
    _pre_engine(lambda: remote.ack_feedback(claim, [e.event_id for e in events], deadline), deadline)
    _log(f"synced {len(events)} feedback event(s)")


def _prepare(remote: TenantRemote, claim: Claim, paths: RunPaths, deadline: Deadline) -> list[str]:
    """Steps 2 to 5: sync, download, map the flags and record ``engine-started``; returns the engine's argv."""
    _sync(remote, claim, paths.history, deadline)
    _pre_engine(lambda: remote.progress(claim, "synced", deadline), deadline)
    try:
        _pre_engine(lambda: remote.download_upload(claim, paths.collection, deadline), deadline)
    except UploadTooLarge as exc:
        raise _PreEngineRefusalError("upload_too_large", str(exc)) from exc
    except UploadUnreadable as exc:
        raise _PreEngineRefusalError("upload_unreadable", str(exc)) from exc
    _pre_engine(lambda: remote.progress(claim, "downloaded", deadline), deadline)
    try:
        argv = build_argv(claim.manifest.flags)
    except WorkerFlagError as exc:
        raise _PreEngineRefusalError("invalid_flags", str(exc)) from exc
    for directory in (paths.reports, paths.diagnostics, paths.export):
        directory.mkdir(parents=True, exist_ok=True)
    if claim.manifest.upload_format == "rekordbox":
        argv = [*argv, "--export", str(paths.export)]
    _pre_engine(lambda: remote.progress(claim, "engine-started", deadline), deadline)
    if deadline.expired():
        raise TransportError("engine-started answered after the pre-engine deadline")
    return argv


def _tail(text: str, secrets: Sequence[str]) -> str:
    """The last 4,000 characters, scrubbed first so a secret cut by the slice cannot keep its end."""
    return scrub(text, secrets)[-_TAIL_LEN:]


def _diagnostics_text(stdout: str, stderr: str, diagnostics_dir: Path, secrets: Sequence[str]) -> str:
    """The last 4,000 characters of stderr then stdout, then every truncation dump the engine wrote (R-AY)."""
    sections = [_tail(stderr + stdout, secrets)]
    if diagnostics_dir.is_dir():
        for dump in sorted(diagnostics_dir.glob("truncated-*.txt")):
            sections.append(f"--- {dump.name} ---\n{dump.read_text(encoding='utf-8', errors='replace')}")
    return "\n".join(sections)


def _fail(
    remote: TenantRemote, claim: Claim, error_code: str, text: str, secrets: Sequence[str], terminal: Deadline
) -> int:
    """Write the scrubbed diagnostics (while more than 60 s remain), then ``fail``, inside the terminal window."""
    ref: str | None = None
    diagnostics_deadline = terminal.shortened(FAIL_RESERVE_SECS)
    if not diagnostics_deadline.expired():
        ref = remote.put_diagnostics(claim, scrub(text, secrets), diagnostics_deadline)
    if ref is None:
        _log("diagnostics not written; fail carries no diagnosticsRef")
    try:
        _terminal(
            lambda: remote.fail(claim, error_code=error_code, diagnostics_ref=ref, deadline=terminal),
            terminal,
        )
    except StaleClaim as exc:
        _log(f"fail answered 409 {exc.error}; the run is another's")
        return 0
    except (ProtocolError, TransportError) as exc:
        _log(f"fail {error_code} not recorded: {exc}")
        return 1
    _log(f"failed run {claim.run_id}: {error_code}")
    return 0


def _complete(
    remote: TenantRemote,
    claim: Claim,
    stdout: str,
    stderr: str,
    paths: RunPaths,
    secrets: Sequence[str],
    terminal: Deadline,
) -> int:
    """Exit ``0`` is ``complete`` with the parts that exist; no part at all is ``worker_error`` (M2b-D31)."""
    export = claim.manifest.upload_format == "rekordbox"
    parts = existing_parts(stdout, paths.scratch, export)
    expected = ["summary.json", "report.html", *(["export.xml"] if export else [])]
    missing = [name for name in expected if name not in parts]
    if missing:
        _log(f"engine exited 0 without {', '.join(missing)}")
    if not parts:
        text = _diagnostics_text(stdout, stderr, paths.diagnostics, secrets)
        return _fail(remote, claim, "worker_error", text, secrets, terminal)
    for name, path in parts.items():
        size = path.stat().st_size
        _log(
            f"part {name}: {size} bytes"
            + (f" (over the {MAX_PART_BYTES}-byte part cap)" if size > MAX_PART_BYTES else "")
        )
    try:
        _terminal(lambda: remote.complete(claim, parts, terminal), terminal)
    except StaleClaim as exc:
        _log(f"complete answered 409 {exc.error}; the run is another's")
        return 0
    except (ProtocolError, TransportError) as exc:
        _log(f"complete not recorded: {exc}")  # never fail after a complete attempt
        return 1
    _log(f"completed run {claim.run_id}")
    return 0


def _run_claimed(remote: TenantRemote, claim: Claim, config: WorkerConfig, secrets: Sequence[str]) -> int:
    deadline = Deadline(PRE_ENGINE_DEADLINE_SECS, start=claim.claimed_at)
    paths = RunPaths.for_config(config)
    try:
        argv = _prepare(remote, claim, paths, deadline)
    except StaleClaim as exc:
        _log(f"409 {exc.error} before the engine; the run is another's")
        return 0
    except (ProtocolError, TransportError) as exc:
        _log(f"before the engine: {exc}; left to the janitor")
        return 1
    except _PreEngineRefusalError as failure:
        _log(f"refused before the engine: {failure.error_code}")
        text = f"tenant-worker: {failure.reason}\n"
        return _fail(remote, claim, failure.error_code, text, secrets, Deadline(TERMINAL_WINDOW_SECS))
    except Exception:
        _log("worker error before the engine")
        text = _tail(traceback.format_exc(), secrets)
        return _fail(remote, claim, "worker_error", text, secrets, Deadline(TERMINAL_WINDOW_SECS))

    _log(f"engine started for run {claim.run_id}")
    returncode: int | None
    try:
        completed = subprocess.run(  # argv comes from build_argv's allow-list
            [sys.executable, "-m", "mixlab", *argv],
            cwd=paths.scratch,
            env=_subprocess_env(paths),
            timeout=config.run_timeout,
            capture_output=True,
            check=False,
        )
        returncode = completed.returncode
        stdout, stderr = _as_text(completed.stdout), _as_text(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout, stderr = _as_text(exc.stdout), _as_text(exc.stderr)
    except Exception:
        _log("the engine could not be started")
        text = _tail(traceback.format_exc(), secrets)
        return _fail(remote, claim, "worker_error", text, secrets, Deadline(TERMINAL_WINDOW_SECS))
    terminal = Deadline(TERMINAL_WINDOW_SECS)  # from the engine's end (M2b-D32)

    if returncode == 0:
        return _complete(remote, claim, stdout, stderr, paths, secrets, terminal)
    error_code = exit_code_to_error_code(returncode)
    _log(f"engine {'timed out' if returncode is None else f'exited {returncode}'}: {error_code}")
    text = _diagnostics_text(stdout, stderr, paths.diagnostics, secrets)
    return _fail(remote, claim, error_code, text, secrets, terminal)


def run_once(remote: TenantRemote, run_id: str, config: WorkerConfig) -> int:
    """Run one claim cycle for ``run_id``; returns the process's exit code.

    ``0`` when the run is reported (``complete`` or ``fail`` answered) or belongs to another
    execution (``409``); ``1`` when the run is left to the janitor.
    """
    pre_claim = Deadline(PRE_CLAIM_BOUND_SECS)
    try:
        worker_token = remote.acquire_worker_token(pre_claim)
        if pre_claim.expired():
            raise TransportError("token answered after the 60 s pre-claim bound")
        claim = _pre_engine(lambda: remote.claim(run_id, worker_token, pre_claim), pre_claim)
    except StaleClaim as exc:
        _log(f"claim answered 409 {exc.error}; run {run_id} is not claimable")
        return 0
    except (ProtocolError, TransportError) as exc:
        _log(f"no claim for run {run_id}: {exc}")
        return 1
    _log(f"claimed run {run_id}")
    secrets = (
        remote.config.client_secret,
        worker_token,
        claim.run_token,
        claim.claim_id,
        os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    return _run_claimed(remote, claim, config, secrets)
