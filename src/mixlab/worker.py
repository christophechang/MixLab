"""Crash-safe worker loop for MixLab Anywhere (#91 / M3).

See ``docs/architecture/mixlab-anywhere.md`` §6 (worker protocol) — the governing
spec — plus §4 #10/#11/#12 (claim/complete/fail) and §8 (failure policies). This
module turns a queued run manifest into a completed (or failed) run without ever
letting a pipeline crash take down the poll loop:

- The engine is always invoked as a **subprocess** (never in-process), so a pipeline
  exception, hang, or hard crash is isolated to that child process.
- Every unexpected error inside :func:`process_one` is caught, best-effort reported to
  the API, and turned into a boolean the loop can act on; nothing escapes except
  ``KeyboardInterrupt``/``SystemExit``.
- Transport-level failures (the API itself being down) deliberately do *not* attempt a
  ``fail`` call — they leave the claim to lease expiry (§8), which requeues the run.

The unit at the heart of correctness is :func:`build_argv`: a strict, deterministic
allow-list mapping the manifest's camelCase ``flags`` (§5.1) onto CLI flags. Anything
unrecognised or ill-typed is rejected before a subprocess is ever spawned.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from mixlab.remote import MixLabRemote, RemoteError, RunManifest
from mixlab.sync import sync_down, sync_up

# Local history file synced around each run (mirrors ``mixlab.__main__._HISTORY_PATH``).
_HISTORY_REL_PATH = Path(".mixlab/concept-history.json")

# The pipeline's default collection import path (mirrors ``mixlab.__main__._XML_PATH``).
_DEFAULT_XML_PATH = "import/rekordbox.xml"

# Log tails posted to the API are bounded so a runaway pipeline can't PUT megabytes.
_TAIL_LEN = 4000

# stdout markers the pipeline prints for its two required artifacts (see
# ``mixlab.__main__._write_html_report`` and the summary.json wiring in ``run``).
_REPORT_PREFIX = "HTML report:"
_SUMMARY_PREFIX = "Run summary:"


class WorkerFlagError(ValueError):
    """Raised when a run manifest carries a flag the worker refuses to map to argv."""


# ---------------------------------------------------------------------------
# build_argv — the pure, correctness-critical flag mapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FlagSpec:
    """One allow-listed flag: its wire key, CLI flag, kind, and (for enums) allowed set."""

    wire_key: str
    cli_flag: str
    kind: str  # "str" | "enum" | "int" | "bool"
    allowed: frozenset[str] | None = None


# Fixed canonical ordering — build_argv emits flags in this order regardless of the
# manifest's dict order, so the resulting argv (and any test asserting on it) is
# deterministic. Enum allowed-sets mirror the CLI's ``choices=`` in ``__main__``.
_FLAG_SPECS: tuple[_FlagSpec, ...] = (
    _FlagSpec("genre", "--genre", "str"),
    _FlagSpec("mode", "--mode", "enum", frozenset({"unplayed", "all", "played"})),
    _FlagSpec("risk", "--risk", "enum", frozenset({"low", "medium", "high"})),
    _FlagSpec("directions", "--directions", "enum", frozenset({"mixed", "off", "only"})),
    _FlagSpec("intent", "--intent", "str"),
    _FlagSpec("mixLength", "--mix-length", "int"),
    _FlagSpec("stage1Seed", "--stage1-seed", "int"),
    _FlagSpec("resequence", "--resequence", "bool"),
    _FlagSpec("deep", "--deep", "bool"),
)
_KNOWN_KEYS = frozenset(spec.wire_key for spec in _FLAG_SPECS)


def build_argv(flags: Mapping[str, object]) -> list[str]:
    """Map a run manifest's ``flags`` (§5.1) onto a deterministic list of CLI argv.

    Strict allow-list: an unknown key, a wrongly-typed value, or an out-of-range enum
    raises :class:`WorkerFlagError` naming the offending key — the manifest never
    reaches a subprocess. ``None`` values and absent keys are omitted. Booleans emit a
    bare flag when truthy and nothing when falsy. Output order is fixed
    (:data:`_FLAG_SPECS`), independent of the input mapping's iteration order.
    """
    for key in flags:
        if key not in _KNOWN_KEYS:
            raise WorkerFlagError(f"unknown flag key: {key!r}")

    argv: list[str] = []
    for spec in _FLAG_SPECS:
        if spec.wire_key not in flags:
            continue
        value = flags[spec.wire_key]
        if value is None:
            continue

        if spec.kind in ("str", "enum"):
            if not isinstance(value, str):
                raise WorkerFlagError(f"flag {spec.wire_key!r} must be a string, got {type(value).__name__}")
            if spec.kind == "str":
                if not value:
                    raise WorkerFlagError(f"flag {spec.wire_key!r} must be a non-empty string")
            elif spec.allowed is not None and value not in spec.allowed:
                allowed = ", ".join(sorted(spec.allowed))
                raise WorkerFlagError(f"flag {spec.wire_key!r} must be one of [{allowed}], got {value!r}")
            argv.extend([spec.cli_flag, value])
        elif spec.kind == "int":
            # bool is a subclass of int — guard it first so True/False can never be
            # smuggled in as 1/0 for a numeric flag.
            if isinstance(value, bool) or not isinstance(value, int):
                raise WorkerFlagError(f"flag {spec.wire_key!r} must be an integer, got {type(value).__name__}")
            argv.extend([spec.cli_flag, str(value)])
        else:  # "bool"
            if not isinstance(value, bool):
                raise WorkerFlagError(f"flag {spec.wire_key!r} must be a boolean, got {type(value).__name__}")
            if value:
                argv.append(spec.cli_flag)

    return argv


# ---------------------------------------------------------------------------
# WorkerConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerConfig:
    """Static configuration for one worker process."""

    xml_path: Path
    run_timeout: float
    repo_root: Path
    worker_id: str = field(default_factory=lambda: f"mixlab-worker-{socket.gethostname()}")

    @classmethod
    def from_env(cls, *, repo_root: Path | None = None) -> WorkerConfig:
        """Build a config from ``MIXLAB_WORKER_XML_PATH`` / ``MIXLAB_WORKER_RUN_TIMEOUT``.

        ``repo_root`` defaults to the current working directory (launchd runs the worker
        from the repo). A relative ``xml_path`` is anchored under ``repo_root`` so the
        downloaded collection lands exactly where the subprocess — run with
        ``cwd=repo_root`` — expects to read it.
        """
        root = repo_root if repo_root is not None else Path.cwd()
        xml_path = Path(os.environ.get("MIXLAB_WORKER_XML_PATH", _DEFAULT_XML_PATH))
        if not xml_path.is_absolute():
            xml_path = root / xml_path
        run_timeout = float(os.environ.get("MIXLAB_WORKER_RUN_TIMEOUT", "1200"))
        return cls(xml_path=xml_path, run_timeout=run_timeout, repo_root=root)


# ---------------------------------------------------------------------------
# process_one — one claim cycle (architecture §6)
# ---------------------------------------------------------------------------


class _ArtifactError(RuntimeError):
    """Internal: a required artifact line/file was missing from a successful run."""


def _as_text(value: object) -> str:
    """Coerce captured subprocess output (str, bytes, or None) to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _combine_tail(stdout: str, stderr: str) -> str:
    """Last :data:`_TAIL_LEN` chars of the combined stderr+stdout capture."""
    return (stderr + stdout)[-_TAIL_LEN:]


def _safe_fail(remote: MixLabRemote, run_id: str, *, error: str, log_tail: str) -> None:
    """Report a run failure, swallowing (and logging) any error from ``fail`` itself."""
    try:
        remote.fail(run_id, error=error, log_tail=log_tail)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # fail() is best-effort; the run is already lost
        print(f"worker: fail() for run {run_id} errored: {exc!r}", file=sys.stderr)


def _resolve_artifact(stdout: str, prefix: str, repo_root: Path) -> Path:
    """Find the ``{prefix} {path}`` line in ``stdout`` and return the existing file path.

    Relative paths are resolved against ``repo_root`` (the subprocess's cwd). Raises
    :class:`_ArtifactError` when the line is absent or the file does not exist.
    """
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            raw = stripped[len(prefix) :].strip()
            path = Path(raw)
            if not path.is_absolute():
                path = repo_root / path
            if not path.is_file():
                raise _ArtifactError(f"pipeline reported {prefix} {raw!r} but no file exists at {path}")
            return path
    raise _ArtifactError(f"pipeline stdout missing '{prefix}' line")


def _run_claimed(remote: MixLabRemote, config: WorkerConfig, manifest: RunManifest) -> bool:
    """Execute a single already-claimed run through the §6 cycle. Returns True (the job
    was consumed — completed or failed) or False (transport error; leave to lease expiry).
    """
    run_id = manifest.run_id
    history_path = config.repo_root / _HISTORY_REL_PATH

    # Step 2 — sync down. A transport error means the API is down: do NOT attempt fail
    # (it would fail too); leave the claim to lease expiry (§8). Any other failure is a
    # job-infrastructure failure we can and should report.
    try:
        sync_down(remote, history_path)
    except RemoteError as exc:
        print(f"worker: sync_down transport error; leaving run {run_id} to lease expiry: {exc}", file=sys.stderr)
        return False
    except Exception as exc:  # any non-transport failure is reportable
        _safe_fail(remote, run_id, error=f"sync_down failed: {exc}", log_tail=traceback.format_exc()[-_TAIL_LEN:])
        return True

    # Step 3 — download the collection. Same failure taxonomy as step 2.
    try:
        remote.download_collection(manifest.upload_id, config.xml_path)
    except RemoteError as exc:
        print(f"worker: download transport error; leaving run {run_id} to lease expiry: {exc}", file=sys.stderr)
        return False
    except Exception as exc:  # any non-transport failure is reportable
        _safe_fail(
            remote, run_id, error=f"download_collection failed: {exc}", log_tail=traceback.format_exc()[-_TAIL_LEN:]
        )
        return True

    # Step 4 — map flags to argv. A rejected flag never reaches a subprocess.
    try:
        argv = build_argv(manifest.flags)
    except WorkerFlagError as exc:
        _safe_fail(remote, run_id, error=f"invalid run flags: {exc}", log_tail="")
        return True

    # Step 5 — run the pipeline as an isolated subprocess with a hard timeout.
    try:
        completed = subprocess.run(  # argv is built from a strict allow-list
            [sys.executable, "-m", "mixlab", *argv],
            capture_output=True,
            text=True,
            timeout=config.run_timeout,
            cwd=config.repo_root,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log_tail = _combine_tail(_as_text(exc.stdout), _as_text(exc.stderr))
        _safe_fail(remote, run_id, error=f"pipeline timed out after {config.run_timeout:g}s", log_tail=log_tail)
        return True

    # Step 7 — non-zero exit is a pipeline failure; report it with the log tail.
    if completed.returncode != 0:
        _safe_fail(
            remote,
            run_id,
            error=f"pipeline exited {completed.returncode}",
            log_tail=_combine_tail(completed.stdout, completed.stderr),
        )
        return True

    # Step 6 — success. Parse the artifact paths the pipeline printed. This epic's runs
    # never request an export, so export_path is always None (v1).
    try:
        summary_path = _resolve_artifact(completed.stdout, _SUMMARY_PREFIX, config.repo_root)
        report_path = _resolve_artifact(completed.stdout, _REPORT_PREFIX, config.repo_root)
    except _ArtifactError as exc:
        _safe_fail(remote, run_id, error=str(exc), log_tail=_combine_tail(completed.stdout, completed.stderr))
        return True

    remote.complete(run_id, summary_path=summary_path, report_path=report_path, export_path=None)

    # Post-complete history push. The run IS complete; a sync_up failure here is only a
    # warning — the next cycle's sync reconciles history.
    try:
        sync_up(remote, history_path)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # run already complete; history reconciles next cycle
        print(f"worker: sync_up after complete failed (run {run_id} already complete): {exc}", file=sys.stderr)

    return True


def process_one(remote: MixLabRemote, config: WorkerConfig) -> bool:
    """Run one claim cycle. Returns True when a job was claimed (and completed/failed),
    False when the queue was empty or the API was unreachable.

    Nothing escapes except ``KeyboardInterrupt``/``SystemExit``: an unexpected error is
    logged, a best-effort ``fail`` is attempted when a job was claimed, and the loop
    keeps running.
    """
    manifest: RunManifest | None = None
    try:
        manifest = remote.claim(config.worker_id)
        if manifest is None:
            return False
        return _run_claimed(remote, config, manifest)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # the loop must survive any pipeline/API surprise
        print(f"worker: unexpected error in process_one: {exc!r}", file=sys.stderr)
        if manifest is not None:
            _safe_fail(
                remote, manifest.run_id, error=f"worker error: {exc!r}", log_tail=traceback.format_exc()[-_TAIL_LEN:]
            )
            return True
        return False


# ---------------------------------------------------------------------------
# run_worker — the poll loop + signal handling
# ---------------------------------------------------------------------------


class _StopSignal:
    """Flag flipped by a SIGINT/SIGTERM handler so the loop finishes the current cycle."""

    def __init__(self) -> None:
        self._stop = False

    def request_stop(self, signum: int, frame: FrameType | None) -> None:
        self._stop = True

    @property
    def stopping(self) -> bool:
        return self._stop


def run_worker(remote: MixLabRemote, config: WorkerConfig, *, poll_seconds: float, once: bool) -> int:
    """Drive the worker: a single cycle (``once``) or a poll loop until interrupted.

    In loop mode, SIGINT/SIGTERM set a flag so the current cycle finishes before a clean
    exit; the previous handlers are restored on the way out. Empty-poll cycles print
    nothing (a 30 s heartbeat would flood launchd logs).
    """
    if once:
        process_one(remote, config)
        return 0

    print(
        f"Worker online — polling {remote.base_url} every {poll_seconds:g}s (worker_id={config.worker_id})",
        flush=True,
    )

    stop = _StopSignal()
    previous: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, stop.request_stop)

    try:
        while not stop.stopping:
            claimed = process_one(remote, config)
            if stop.stopping:
                break
            if not claimed:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        pass  # clean shutdown — treated the same as a signal-flag stop
    finally:
        for saved_sig, handler in previous.items():
            signal.signal(saved_sig, handler)  # type: ignore[arg-type]  # restore original handler object

    return 0
