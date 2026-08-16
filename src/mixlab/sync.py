"""History + feedback sync between the local engine and MixLab Anywhere (#92 / M4).

See ``docs/architecture/mixlab-anywhere.md`` §5.3 (feedback event shape) and §6
steps 2 & 6 (worker protocol) for the governing contract. This module has no CLI
surface of its own yet — it is consumed by the worker loop (M3) around each claim
cycle: ``sync_down`` before executing a run, ``sync_up`` after a run appends a new
history entry.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mixlab.history import ConceptHistory, apply_feedback_verdict, load_history, save_history
from mixlab.remote import FeedbackEvent, MixLabRemote, RemoteConflictError


@dataclass
class SyncReport:
    """Outcome of one :func:`sync_down` cycle."""

    applied: int = 0
    skipped: int = 0
    acked: int = 0

    def __str__(self) -> str:
        return f"sync: applied {self.applied} verdict(s), skipped {self.skipped}, acked {self.acked}"


def _etag_path(history_path: Path) -> Path:
    return Path(str(history_path) + ".etag")


def _load_cached_etag(history_path: Path) -> str | None:
    etag_path = _etag_path(history_path)
    if not etag_path.exists():
        return None
    text = etag_path.read_text().strip()
    return text or None


def _save_cached_etag(history_path: Path, etag: str | None) -> None:
    if not etag:
        return
    _etag_path(history_path).write_text(etag)


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via temp-file + rename so a crash mid-write never
    leaves a truncated/partial history file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as tmp_file:
            tmp_file.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _compose_notes(event: FeedbackEvent) -> str:
    """Fold rating/publishedMixSlug/notes into one structured text blob.

    ``ConceptRecord`` has a single free-text ``feedback_notes`` field (no separate
    rating/slug columns — see M1), so a remote feedback event (§5.3) is flattened into
    a ``"rating=4; mix=<slug>; <notes>"`` style string, omitting any part the event
    doesn't carry.
    """
    parts: list[str] = []
    if event.rating is not None:
        parts.append(f"rating={event.rating}")
    if event.published_mix_slug:
        parts.append(f"mix={event.published_mix_slug}")
    if event.notes:
        parts.append(event.notes)
    return "; ".join(parts)


def _apply_events(history: ConceptHistory, events: list[FeedbackEvent]) -> tuple[int, int, list[str]]:
    """Apply every event to ``history`` in place via the shared verdict helper.

    Returns ``(applied, skipped, processed_event_ids)``. An event whose
    ``concept_id`` matches nothing is counted as skipped (and reported on stderr) but
    still marked processed — re-delivering an unmatchable event forever is worse than
    acking it once (docs/architecture/mixlab-anywhere.md §6 step 2).
    """
    applied = 0
    skipped = 0
    processed_ids: list[str] = []
    for event in events:
        record = apply_feedback_verdict(history, event.concept_id, event.verdict, _compose_notes(event))
        processed_ids.append(event.event_id)
        if record is None:
            skipped += 1
            print(
                f'sync: no concept matching "{event.concept_id}" (event {event.event_id}); skipping',
                file=sys.stderr,
            )
        else:
            applied += 1
    return applied, skipped, processed_ids


def sync_down(remote: MixLabRemote, history_path: Path) -> SyncReport:
    """Pull remote history + pending feedback into the local history file.

    Implements docs/architecture/mixlab-anywhere.md §6 step 2:

    1. Adopt the remote history file (atomic write + cached ETag) unless the remote
       has none yet (404 -> keep the local file; first run seeds the remote from it).
    2. Apply every pending feedback event to the (possibly just-adopted) local
       history via the shared :func:`~mixlab.history.apply_feedback_verdict` helper.
    3. If anything was applied, push the updated history back with ``If-Match``;
       on a 412 conflict, re-pull once, re-apply all events again, and retry the
       push exactly once more (raising on a second conflict).
    4. Ack every processed event id (applied and skipped alike) — but only once the
       push has succeeded, or immediately when nothing needed pushing.
    """
    etag, body = remote.get_history()
    if body:
        _atomic_write(history_path, body)
        _save_cached_etag(history_path, etag)
    # body == "" (404): no remote history yet — keep the local file as the seed.

    events = remote.pending_feedback()
    if not events:
        return SyncReport()

    history = load_history(history_path)
    applied, skipped, processed_ids = _apply_events(history, events)

    if applied:
        cached_etag = _load_cached_etag(history_path)
        save_history(history, history_path)
        body_text = history_path.read_text()
        try:
            new_etag = remote.put_history(body_text, etag=cached_etag)
        except RemoteConflictError:
            fresh_etag, remote_body = remote.get_history()
            if remote_body:
                _atomic_write(history_path, remote_body)
            history = load_history(history_path)
            applied, skipped, processed_ids = _apply_events(history, events)
            save_history(history, history_path)
            body_text = history_path.read_text()
            new_etag = remote.put_history(body_text, etag=fresh_etag)
        _save_cached_etag(history_path, new_etag)

    remote.ack_feedback(processed_ids)
    return SyncReport(applied=applied, skipped=skipped, acked=len(processed_ids))


def _parse_history_text(text: str) -> ConceptHistory:
    """Parse history JSON text into a :class:`ConceptHistory`.

    ``load_history`` only reads from a path; the text is round-tripped through a
    scratch file so this reuses its exact field-filtering/coercion semantics rather
    than hand-rolling a second parser.
    """
    if not text:
        return ConceptHistory()
    with tempfile.TemporaryDirectory() as tmp_dir:
        scratch = Path(tmp_dir) / "history.json"
        scratch.write_text(text)
        return load_history(scratch)


def _serialize_history(history: ConceptHistory) -> str:
    """Serialise a :class:`ConceptHistory` using ``save_history``'s exact schema
    (round-tripped through a scratch file rather than hand-rolled)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        scratch = Path(tmp_dir) / "history.json"
        save_history(history, scratch)
        return scratch.read_text()


def _merge_histories(remote_text: str, local_text: str) -> ConceptHistory:
    """Merge remote + local history: every remote run, plus any local-only run_ids.

    Used on a ``sync_up`` PUT conflict (§6 step 6) — the remote moved on since we
    last read it (another sync cycle or run), so remote entries win by identity and
    local entries are added only when their ``run_id`` isn't already present
    remotely.

    A run *deleted* remotely mid-run is indistinguishable here from one this worker
    generated while the remote moved on: both are local-only, so this merge would
    re-add it, resurrecting a history entry that belongs to no manifest. The API is
    what prevents that — it refuses a run delete while any run holds a live claim,
    so a purge can never land inside one worker's sync_down → sync_up window. Keep
    the two in step: relaxing that guard needs deletion tombstones here first.
    """
    remote_history = _parse_history_text(remote_text)
    local_history = _parse_history_text(local_text)
    remote_run_ids = {entry.run_id for entry in remote_history.runs}
    merged_runs = list(remote_history.runs)
    merged_runs.extend(entry for entry in local_history.runs if entry.run_id not in remote_run_ids)
    return ConceptHistory(runs=merged_runs)


def sync_up(remote: MixLabRemote, history_path: Path) -> None:
    """Push the local history file to the remote, merging on conflict.

    Implements docs/architecture/mixlab-anywhere.md §6 step 6 (post-run history
    push): PUT the local file with ``If-Match``; on a 412 conflict, GET the remote
    copy, merge it with the local copy (remote entries plus local-only run_ids),
    and PUT the merge exactly once more (raising on a second conflict).
    """
    body = history_path.read_text()
    cached_etag = _load_cached_etag(history_path)
    try:
        new_etag = remote.put_history(body, etag=cached_etag)
    except RemoteConflictError:
        fresh_etag, remote_body = remote.get_history()
        merged = _merge_histories(remote_body, body)
        merged_text = _serialize_history(merged)
        new_etag = remote.put_history(merged_text, etag=fresh_etag)
    _save_cached_etag(history_path, new_etag)
