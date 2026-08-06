# Worker Map Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The MixLab Anywhere worker claims queued map jobs from the API, runs `mixlab --map`, and pushes the JSON payload back — closing the loop between the API's maps endpoints (v1.63) and the web's constellation overlay.

**Architecture:** Additive twins of the run machinery: three new methods on `MixLabRemote` (claim/complete/fail against the `api/mixlab/maps` surface) and a `_run_claimed_map` in `worker.py` mirroring `_run_claimed`'s subprocess discipline. `process_one` tries the run queue first (runs keep priority), then the map queue. The subprocess is `./mixlab --map --mode unplayed` with `MIXLAB_COLLECTION_PATH` pointed at a map-specific download path; stdout bytes are pushed verbatim (the API validates JSON at push time).

**Tech Stack:** Python 3.12, httpx, respx tests, mypy --strict, ruff.

## Global Constraints

- mypy --strict, full annotations, `from __future__ import annotations`, no `Any`; ruff format + check clean, no suppressions.
- Worker crash-safety invariants hold: subprocess isolation, nothing escapes `process_one` except `KeyboardInterrupt`/`SystemExit`, transport failures leave claims to lease expiry (no fail call), unexpected errors best-effort fail the claimed job.
- No behavior change to the existing run path — `_run_claimed`, `build_argv`, run claim ordering untouched.
- API wire facts (v1.63, verified): `POST /api/mixlab/maps/claim` body `{"workerId": "..."}` → 200 `MixLabMapJob` JSON (camelCase, `uploadId` field) | 204 empty; `POST /api/mixlab/maps/{uploadId}/result` raw JSON body (the payload bytes), `Content-Type: application/json` → 204 | 404 | 400 unparseable; `POST /api/mixlab/maps/{uploadId}/fail` body `{"error": "..."}` → 204 | 404. All bearer-authed like the run endpoints.
- Tests: respx for HTTP (mirror tests/test_remote.py), subprocess mocked via the same seams tests/test_worker.py already uses; naming `test_<what>_<condition>_<expected>`.
- Commands: `.venv/bin/python -m ruff format .`, `ruff check .`, `mypy .`, `pytest`.
- Conventional commits, no Co-Authored-By.

---

### Task 1: MixLabRemote map methods

**Files:**
- Modify: `src/mixlab/remote.py`
- Test: `tests/test_remote.py` (append)

**Interfaces:**
- Produces on `MixLabRemote`:
  - `claim_map(self, worker_id: str) -> str | None` — POST `/api/mixlab/maps/claim` body `{"workerId": worker_id}`; 200 → parse JSON, return `uploadId` (raise `RemoteError` if the field is missing/empty, mirroring how `claim` treats a malformed manifest); 204 → `None`; other statuses → `_raise_for_status`.
  - `complete_map(self, upload_id: str, payload: bytes) -> None` — POST `/api/mixlab/maps/{upload_id}/result`, body exactly `payload`, header `Content-Type: application/json`; non-2xx → `_raise_for_status`.
  - `fail_map(self, upload_id: str, *, error: str) -> None` — POST `/api/mixlab/maps/{upload_id}/fail` body `{"error": error}`; non-2xx → `_raise_for_status`.

Implementation mirrors the existing `claim`/`complete`/`fail` methods' structure exactly (per-call `httpx.Client` context manager, `self._headers()`, `self._url(path)`, timeout from config, `_raise_for_status` helper). Read those methods first and copy their shape; `upload_id` goes through the same URL-quoting treatment the run methods give `run_id` (match whatever they do — quote if they quote, plain f-string if they don't).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_remote.py`, mirroring its respx conventions (fixtures, base-url, auth-header assertions). Cover:
  - `test_claim_map_returns_upload_id_on_200` (respond `{"uploadId": "up-1", "status": "running"}`; assert bearer header + body `{"workerId": ...}` sent)
  - `test_claim_map_returns_none_on_204`
  - `test_claim_map_missing_upload_id_raises_remote_error` (200 with `{}`)
  - `test_complete_map_posts_payload_bytes_verbatim` (assert request content bytes equal the payload and Content-Type is application/json)
  - `test_complete_map_non_2xx_raises` (404)
  - `test_fail_map_posts_error_body` (assert `{"error": ...}`)
- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_remote.py -k map -q` → FAIL.
- [ ] **Step 3:** Implement the three methods.
- [ ] **Step 4:** `pytest tests/test_remote.py -q` → all pass.
- [ ] **Step 5:** Commit — `feat(remote): add map job claim/complete/fail client methods`

---

### Task 2: Worker map cycle

**Files:**
- Modify: `src/mixlab/worker.py`
- Test: `tests/test_worker.py` (append)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 1's `claim_map`/`complete_map`/`fail_map`.
- Produces in `worker.py`:
  - Module constant `_MAP_XML_PATH = ".mixlab/worker-map-collection.xml"` (own download path — never clobbers `_DEFAULT_XML_PATH` or the local `import/rekordbox.xml`, same reasoning as the existing comment).
  - `_run_claimed_map(remote: MixLabRemote, config: WorkerConfig, upload_id: str) -> bool` — mirrors `_run_claimed`'s discipline:
    1. `remote.download_collection(upload_id, repo_root / _MAP_XML_PATH)`
    2. Subprocess: the same executable invocation style `_run_claimed` uses (read it — reuse its command-construction and env-merge approach) with argv `["--map", "--mode", "unplayed"]`, env override `MIXLAB_COLLECTION_PATH` → the map download path, `capture_output=True`, the same timeout mechanism `_run_claimed` applies (reuse its constant/config).
    3. Exit 0 → `remote.complete_map(upload_id, proc.stdout_bytes)` (stdout as bytes, untouched — the payload contract). Print one stderr status line (`map job {upload_id}: completed`).
    4. Exit non-0 → `remote.fail_map(upload_id, error=f"mixlab --map exited {rc}", ...)` — include a bounded stderr tail per `_TAIL_LEN`, matching how `_run_claimed` bounds log tails (check `fail_map`'s signature takes only `error`: fold the tail into the error string, bounded).
    5. Return True.
  - `process_one` change (minimal): after `remote.claim(...)` returns `None`, try `upload_id = remote.claim_map(config.worker_id)`; `None` → return False; else `return _run_claimed_map(remote, config, upload_id)`. The existing `except BaseException` handler must best-effort fail whichever job kind was claimed — extend the handler so a claimed map job gets `fail_map` (mirror the `_safe_fail` pattern with a `_safe_fail_map` twin swallowing `RemoteError`).
- CHANGELOG: `## Unreleased` entry describing worker map-job claiming (bold-lead style).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_worker.py`, reusing its existing fakes/monkeypatch seams (read the file first; it fakes `MixLabRemote` and subprocess). Cover:
  - `test_process_one_prefers_run_queue_over_maps` (both queues nonempty → run processed, `claim_map` never called)
  - `test_process_one_claims_map_when_run_queue_empty` (run claim None, map claim "up-1" → `_run_claimed_map` path: download called with the map path, subprocess argv contains `--map` and `--mode unplayed`, env carries `MIXLAB_COLLECTION_PATH` pointing at the map path, stdout bytes pushed via `complete_map`)
  - `test_process_one_returns_false_when_both_queues_empty`
  - `test_run_claimed_map_nonzero_exit_fails_job_with_bounded_tail`
  - `test_process_one_map_claim_unexpected_error_best_effort_fails_map` (exception after claim → `fail_map` attempted, True returned, nothing escapes)
- [ ] **Step 2:** `pytest tests/test_worker.py -k map -q` → FAIL.
- [ ] **Step 3:** Implement per the spec above.
- [ ] **Step 4:** `pytest tests/test_worker.py tests/test_remote.py -q` → PASS; then full gate: ruff format + check, mypy, full pytest — all clean.
- [ ] **Step 5:** Changelog entry, then commit — `feat(worker): claim and execute library-map jobs`

---

## Out of scope

Web overlay (next plan, mixlab-web). Deployment: worker restart on the Mac mini happens at the next MixLab release.
