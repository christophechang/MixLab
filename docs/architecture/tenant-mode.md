# Tenant mode: the multi-tenant worker

`MIXLAB_API_MODE=tenant` is how the engine runs for the multi-tenant MixLab, one run per container
execution. The wire protocol is `tools/run-contract/protocol.md` in the `mixlab-multi-tenant`
repository, which is binding; this page describes the engine's side of it. With no tenant variable
set, nothing here applies: `mixlab --genre …`, `mixlab --worker` and the MixLab Anywhere protocol
(`mixlab-anywhere.md`) behave as before.

Two modules carry it, beside the untouched Anywhere ones (`worker.py`, `remote.py`, `sync.py`):

- `src/mixlab/tenant_remote.py`: the protocol client. One `httpx.Client` per request; every
  request's timeout is 30 seconds or the time its window has left, whichever is less.
- `src/mixlab/tenant_worker.py`: one claim cycle. It shares `build_argv` with the Anywhere worker
  and nothing else.

## Starting an execution

```bash
python -m mixlab --worker-once --run <runId>   # or MIXLAB_RUN_ID=<runId>
```

In tenant mode `--worker-once` runs the tenant cycle for the run named by `--run`, else by
`MIXLAB_RUN_ID`. Without either, or with a worker variable below missing, it prints the reason and
exits 1 before any request. `--worker`, the poll loop, exits 1: one execution runs one run.

## Variables

The dispatcher sets these on the job execution:

| Variable | Read by | Meaning |
|---|---|---|
| `MIXLAB_API_MODE` | worker and engine | `tenant` |
| `MIXLAB_RUN_ID` | worker | The run to claim (`--run` overrides it) |
| `MIXLAB_API_URL` | worker | The API's base URL |
| `MIXLAB_PROTOCOL_VERSION` | worker and engine | Sent in the claim; copied into `summary.json` |
| `MIXLAB_WORKER_CLIENT_ID` | worker | The worker app registration's client id |
| `MIXLAB_WORKER_CLIENT_SECRET` | worker | Its client secret |
| `MIXLAB_WORKER_TOKEN_ENDPOINT` | worker | The tenant's OAuth 2.0 token endpoint |
| `MIXLAB_WORKER_SCOPE` | worker | The API's `.default` scope |
| `MIXLAB_WORKER_XML_PATH` | worker | Where the upload is gunzipped; its directory is the run's scratch |
| `MIXLAB_HISTORY_PATH` | worker and engine | The concept history in scratch (default `<scratch>/concept-history.json` for the worker) |
| `MIXLAB_WORKER_RUN_TIMEOUT` | worker | The engine subprocess's timeout in seconds (default `1200`) |
| `MIXLAB_ENGINE_DIGEST` | engine | The image digest, copied into `summary.json` when it reads `sha256:` and 64 hex digits |
| `ANTHROPIC_API_KEY` | engine | The one secret the engine sees |

The engine's other tenant variables (`MIXLAB_TOKEN_BUDGET`, `MIXLAB_STAGE0_MODEL`,
`MIXLAB_STAGE2_MODEL`, `MIXLAB_STAGE2_EFFORT`, `MIXLAB_FALLBACKS`, `MIXLAB_LLM_STUB`,
`MIXLAB_PLAYED_SOURCE`, `MIXLAB_KEY_NOTATION`) pass through to the engine unchanged; the README
describes each.

## The scratch directory and the subprocess

The scratch is the directory of `MIXLAB_WORKER_XML_PATH`. The worker creates `reports/`,
`diagnostics/` and `export/` under it and names `genres.json` for the genre cache. The engine
runs as

```
python -m mixlab <flags from the manifest> [--export <scratch>/export]   # cwd: the scratch
```

with `--export` only for a Rekordbox upload. Its environment is the worker's **minus**
`MIXLAB_WORKER_CLIENT_ID`, `MIXLAB_WORKER_CLIENT_SECRET`, `MIXLAB_WORKER_TOKEN_ENDPOINT` and
`MIXLAB_WORKER_SCOPE`, **plus** `MIXLAB_COLLECTION_PATH`, `MIXLAB_HISTORY_PATH`,
`MIXLAB_REPORT_DIR`, `MIXLAB_DIAGNOSTICS_DIR` and `MIXLAB_GENRE_CACHE_PATH` for this run. The run
token and the claim id are never environment variables, so nothing inherits them. Everything the
engine writes lands in the scratch.

## One cycle

1. **Token and claim.** One form `POST` to the token endpoint (client credentials), then
   `POST …/claim {"protocolVersion": n}` with that token. `409 not_claimable` exits 0; `426`, `400`,
   `401`, `403` and a refused token exit 1; a `5xx` or no answer is tried three times, 1 s then 2 s
   apart.
2. **Sync.** `GET …/history` (a `204` writes `{"runs": []}`), `GET …/feedback`. Each event is
   applied to the history (rating, mix link and notes folded into the notes as
   `rating=4; mix=<link>; <notes>`); if any matched, `PUT …/history` under `If-Match` or
   `If-None-Match: *`, with one re-read and re-apply on `409 etag_mismatch`. Every event read is
   acknowledged, matched or not. With no events there is no `PUT` and no ack. Then
   `progress synced`.
3. **Download.** `GET …/upload`, gunzipped as it streams. The inflater is never asked past
   100,663,296 bytes (96 MiB): the byte past it fails the run `upload_too_large`. A stream that is
   not gzip, ends before its member does, or carries bytes after it fails `upload_unreadable`. A
   break in the stream is retried from the start. Then `progress downloaded`.
4. **Flags.** The manifest's flags through `build_argv`; a refused flag fails `invalid_flags`.
5. **`engine-started`.** The engine is spawned only after this `progress` is answered `204`
   inside the pre-engine deadline; otherwise the worker exits 1 without spawning and without
   `fail`, and the janitor requeues the run.
6. **The engine** runs as above.
7. **Exit 0 is `complete`**, one multipart `POST` with a part per artifact the engine printed and
   wrote: `summary.json`, `report.html` and, for Rekordbox, `export.xml`. Because the engine's
   stdout also carries text from the DJ's file and the model, a part is taken only from a line
   whose path resolves, symlinks followed, inside `<scratch>/reports` (or `<scratch>/export` for
   the export). A missing part is left out and logged; only when none exists does the worker fail `worker_error`. A part over
   4,194,304 bytes is posted anyway (its size logged first); the `413` ends the execution.
   **Any other exit, or the timeout, is `fail`**, never after a `complete` attempt.

Any `409 stale_claim` or `not_running` exits 0 at once: the run belongs to another execution or
the janitor. A `409 busy` is repeated once after a second, then treated as a `5xx`.

## The three windows

| Window | Starts | Length | Bounds |
|---|---|---|---|
| Pre-claim | the process | 60 s | the token request and the claim's attempts |
| Pre-engine | the claim's `200` | 180 s | every sync, download and `progress` request and retry sleep; `engine-started`'s `204` counts only inside it |
| Terminal | the engine's end | 180 s | the diagnostics write, `complete` or `fail`, and their back-off (1, 2, 4, 8, 16, then 30 s) |

Each request is bounded by its window as a whole, not only per read: an answer still arriving
when the window ends is abandoned, and one that arrives after it is refused, a claim included.
With the engine's 20 minutes, an execution lasts at most 1 + 3 + 20 + 3 = 27 minutes, the job's
replica timeout, and 26 minutes once claimed.

## `fail` and the diagnostics blob

Before `fail`, the worker writes one text to the claim's `diagnosticsUploadUrl`: the last 4,000
characters of the engine's stderr then stdout, then every `truncated-*.txt` the engine left in
`<scratch>/diagnostics`. Every occurrence of the client secret, the worker token, the run token,
the claim id and the Anthropic key is replaced by `[redacted]`; a value shorter than 8 characters
is skipped. The write is one attempt and one retry and runs only while more than 60 seconds of the
terminal window remain; `fail`'s `diagnosticsRef` is the blob's path after `/runs/{runId}/`, or
`null` when the write did not succeed. The worker's own log lines carry ids, stages, codes and
sizes, never the engine's output or anything from the DJ's file.

## Exit codes

| Engine exit | Worker sends |
|---|---|
| `0` | `complete` |
| `2` | `fail no_shortlist` |
| `3` | `fail token_budget` |
| `4` | `fail provider_cap` |
| `5` | `fail refusal` |
| `6` | `fail provider_unavailable` |
| any other | `fail engine_failed` |
| killed at the timeout | `fail engine_timeout` |

Before the engine the worker sends `invalid_flags`, `upload_too_large`, `upload_unreadable` or
`worker_error` itself. The worker process exits 0 when the run was reported (or belongs to
another execution) and 1 when it is left to the janitor.

## What the engine writes in tenant mode

`summary.json` is `schemaVersion` 2: today's fields plus `llmCalls` (every Anthropic attempt, in
the order it finished), `historyEntry` (the history entry the run appended), `engineDigest` and
`protocolVersion`. Playlist mode appends a history entry too, its canvas fields empty, so feedback
on a playlist concept joins a record. Outside tenant mode the summary is version 1 and playlist
mode writes no history, as before.
