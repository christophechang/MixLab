# MixLab Anywhere — remote runs, archive, and the feedback loop

**Status**: approved design (owner, 2026-07-08). This document is the single source of
truth for the cross-repo build. Every implementation ticket references a section here;
when a ticket and this doc disagree, this doc wins until amended.

## 1. Goals

- Trigger MixLab runs from anywhere (phone, holiday) via a web UI at
  **mixlab.changsta.com**, without exposing the home network.
- Archive **every run** (report, export XML, structured summary) permanently.
- Record **feedback from anywhere** — verdicts, 1–5 ratings, notes, and links from a
  concept to the published SoundCloud mix it became — and feed it back into the
  engine's existing novelty/name machinery automatically.
- Kill the Rekordbox-XML copy automation: uploading the collection through the UI *is*
  the transfer.
- Keep the Python engine unchanged in its behaviour and location (Mac mini); keep the
  API's cost model (no database — JSON in Azure Blob Storage); keep the owner's
  existing CLI workflow working byte-for-byte.

### Non-goals (v1)

- Multi-user anything. This is a single-DJ system with one shared secret.
- Public read access (deliberately deferred; see §8 — turning it on later is a config
  change, not a redesign).
- Scheduled/recurring runs from the API (the owner's OpenClaw agent already does
  scheduling well; it can POST to the runs endpoint if wanted).
- Real-time run progress streaming. Status is `queued → running → succeeded|failed`,
  polled by the SPA.

## 2. System overview

```
   phone / laptop, anywhere                Azure (existing B1 app)              home network
┌───────────────────────────┐    HTTPS   ┌──────────────────────────┐  HTTPS  ┌─────────────────────┐
│  SPA  mixlab.changsta.com │──────────▶│  Changsta .NET API        │◀────────│  Mac mini            │
│  (Cloudflare Pages,       │           │  /api/mixlab/*            │  poll,  │  MixLab engine       │
│   React+Vite+TS, private  │           │  BearerSecret auth        │  claim, │  `mixlab --worker`   │
│   repo, key in localStorage)          │  JSON+blobs in Blob Store │  push   │  (launchd keep-alive)│
└───────────────────────────┘           └──────────────────────────┘         └─────────────────────┘
```

**The direction of every arrow into the home network is outbound from the Mac mini.**
The worker polls the API for queued jobs; nothing ever connects inbound. The API is a
"dumb-ish" system of record: uploads, a job queue, artifacts, history, feedback.

### Decisions (fixed)

| Decision | Choice | Why |
|---|---|---|
| Access | Private: every `/api/mixlab/*` endpoint behind the existing `BearerSecretAttribute` pattern | Owner's data; public-read later = removing the attribute from GET endpoints |
| Persistence | Azure Blob JSON + artifact blobs, ETag optimistic concurrency | Matches `BlobMixCatalogueRepository` pattern; zero DB cost |
| Trigger | Worker **pull** (poll loop) | No inbound networking; survives any NAT/router |
| Poll cadence | 30 s default, `MIXLAB_WORKER_POLL_SECONDS` env override | Snappy enough; ~2.9k requests/day is noise on B1 |
| Worker runtime | Persistent loop under **launchd** on the Mac mini; `--worker-once` also provided for agent/cron use | Survives reboots; OpenClaw optional, not required |
| Run execution | Worker invokes the pipeline as a **subprocess** (`./mixlab <flags>`) | Crash isolation: a pipeline failure can never kill the worker loop |
| SPA stack | React + Vite + TypeScript + Tailwind, private repo `mixlab-web`, Cloudflare Pages, custom domain mixlab.changsta.com | Agent-managed; CF Pages = zero Azure slots |
| Engine changes | Strictly additive (new flags/modes/artifacts); all existing CLI paths byte-identical | 1,047-test suite keeps guarding the classic path |

## 3. Blob layout (container `mixlab`)

```
uploads/index.json                    # [{uploadId, uploadedAt, sizeBytes, label}], newest first, keep 5
uploads/{uploadId}.xml.gz             # gzipped rekordbox.xml
runs/index.json                       # [{runId, createdAt, status, genre, flagsSummary, conceptCount}], newest first
runs/{runId}/run.json                 # manifest — the mutable state document (see §5.1)
runs/{runId}/report.html              # artifact (immutable once written)
runs/{runId}/export.xml               # artifact
runs/{runId}/summary.json             # artifact (see §5.2)
history/concept-history.json          # canonical engine history (mirrors .mixlab/concept-history.json)
feedback/pending.json                 # feedback events not yet synced into history by the worker
```

Index documents follow the repo's existing pattern: read-modify-write with **ETag
`If-Match`**; on `412` re-read and retry (bounded). `CatalogConcurrencyException`
semantics apply. Artifact blobs are written once and never mutated; `run.json` is the
only per-run mutable document.

## 4. API surface (`/api/mixlab/*`, all `[BearerSecret("MixLab:ApiSecret")]`)

| # | Endpoint | Purpose |
|---|---|---|
| 1 | `POST /api/mixlab/uploads` | Upload collection XML. Accepts raw XML or gzip (`Content-Encoding: gzip` or magic-byte sniff). Stores gzipped. → `201 {uploadId, sizeBytes}` |
| 2 | `GET /api/mixlab/uploads` | List uploads (from index) |
| 3 | `GET /api/mixlab/uploads/{id}` | Stream the gzipped XML (worker download) |
| 4 | `POST /api/mixlab/runs` | Enqueue a run: `{flags: {...}, uploadId: "<id>"\|"latest"}` → `202 {runId}` (`status=queued`) |
| 5 | `GET /api/mixlab/runs?take=&skip=` | Archive list (from index) |
| 6 | `GET /api/mixlab/runs/{id}` | Run manifest |
| 7 | `GET /api/mixlab/runs/{id}/report` | `text/html` artifact |
| 8 | `GET /api/mixlab/runs/{id}/export` | `application/xml` artifact |
| 9 | `GET /api/mixlab/runs/{id}/summary` | `application/json` artifact |
| 10 | `POST /api/mixlab/worker/claim` | Atomically claim the **oldest queued** run: sets `status=running, claimedAt, workerId`. → `200 {run manifest}` or `204` when queue empty. Also **requeues stale claims** (running + claimedAt older than `MixLab:ClaimLeaseMinutes`, default 45) before claiming |
| 11 | `POST /api/mixlab/runs/{id}/complete` | Multipart: `summary.json`, `report.html`, `export.xml` → writes artifacts, `status=succeeded`. **Idempotent**: repeat for an already-succeeded run returns `200` without rewriting |
| 12 | `POST /api/mixlab/runs/{id}/fail` | `{error, logTail}` → `status=failed` |
| 13 | `GET /api/mixlab/history` / `PUT /api/mixlab/history` | Concept-history sync (ETag on GET, `If-Match` on PUT) |
| 14 | `POST /api/mixlab/runs/{id}/concepts/{conceptId}/feedback` | `{verdict?, rating?, notes?, publishedMixSlug?}` → merged into `run.json` concept AND appended to `feedback/pending.json` |
| 15 | `GET /api/mixlab/feedback/pending` / `POST /api/mixlab/feedback/ack` | Worker pulls unsynced feedback; acks by event id after merging into history |

Controller stays thin per AGENTS.md: one `MixLabController` delegating to use cases in
`Core.BusinessProcesses` (`EnqueueMixLabRunUseCase`, `ClaimMixLabRunUseCase`, …) with
blob repositories in `Infrastructure.Services.Azure`. Request body limit for uploads:
64 MB (a 2,100-track rekordbox.xml is ~15–25 MB raw, ~2 MB gzipped — SPA gzips before
upload via `CompressionStream`, so the normal case is small).

### Auth

One shared secret, configuration key `MixLab:ApiSecret`, enforced by the existing
`BearerSecretAttribute` (fails closed outside Development). SPA stores the key in
`localStorage` after the user enters it once (Settings page); worker reads
`MIXLAB_API_SECRET` from `.env`. Rotation = change config in Azure + re-enter in two
places. CORS: allow origin `https://mixlab.changsta.com` for the `/api/mixlab/*`
routes, following the existing CORS wiring pattern.

## 5. Data contracts

### 5.1 `run.json` (manifest — mutable state document)

```json
{
  "schemaVersion": 1,
  "runId": "r_20260708_1c9f",
  "createdAt": "2026-07-08T14:02:11Z",
  "status": "queued|running|succeeded|failed",
  "flags": {
    "genre": "traverse", "mode": "all", "risk": "high",
    "directions": "mixed", "intent": null, "mixLength": null,
    "resequence": false, "deep": false, "stage1Seed": null
  },
  "uploadId": "u_20260708_a1",
  "claimedAt": null, "workerId": null,
  "completedAt": null, "error": null,
  "concepts": [
    { "conceptId": "…", "title": "107 Degrees",
      "feedback": { "verdict": "played", "rating": 4, "notes": "…",
                     "publishedMixSlug": "…", "recordedAt": "…" } }
  ]
}
```

`concepts` is populated at completion from `summary.json` (id + title only; feedback
fields filled in later by endpoint 14). `publishedMixSlug` references the existing
catalogue's mix slug — the join that later powers "concepts that shipped".

### 5.2 `summary.json` (structured run artifact — emitted by the engine)

Versioned (`schemaVersion: 1`). Contents: run flags + effective seed; per concept —
**`conceptId` (identical to the id stored in the engine's `ConceptRecord` history —
see M1)**, title, `direction_type`, `arc_type`, mood, thesis, practicality breakdown,
ordered tracks (id/artist/title/bpm/key/energy/duration), transitions (mechanism,
score, blend label, risk), booth-sheet steps, validation warnings scoped to the
concept; plus run-level warnings, run notes (renames/revisions), and timing.
The single non-negotiable: **`conceptId` unification** — the uuid minted for a
concept's history record and the id in `summary.json` must be the same value, minted
once, or the feedback loop cannot join API feedback back onto history records.

### 5.3 Feedback event (`feedback/pending.json` entries)

```json
{ "eventId": "f_…", "runId": "…", "conceptId": "…",
  "verdict": "played|played_modified|rejected|unused|null",
  "rating": 4, "notes": "…", "publishedMixSlug": "…", "recordedAt": "…" }
```

## 6. Worker protocol (Python, Mac mini)

New CLI modes — **all additive**; every existing invocation is untouched:

- `mixlab --worker` — loop: claim → execute → push, sleeping `MIXLAB_WORKER_POLL_SECONDS`
  (default 30) between empty polls. Runs under launchd (`KeepAlive=true`).
- `mixlab --worker-once` — one poll cycle, then exit (for cron/agent/manual use).

One claim cycle:

1. `POST /worker/claim`. `204` → sleep. Otherwise receive manifest.
2. **Sync down**: `GET /history` → write `.mixlab/concept-history.json` (ETag cached);
   `GET /feedback/pending` → apply each event via the existing `run_feedback`
   machinery → `PUT /history` (If-Match) → `POST /feedback/ack`.
3. Download `GET /uploads/{uploadId}` → gunzip → `import/rekordbox.xml`.
4. Build argv from `flags` (strict allow-list mapping — unknown flags rejected at
   enqueue time by the API contract, and defensively ignored here) and run
   `./mixlab …` as a **subprocess** with a hard timeout (`MIXLAB_WORKER_RUN_TIMEOUT`,
   default 20 min), capturing stdout/stderr.
5. Exit 0 → gather `report.html` (report dir), `export.xml`, `summary.json` →
   `POST /runs/{id}/complete`; then `PUT /history` again (the run appended a new
   entry). Non-zero/timeout → `POST /runs/{id}/fail` with the log tail.
6. Discord delivery keeps happening inside the pipeline run itself, unchanged.

Crash safety: if the worker dies mid-run, the claim lease (§4 #10) expires and the
run is requeued; `complete` idempotency makes a double-push harmless. The worker
never holds state that isn't reconstructible from the API.

## 7. SPA (`mixlab-web`, private, Cloudflare Pages)

Pages: **Trigger** (upload XML with client-side gzip + "reuse latest"; flag form
mirroring the CLI levers; submit → runId), **Archive** (run list with status chips,
genre, date), **Run detail** (summary-driven concept cards — title/badges/tracklist/
transitions — plus embedded `report` HTML in a sandboxed iframe, download buttons for
export.xml), **Feedback** (per concept: verdict buttons, star rating, notes,
published-mix picker fed by the *public* catalogue titles endpoint), **Settings**
(API base URL + secret, stored in localStorage; "test connection"). Stack: React 18,
Vite, TypeScript strict, Tailwind; no component framework; fetch wrapper injecting
`Authorization: Bearer`. Deploy: Cloudflare Pages from the repo, custom domain
`mixlab.changsta.com`. Nothing secret ships in the bundle.

## 8. Failure modes & policies

| Failure | Handling |
|---|---|
| Worker offline | Runs queue; SPA shows `queued`; worker drains on return |
| Worker dies mid-run | Lease expiry requeues on next claim call |
| Double `complete` | Idempotent no-op |
| Index ETag conflict | Bounded re-read/retry in the repository (existing pattern) |
| Concurrent CLI + API history writes | Last-writer-wins on `PUT /history` with If-Match retry; single-user scale accepts this |
| Upload too large | 64 MB cap → 413; SPA gzips client-side so real payloads ~2 MB |
| Bad flags at enqueue | API validates against the flag allow-list → 400 before anything queues |
| Opening up later | Remove `BearerSecret` from GET endpoints (runs/report/summary), keep it on POSTs — no data migration |

## 9. Delivery phases & ticket map

Model tiers: `haiku` = mechanical, `sonnet` = standard build, `opus` = concurrency or
contract-critical. Every ticket carries the label and a `Model-tier:` body line.

- **Phase 1 — contract & engine artifacts** (unblocks everything):
  M1 `summary.json` emission + conceptId unification (sonnet) · A1 domain/contracts/
  blob repositories + indexes + ETag (sonnet) — *the two keystones; both sides then
  proceed independently.*
- **Phase 2 — parallel tracks**: MixLab M2 API client (sonnet), M3 worker loop (opus),
  M4 history/feedback sync (sonnet), M5 docs+launchd runbook (haiku) · API A2 uploads
  (sonnet), A3 queue/claim/complete state machine (opus), A4 history+feedback
  endpoints (sonnet), A5 DI/config/Bicep/deploy (sonnet), A6 CORS+limits (haiku) ·
  SPA S1 scaffold+Pages (sonnet), S2 client+auth (sonnet), S3 trigger (sonnet),
  S4 archive+detail (sonnet), S5 feedback UI (sonnet), S6 domain runbook (haiku).
- **Phase 3 — integration**: I1 end-to-end smoke runbook (haiku) — enqueue from SPA →
  worker executes → artifacts visible → feedback round-trips into the next run's
  history.

Dependency spine: M1+A1 → everything; A3 → M3; S2 → S3/S4/S5. All other pairs are
parallel-safe across repos.
