# MixLab Anywhere — end-to-end smoke test

**Audience**: deployment owner, post-full-deployment. **Time**: ~15 minutes (first 7 checks) + optional 5 min for crash-resilience (step 8).

This document walks a single checklist per deployment — once. It verifies the API, worker, SPA, upload pipeline, run execution, feedback loop, and crash resilience. For ongoing production observability, see [worker-launchd.md](./worker-launchd.md).

**Architecture reference**: [docs/architecture/mixlab-anywhere.md](../architecture/mixlab-anywhere.md) §3–8. See that doc for blob layout, endpoint details, failure policies, and the worker protocol.

---

## Prerequisites

- `.env` file on the Mac mini: `MIXLAB_API_URL` and `MIXLAB_API_SECRET` (same secret in Azure App Service config as `MixLab:ApiSecret`)
- Worker launchd installed (via [worker-launchd.md](./worker-launchd.md) install steps)
- SPA deployed to `mixlab.changsta.com` (Cloudflare Pages from private `mixlab-web` repo)
- Azure App Service (`changsta-ai-mixrec-prod`) running the .NET API, with the `mixlab` blob container provisioned
- Test Rekordbox XML file handy (e.g. a recent export from the collection)

---

## Step 1: API online + secret validation

**Goal**: Confirm the API is reachable and bearer-token auth is enforced.

**Command (with valid secret)**:

```bash
export MIXLAB_API_SECRET="<your-secret-from-Azure-config>"
curl -X GET \
  -H "Authorization: Bearer $MIXLAB_API_SECRET" \
  "https://changsta-ai-mixrec-prod.azurewebsites.net/api/mixlab/runs?take=10&skip=0"
```

**Expected output**:
- HTTP 200
- JSON array (empty `[]` on first deployment, or a list of prior runs)

**Example**:
```json
[
  {
    "runId": "r_20260708_1c9f",
    "createdAt": "2026-07-08T14:02:11Z",
    "status": "succeeded",
    "genre": "house",
    "conceptCount": 3
  }
]
```

**Command (without secret or invalid secret)**:

```bash
curl -X GET "https://changsta-ai-mixrec-prod.azurewebsites.net/api/mixlab/runs?take=10&skip=0"
```

**Expected output**:
- HTTP 401 Unauthorized

**Troubleshooting**:
- **401 with valid secret**: Check Azure config key `MixLab:ApiSecret` matches `$MIXLAB_API_SECRET`. Portal: App Service > Configuration > Application settings.
- **Network timeout**: Check Azure App Service status in the portal; ensure the B1 instance is running.
- **200 but malformed JSON**: Check API logs via Azure portal > Log stream.

---

## Step 2: Worker online + polling

**Goal**: Confirm the worker process started, is claimed by launchd, and prints the startup marker.

**Check launchd status**:

```bash
launchctl print gui/$(id -u)/com.changsta.mixlab-worker
```

**Expected output**: A plist dump including:
- `"state" = "running"`
- `"program" = ".../mixlab"` with the full path to your repo's `.venv/bin/python`

**Tail stdout log** (the startup line):

```bash
tail -20 ~/Library/Logs/mixlab-worker/stdout.log
```

**Expected first line** (or within the last 20 lines on an already-running worker):
```
Worker online — polling https://changsta-ai-mixrec-prod.azurewebsites.net every 30s (worker_id=mixlab-worker-<hostname>)
```

(The `worker_id` includes your mini's hostname. It identifies this worker in run manifests.)

**Tail stderr log** (should be mostly empty on a healthy worker):

```bash
tail -20 ~/Library/Logs/mixlab-worker/stderr.log
```

**Expected output**: Empty, or only transport/sync messages (see Troubleshooting).

**Troubleshooting**:
- **No log files**: Logs directory not created. Run: `mkdir -p ~/Library/Logs/mixlab-worker`.
- **"Worker online" missing**: Worker crashed at startup. Check stderr for env var errors; run `./mixlab --worker-once` manually to see the error.
- **Transport error in stderr** (e.g. `sync_down transport error`): API is unreachable or `.env` secret is wrong. Fix `.env`, then restart: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist`.
- **"invalid run flags" in stderr**: A queued run carries an unknown flag. This is a contract violation; check the API's enqueue validation and the worker's `build_argv` allow-list in `src/mixlab/worker.py`.

---

## Step 3: SPA connection + auth

**Goal**: Confirm the web UI can authenticate with the API and display the settings.

**Open browser** to `https://mixlab.changsta.com` (HTTPS required; self-signed certs will fail).

**Navigate to Settings** (link in top nav).

**Paste the API secret** into the "API Secret" input field:
```
<your-secret-from-Azure-config>
```

**Click "Test connection"** button.

**Expected UI state**:
- Button feedback changes to a green checkmark or "✓ Connected" message.
- No error toast or red alert.

**If test fails**:
- **Red ✗ or "401"**: Secret is wrong or doesn't match Azure config. Verify `MixLab:ApiSecret` in the portal.
- **Network error**: Check browser devtools (F12 > Console, Network tab) for CORS errors or timeouts. Verify Cloudflare Pages deployment is live and `mixlab.changsta.com` DNS resolves to Cloudflare's IP.
- **CORS error**: Check Azure App Service CORS config includes `https://mixlab.changsta.com` for `/api/mixlab/*` routes.

**Troubleshooting**:
- **localStorage persists the bad secret**: Open DevTools > Application > Local Storage > https://mixlab.changsta.com, delete the `mixlab_api_secret` key, and re-enter.
- **Cached responses**: Hard-refresh the page (Cmd+Shift+R on macOS).

---

## Step 4: Upload collection XML

**Goal**: Confirm the upload endpoint works and the file is stored.

**On the SPA**, navigate to **Trigger** page (if not already there).

**Select a Rekordbox XML file** from your computer (recent export, e.g. `collection.xml`).

**Expected SPA output**:
- Upload progress indicator (or instant if file is small).
- A confirmation message showing `uploadId` (e.g. `u_20260708_a1`) and file size in bytes.
- A new entry in the "Recent uploads" section below with the file name and timestamp.

**Verify via curl** (the upload is now in blob storage):

```bash
export MIXLAB_API_SECRET="<your-secret>"
curl -X GET \
  -H "Authorization: Bearer $MIXLAB_API_SECRET" \
  "https://changsta-ai-mixrec-prod.azurewebsites.net/api/mixlab/uploads"
```

**Expected output**: JSON array with at least one entry:
```json
[
  {
    "uploadId": "u_20260708_a1",
    "uploadedAt": "2026-07-08T14:15:00Z",
    "sizeBytes": 2847621,
    "label": "collection.xml"
  }
]
```

**Troubleshooting**:
- **Upload fails silently**: Check SPA browser devtools (Network tab) for 413 (payload too large) or 401 (bad secret). Max is 64 MB.
- **`uploadId` doesn't appear in list**: Wait a moment (blob index updates asynchronously), then refresh. If it still doesn't appear, check Azure portal > Storage account > Blob containers > `mixlab` > `uploads/` folder.
- **Upload works but list shows old uploads**: Browser cache. Hard-refresh.

---

## Step 5: First run — queued → running → succeeded

**Goal**: Confirm the full pipeline: enqueue from SPA → worker claims → executes → uploads results.

**On the SPA Trigger page**, with the uploaded XML file now selected:

1. **Set genre** to `house` (or any genre).
2. **Optionally set mode** to `all` (default `unplayed` requires catalog API).
3. **Leave other flags default** (or adjust as desired).
4. **Click "Trigger run"** button.

**Expected SPA output**:
- A confirmation toast or modal showing `runId` (e.g. `r_20260708_1c9f`).
- Redirect to the **Run detail** page showing:
  - Status chip: **Queued** (immediately)

**Check status progression** (the SPA polls every ~10 seconds):

- Within **~30 seconds**, status should change to **Running** (worker's next poll cycle claims the run).
- Within **2–5 minutes**, status should change to **Succeeded** (depends on pipeline complexity and API upload time).

**Verify the run appeared in blob storage** (same curl pattern):

```bash
export MIXLAB_API_SECRET="<your-secret>"
curl -X GET \
  -H "Authorization: Bearer $MIXLAB_API_SECRET" \
  "https://changsta-ai-mixrec-prod.azurewebsites.net/api/mixlab/runs?take=10&skip=0"
```

**Expected**: The run you just triggered appears at the top of the list with `"status": "succeeded"`.

**Verify artifacts in blob storage** (via Azure portal):

- Portal > Storage account > Blob containers > `mixlab` > `runs/{runId}/`
- Should contain: `run.json` (manifest), `summary.json`, `report.html`, `export.xml`

**Run detail on the SPA**:
- Click the succeeded run from the Archive list.
- Expected to see:
  - Concept cards (title, direction, mood, track count) extracted from `summary.json`.
  - **"Report"** tab: embedded `report.html` in a sandboxed iframe.
  - **Download** buttons for export.xml and summary.json.

**Verify Discord delivery** (if configured):
- Your Discord channel (specified by `MIXLAB_DISCORD_CHANNEL_ID` in the pipeline) should receive the normal MixLab report message and attachments.
- If no Discord delivery: Check the pipeline's stderr log on the mini (`~/Library/Logs/mixlab-worker/stderr.log`) for Discord token errors.

**Troubleshooting**:
- **Run stuck in Queued**: Worker is not claiming. Check `~/Library/Logs/mixlab-worker/stderr.log` for `sync_down transport error` or `download transport error`. Likely causes: `.env` secret wrong, API is down, network unreachable from the mini.
- **Run stuck in Running**: Pipeline is taking longer than expected (normal for large collections on first run) or crashed. SSH to the mini and check the active processes (`ps aux | grep mixlab`). If process is gone, check `stderr.log` for a crash message. If process hangs, wait up to 20 minutes (default `MIXLAB_WORKER_RUN_TIMEOUT`); if still running, restart the worker via launchctl.
- **Run shows Failed**: Check the run detail's error message. It usually includes the last 4000 chars of pipeline stderr. Common: missing collection file, invalid flags, or LLM API error (Anthropic/Groq/etc key missing or exhausted).
- **Artifacts missing from Run detail**: Report iframe or cards not rendering. Check browser devtools for JS errors. If `summary.json` is present in blob storage but not rendering, there may be a schema version mismatch; verify `summary.json` has `"schemaVersion": 1`.

---

## Step 6: Feedback — record verdict + rating

**Goal**: Confirm feedback events can be recorded and queued for the next run.

**On the SPA Run detail page**, scroll to **Feedback** section (below the concept cards).

**For one concept**, click to expand and:
1. **Select a verdict** button (one of: `Played`, `Rejected`, `Unused`).
2. **Optionally set a 1–5 star rating**.
3. **Optionally paste a published-mix link** (e.g. a SoundCloud mix slug).
4. **Optionally add notes** in the text field.
5. **Click "Submit feedback"** or similar button.

**Expected SPA output**:
- Feedback is saved locally (localStorage) and/or to the API.
- A confirmation toast.
- The verdict/rating/notes are reflected immediately on the concept card.

**Verify feedback is pending** (synced to the API):

```bash
export MIXLAB_API_SECRET="<your-secret>"
curl -X GET \
  -H "Authorization: Bearer $MIXLAB_API_SECRET" \
  "https://changsta-ai-mixrec-prod.azurewebsites.net/api/mixlab/feedback/pending"
```

**Expected output**: JSON array with at least one feedback event:
```json
[
  {
    "eventId": "f_…",
    "runId": "r_20260708_1c9f",
    "conceptId": "…",
    "verdict": "played",
    "rating": 4,
    "notes": "Great energy arc",
    "publishedMixSlug": "house-grooves-21",
    "recordedAt": "2026-07-08T14:25:00Z"
  }
]
```

**Troubleshooting**:
- **Feedback button disabled or grayed**: SPA auth may be stale. Re-enter the secret in Settings and test connection again.
- **Feedback submitted but doesn't appear in pending list**: Wait a moment (API write delay), then re-query. If still missing, check Azure App Service logs for a 500 error on the feedback endpoint.
- **Feedback visible locally on the SPA but not in the API**: localStorage only — the feedback was not POSTed. Check browser devtools Network tab for a failed POST. If you see 401, the secret changed or is missing.

---

## Step 7: Second run — feedback applied + history synced

**Goal**: Confirm the worker syncs pending feedback into history before executing a run, and the verdict multipliers take effect.

**Trigger a second run** on the SPA (same or different genre; reuse the uploaded collection).

**Expected**: Within ~30 s, the worker claims it and starts executing.

**While the run is executing** (status = Running), SSH to the mini and tail the stderr log:

```bash
tail -f ~/Library/Logs/mixlab-worker/stderr.log
```

**Expected log lines** (as the worker processes this run):
```
sync: applied 1 verdict(s), skipped 0, acked 1
```

This indicates: 1 feedback event was applied to the concept history, 0 skipped (no concept matches), and 1 acked (marked processed in the pending list).

**After the run completes**, verify the feedback list is now empty:

```bash
export MIXLAB_API_SECRET="<your-secret>"
curl -X GET \
  -H "Authorization: Bearer $MIXLAB_API_SECRET" \
  "https://changsta-ai-mixrec-prod.azurewebsites.net/api/mixlab/feedback/pending"
```

**Expected output**: Empty array `[]` (all feedback events were acked).

**Verify history was updated** (with verdict multipliers):

**On the mini**, check the concept history file:

```bash
cat ~/.mixlab/concept-history.json | jq '.runs[-1].concepts[] | select(.feedback != null) | {title, feedback}'
```

**Expected output** (for a concept with `"verdict": "played"`):
```json
{
  "title": "107 Degrees",
  "feedback": "played"
}
```

The concept's novelty multiplier is now **1.5×** (played boost). If it were "rejected", it would be **0.25×**. These multipliers are applied during the engine's selection passes on future runs (Stage 1 concept filtering and Stage 2 deduplication).

**Verify RECENT_CONCEPTS name-avoid list** is active:

The stage 2 prompt now includes the last 8 concepts (from the most recent ~10 runs) in a name-avoid guard, so the AI will not re-generate titles that were already used. This is especially visible on subsequent runs when the same genre is re-triggered — new titles should be generated rather than re-using old names.

**Troubleshooting**:
- **"sync: applied 0"**: No feedback event was processed. Check that pending feedback is actually in the API (step 6 curl). If pending is empty but you recorded feedback, it may not have been POSTed to the API; check SPA browser devtools.
- **"no concept matching"**: The feedback event's `conceptId` doesn't match any concept in history. Check that the concept was actually from the prior run. This can happen if you recorded feedback on an old run's concept, then triggered a brand new genre; those concepts aren't in the engine's history yet.
- **History file is empty or missing**: First run didn't create it. The worker creates it on the first sync-down. If missing after step 5, check worker stderr for `sync_down failed` or file-write errors.

---

## Step 8: Crash resilience — kill worker mid-run, observe requeue

**Goal**: Confirm the worker's crash isolation and the API's lease-expiry requeue mechanism.

**Trigger a long run** (e.g. large collection, complex genre) on the SPA so there's time to interrupt it.

**Wait for status to show Running** (~30 s from trigger).

**Kill the worker**:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist
```

**Expected launchd behaviour**:
- Worker process is terminated.
- Run in the API remains in `status=running` (the claim lease is still valid).

**Check the run detail on the SPA**: Status is still **Running** (no immediate requeue).

**Wait for claim lease to expire**:

By default, `MixLab:ClaimLeaseMinutes` = 45 min. You have two options:

1. **Wait 45 minutes**: Not practical for a smoke test. Skip to step 2 or note "lease expiry requeue verified through code inspection + unit tests" in your sign-off.

2. **Lower the lease temporarily** (optional, for faster testing):
   - Portal: App Service > Configuration > Application settings.
   - Find key `MixLab:ClaimLeaseMinutes`, change value to e.g. `1` (1 minute).
   - Save and wait for the instance to restart (< 1 min).
   - (After testing, restore to `45`.)

**Restart the worker**:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.changsta.mixlab-worker.plist
```

**Wait for next claim cycle** (poll interval ~30 s, or up to 1 min if you lowered the lease).

**Expected outcome**:
- Worker claims the run again (now queued thanks to lease expiry).
- Status on SPA changes back to **Queued**.
- Worker's next cycle claims it, executes, and uploads artifacts.
- Run ends up **Succeeded** as normal.

**Verify in worker logs**:

```bash
tail -100 ~/Library/Logs/mixlab-worker/stdout.log
```

**Expected**: You should see the claim-sleep cycle repeat and eventually a new claim (status change to Running, then Succeeded).

**Troubleshooting**:
- **Run stays stuck in Running after lease expires**: Next worker poll didn't requeue it. Check Azure App Service logs for an error in the claim endpoint. Verify lease config was actually lowered (check Azure portal to confirm the setting changed).
- **Worker restarted but immediately exited**: Likely .env or launchd plist is broken. Check `launchctl print gui/$(id -u)/com.changsta.mixlab-worker` output and `stderr.log`.

**Restore lease timeout** (if you lowered it):

```
Portal > App Service > Configuration > MixLab:ClaimLeaseMinutes > value: 45 > Save.
```

---

## Summary — known-good state after smoke passes

After successfully completing steps 1–7 (and optionally 8), your deployment is live. The blob storage should contain:

```
mixlab/
├── uploads/
│   ├── index.json                     # ≥1 entry
│   └── u_<timestamp>_*.xml.gz         # ≥1 gzipped collection
├── runs/
│   ├── index.json                     # ≥2 entries (step 5 + 7)
│   └── r_<timestamp>_*/
│       ├── run.json                   # status=succeeded
│       ├── summary.json               # schemaVersion=1
│       ├── report.html                # rendered artifact
│       ├── export.xml                 # playlist XML (if requested)
├── history/
│   └── concept-history.json           # ≥2 runs; feedback applied on recent run
└── feedback/
    └── pending.json                   # [] (empty; all acked)
```

**Key files on the mini**:

```
~/.mixlab/
├── concept-history.json               # synced from API, feedback applied
└── concept-history.json.etag          # cached ETag for conflict prevention
~/Library/Logs/mixlab-worker/
├── stdout.log                         # "Worker online" startup line, poll cycles
└── stderr.log                         # sync/transport messages (mostly empty on healthy runs)
```

**Next steps**:

- Set up log rotation for `~/Library/Logs/mixlab-worker/` to prevent unbounded growth (e.g. via `newsyslog` or periodic manual truncation).
- Monitor the worker logs periodically (weekly) for transport errors that might indicate API connectivity issues.
- Document the API secret rotation process for your team (change in Azure Portal > App Service config, then update `.env` on the mini and re-enter in the SPA Settings).
- The SPA is now ready for daily use: upload collections, trigger runs, record feedback, and iterate.
