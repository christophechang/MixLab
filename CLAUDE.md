`pyproject.toml` is source of truth for Ruff, mypy, and pytest config. Refer to `AGENTS.md` for repo-level guidance.

Full type annotations required. `mypy --strict`, Python 3.12. `from __future__ import annotations` in every file. Built-in generics (`list[str]`, `X | None`). Avoid `Any`. Use `TypedDict`, dataclasses, or Pydantic for structured data.

Ruff errors block task completion — fix them, never suppress without a rule code and reason. Same rule for `# type: ignore`. No hardcoded secrets — load from env vars and document in `.env.example`. `httpx.AsyncClient` must be used as async context manager. Respect existing module boundaries.

Project layout:
- `src/mixlab/__main__.py` — CLI entrypoint
- `src/mixlab/reader.py` — Rekordbox XML ingestion
- `src/mixlab/client.py` — catalog/play-history API calls
- `src/mixlab/clustering.py` — genre grouping and pool selection
- `src/mixlab/llm.py` — Stage 1 and Stage 2 LLM orchestration
- `src/mixlab/playlist_exporter.py` — Rekordbox playlist XML export
- `src/mixlab/discord_client.py` — Discord delivery
- `src/mixlab/matcher.py` — played/unplayed matching
- `src/mixlab/cache.py` — cached genre availability
- `src/mixlab/config.py` — genre maps and config constants
- `src/mixlab/models.py` — Pydantic models and typed payloads
- `tests/` — mirrors src structure; fixtures in `conftest.py`

Tests must pass before any task is complete. New or changed behaviour requires new or updated tests. No external service calls in tests — mock HTTP with `respx`, LLM calls with `pytest-mock`. Test naming: `test_<what>_<condition>_<expected>`.

Commands:
```bash
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy .
.venv/bin/python -m pytest
```

Run via `./mixlab [args]` or `PYTHONPATH=.../MixLab/src .venv/bin/python -m mixlab [args]`.

## Deploy Release Process

When user says **"deploy release"**:

All work happens on `develop`; releases are cut to `main` (tags and GitHub releases live on `main`). Do the steps in order and stop if any check fails.

1. **Determine version** — inspect `CHANGELOG.md` for the next version to use (or decide based on unreleased changes: patch/minor/major). Referred to as `vX.Y.Z` below.

2. **Update CHANGELOG.md** — gather all changes that are either:
   - Uncommitted (working tree / staged), or
   - Committed but not yet pushed to `origin/develop`

   Add them under a new version header with today's date.

3. **Update README.md** — check readme for stale / incorrect details. If inconsistencies found, update the file.

4. **Commit + push develop** — stage and commit `CHANGELOG.md` (and any other uncommitted changes) with message `chore: prepare vX.Y.Z release`, then `git push origin develop`.

5. **Merge develop → main** — fast-forward is not used; keep an explicit merge commit so each release is a marker on `main`:
   ```bash
   git checkout main && git pull origin main
   git merge --no-ff develop -m "Merge develop into main for vX.Y.Z"
   ```

6. **Tag on main + push** — tag the merge commit (tags belong on `main`), then push branch and tag:
   ```bash
   git tag vX.Y.Z
   git push origin main && git push origin vX.Y.Z
   ```

7. **Create the GitHub release** — keep the Releases page current (do not let this drift; the whole back-catalogue had to be backfilled once because this step was missing). Use the new `CHANGELOG.md` section as the notes:
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" \
     --notes "$(awk '/^## vX\.Y\.Z/{f=1;next} /^## v/{f=0} f' CHANGELOG.md)"
   ```
   Confirm `gh release view vX.Y.Z` shows it as Latest.

8. **Return local checkout to develop** — `git checkout develop` so subsequent work continues on the integration branch.

9. **Deploy to production server** — SSH as `christophechang@192.168.1.122` and:

   a. **Pull the new code:**
      ```bash
      cd /Users/christophechang/OpenClaw/Automations/MixLab && git checkout main && git pull origin main
      ```
      Confirm the pull succeeded and the working tree is clean, and that `git describe --tags` reports `vX.Y.Z`.

   b. **Restart the MixLab Anywhere worker.** The `mixlab --worker` daemon (launchd job `com.changsta.mixlab-worker`) is a long-running process that polls the prod API every 30s. Its poll-loop / harness code (`worker.py`, `remote.py`, `sync.py`, `__main__.py`) is loaded once at start and does **not** hot-reload — it keeps running the old code until restarted. (Each claimed run executes as a fresh subprocess, so pipeline-only changes are picked up automatically, but always restart so the daemon itself is on current code.) The worker traps SIGTERM and finishes its current cycle before exiting, so this is a graceful restart:
      ```bash
      launchctl kickstart -k gui/$(id -u)/com.changsta.mixlab-worker
      ```

   c. **Verify the worker is healthy** — it should come back online within a few seconds:
      ```bash
      launchctl list | grep mixlab-worker                # PID non-zero (2nd col status 0)
      tail -n 3 ~/Library/Logs/mixlab-worker/stdout.log  # fresh "Worker online — polling ... every 30s" banner
      tail -n 5 ~/Library/Logs/mixlab-worker/stderr.log  # no new errors
      ```

   Only report the deploy done once every check passes: `main` pulled clean at `vX.Y.Z`, and the worker shows a fresh "Worker online" banner with no new errors.