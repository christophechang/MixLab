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

## Releasing a version

When the operator says **"deploy release"**: run the user-level `deploy-release` skill (changelog → README
staleness check → verify → commit `chore: prepare vX.Y.Z release` → push `develop` → merge `main` → tag →
GitHub release → back to `develop`). Do the steps in order and stop if any check fails. Repo specifics:

- Merge with an explicit merge commit — `git checkout main && git pull origin main && git merge --no-ff
  develop -m "Merge develop into main for vX.Y.Z"` — so each release is a marker on `main`.
- Tag the merge commit on `main` (tags and GitHub releases live on `main`), then push branch and tag.
- Keep the Releases page current — the whole back-catalogue once had to be backfilled because this step was
  skipped. Notes come from the new `CHANGELOG.md` section; confirm `gh release view vX.Y.Z` shows Latest.
- Deploy: SSH `christophechang@192.168.1.122`, then
  `cd /Users/christophechang/OpenClaw/Automations/MixLab && git checkout main && git pull origin main && git fetch --tags origin`
  (a plain pull does not fetch the release tag). Confirm the working tree is clean (untracked `output/` is
  expected) and `git describe --tags` reports `vX.Y.Z`.
- Restart the MixLab Anywhere worker — the `mixlab --worker` daemon (launchd job
  `com.changsta.mixlab-worker`) loads its poll-loop code once at start and does **not** hot-reload (claimed
  runs execute as fresh subprocesses, but restart anyway so the daemon is on current code). Graceful restart:
  `launchctl kickstart -k gui/$(id -u)/com.changsta.mixlab-worker`.
- Verify: `launchctl list | grep mixlab-worker` shows it running, a fresh "Worker online" banner in
  `~/Library/Logs/mixlab-worker/stdout.log`, and no new errors in `stderr.log`. Only report the deploy done
  once every check passes.