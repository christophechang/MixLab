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

1. **Determine version** — inspect `CHANGELOG.md` for the next version to use (or decide based on unreleased changes: patch/minor/major).

2. **Update CHANGELOG.md** — gather all changes that are either:
   - Uncommitted (working tree / staged), or
   - Committed but not yet pushed to `origin/develop`
   Add them under a new version header with today's date.

2. **Update README.md** — check readme to check for stale / incorrect details. If inconsistencies found, update the file.

3. **Commit changelog** — stage and commit `CHANGELOG.md` (and any other uncommitted changes) with message `chore: prepare vX.Y.Z release`.

4. **Push develop** — `git push origin develop`.

5. **Tag the release** — `git tag vX.Y.Z` then `git push origin vX.Y.Z`.

6. **Deploy to production server** — SSH as `christophechang@192.168.1.122` and run:
   ```bash
   cd /Users/christophechang/OpenClaw/Automations/MixLab && git checkout main && git pull origin main
   ```
   Confirm the pull succeeded and the working tree is clean before reporting done.