# AGENTS.md

Repo-level instructions for coding agents working in MixLab.

## Purpose
- MixLab is a Python CLI app that analyses a Rekordbox XML export, filters played tracks, generates LLM-backed mix concepts, exports Rekordbox playlists, and optionally posts results to Discord.
- Prefer small, targeted changes that preserve the current CLI workflow and existing output formats.

## Project Shape
- Application code lives in `src/mixlab/`.
- Tests live in `tests/` and usually mirror source modules.
- The main entrypoint is `src/mixlab/__main__.py`.
- The user-facing wrapper script is `./mixlab`.
- Tooling and quality rules are defined in `pyproject.toml`.

## Working Style
- Verify assumptions by reading the repo before changing code.
- Follow existing patterns in the touched module instead of introducing new architecture.
- Keep changes within the requested scope. Do not refactor unrelated modules.
- Do not add dependencies, change public CLI behaviour, or alter environment variable names unless explicitly requested.
- Preserve the current `src/` layout and existing command-line entrypoints.

## Python Standards
- `pyproject.toml` is the source of truth for formatter, lint, mypy, and pytest configuration.
- Use full type annotations on new or changed code. This repo runs mypy in strict mode.
- Keep imports and formatting Ruff-compatible rather than hand-formatting around the tools.
- Use `from __future__ import annotations` in Python modules, matching the existing codebase.

## MixLab-Specific Guardrails
- Treat Rekordbox parsing, genre mapping, playlist export, and Discord formatting as behaviour-sensitive areas: make minimal changes and preserve current output contracts unless the task requires otherwise.
- Keep external calls behind the existing boundaries:
- `client.py` for catalog API access
- `llm.py` for provider calls and prompt orchestration
- `discord_client.py` for Discord delivery
- Do not hardcode secrets, tokens, URLs, guild IDs, or channel IDs. Use environment variables and document any new ones in `.env.example` and `README.md`.
- Prefer deterministic tests for LLM/provider behaviour by mocking `httpx` calls with `respx`.

## Commands
- Setup: `./setup.sh`
- Run app: `./mixlab`
- Run a genre: `./mixlab --genre house`
- Format: `.venv/bin/python -m ruff format .`
- Lint: `.venv/bin/python -m ruff check .`
- Type-check: `.venv/bin/python -m mypy .`
- Test: `.venv/bin/python -m pytest`

## Testing Expectations
- Run relevant tests after changes. Run the full test suite when the change touches shared pipeline code or multiple modules.
- Add or update tests when behaviour changes.
- Do not let tests hit real external services; mock HTTP and provider responses.

## Documentation
- Keep documentation aligned with the real repo state.
- Prefer concise instructions that reference actual files, commands, and environment variables present in this repository.

## Commits
- Use conventional commits when creating commits.
- Do NOT add `Co-Authored-By` trailers to commit messages.
