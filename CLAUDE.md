# Claude.md — Coding Rules & Workflow

## Commits
- Do NOT add `Co-Authored-By` trailers to commit messages.

---

## Code Conventions

**`pyproject.toml` is the source of truth for all code style and quality rules.**
Do not deviate from rules defined there. Ruff enforces formatting and linting at check time; mypy enforces type correctness.

### Formatter — Ruff Format

- Line length: 120 characters (configured in `pyproject.toml`)
- Ruff format is non-negotiable — never manually reformat what Ruff controls
- Run `ruff format .` before committing

### Linter — Ruff Check (enforced rule sets)

The following rule sets are enabled in `pyproject.toml`:

#### Style & Errors (E, W — pycodestyle)
- E1: Indentation errors (tabs vs spaces, unexpected indent)
- E2: Whitespace errors (around operators, after commas, before colons)
- E3: Blank line violations (too many blank lines)
- E4: Import ordering violations
- E5: Line length violations
- E7: Statement errors (multiple statements on one line, bare `except`)
- E9: Runtime errors (syntax errors, undefined names)
- W: Warnings — trailing whitespace, blank lines at end of file

#### Pyflakes (F)
- F401: Unused imports — remove them, never suppress
- F811: Redefinition of unused name
- F841: Local variable assigned but never used

#### isort (I)
- I001: Import order — standard library first, then third-party, then local; alphabetical within each group

#### Naming (N — pep8-naming)
- N801: Class names must use `CapWords` (PascalCase)
- N802: Function names must be `snake_case`
- N803: Argument names must be `snake_case`
- N806: Variable names in functions must be `snake_case`
- N815: Class variables must not use `mixedCase`
- N816: Module-level variables must not use `mixedCase`

#### Bugbear (B)
- B006: Do not use mutable default arguments (e.g. `def f(x=[])`) — use `None` and assign inside
- B007: Unused loop control variable — prefix with `_` if intentional
- B008: Do not perform function calls in default arguments
- B023: Function definition in a loop without binding

#### Simplify (SIM)
- SIM101: Duplicate `isinstance` checks — merge with tuple
- SIM105: Use `contextlib.suppress(...)` instead of `try/except/pass`
- SIM108: Use ternary operator where it improves readability

### Disabled rules (explicitly off)
- E501 is handled by Ruff format, not flagged as a lint error
- D (pydocstring) rules — not enforced; doc comments are optional

### Naming conventions (not enforced by tooling — apply manually)

| Element | Convention | Example |
|---|---|---|
| Module / file | `snake_case` | `mix_report.py` |
| Class | `PascalCase` | `MixReportGenerator` |
| Function / method | `snake_case` | `fetch_track_data()` |
| Variable | `snake_case` | `track_count` |
| Constant (module-level) | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3` |
| Private attribute | `_snake_case` | `self._client` |
| Type alias | `PascalCase` | `TrackList = list[str]` |

---

## Type Annotations — mypy (strict)

**All code must be fully type-annotated.** mypy runs in strict mode.

### Rules
- Every function must have annotated parameters and return type
- Use `from __future__ import annotations` at the top of every file for forward-reference support
- Use built-in generics (`list[str]`, `dict[str, int]`) not `List`, `Dict` from `typing` (Python 3.10+)
- Use `X | None` instead of `Optional[X]`
- Use `X | Y` instead of `Union[X, Y]`
- Never use `Any` without a `# type: ignore` comment explaining why
- Use `TypedDict` for structured dicts; prefer dataclasses or Pydantic models over raw dicts

### mypy config (in `pyproject.toml`)
```toml
[tool.mypy]
strict = true
python_version = "3.12"
```

---

## Build & Test Workflow

### Commands
```bash
ruff format .                    # format all files
ruff check .                     # lint all files
mypy .                           # type-check all files
pytest                           # run all tests
pytest --tb=short -q             # terse output
```

### Rules
1. **Run tests after every change.** Tests must pass before any task is considered complete.
2. **New features require new unit tests.** Any added functionality must have corresponding test coverage.
3. **Changes to existing features require updated unit tests.** Modify tests to reflect changed behaviour; do not leave stale tests in place.
4. **Tests live in** `tests/` — mirror the source structure (e.g. `src/mixlab/reader.py` → `tests/test_reader.py`).
5. **Test framework:** pytest. Use `pytest.mark.parametrize` for data-driven cases.
6. **No test should rely on external services.** Mock all HTTP calls (httpx) and LLM API calls using `pytest-mock` or `respx`.

### Test conventions
- Test function names: `test_<what>_<condition>_<expected>` (e.g. `test_parse_xml_missing_field_raises_value_error`)
- Use `pytest.raises(SomeError)` for exception assertions
- Fixtures go in `conftest.py` at the appropriate directory level
- Use `respx` to mock `httpx` calls — never hit real endpoints in tests

---

## Project Structure

```
mixlab/
├── src/
│   └── mixlab/
│       ├── __init__.py
│       ├── reader.py        # XML ingestion
│       ├── client.py        # REST API calls via httpx
│       ├── reporter.py      # LLM report generation
│       └── models.py        # Pydantic models / TypedDicts
├── tests/
│   ├── conftest.py
│   ├── test_reader.py
│   ├── test_client.py
│   └── test_reporter.py
├── pyproject.toml
├── .env.example             # document all required env vars here
└── CLAUDE.md
```

---

## Key Dependencies

| Purpose | Package |
|---|---|
| HTTP client | `httpx` (async) |
| XML parsing | `lxml` |
| Data models | `pydantic` |
| LLM calls | `anthropic` (or via OpenClaw integration) |
| HTTP mocking in tests | `respx` |
| Test framework | `pytest`, `pytest-asyncio`, `pytest-mock` |

---

## pyproject.toml Baseline

```toml
[project]
name = "mixlab"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "httpx",
    "lxml",
    "pydantic",
    "anthropic",
]

[project.optional-dependencies]
dev = [
    "ruff",
    "mypy",
    "pytest",
    "pytest-asyncio",
    "pytest-mock",
    "respx",
    "lxml-stubs",
]

[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "B", "SIM"]
ignore = ["E501"]

[tool.mypy]
strict = true
python_version = "3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## Additional Constraints
- Ruff errors block completion of any task — fix them, never suppress without explicit justification
- `# noqa` comments are allowed only with a rule code and a reason (e.g. `# noqa: F401 — re-exported for public API`)
- `# type: ignore` follows the same rule — always explain why
- Do not add docstrings to methods you did not author unless specifically requested
- One class per file is strongly preferred; split if a file grows beyond ~300 lines
- All secrets (API keys, endpoints) must be loaded from environment variables — never hardcoded; document them in `.env.example`
- `httpx.AsyncClient` must be used as an async context manager (`async with httpx.AsyncClient() as client:`) — never instantiate and leave open