from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import jsonschema  # type: ignore[import-untyped]  # no stubs ship; types-jsonschema would be a new dependency (R-EX)
import pytest

from mixlab import llm, modes, run_contract
from mixlab.__main__ import main
from mixlab.config import CUSTOM_GENRES, GENRE_MAP, IGNORED_GENRES
from mixlab.worker import _FLAG_SPECS

_FILES = {
    "flags.json": "9fccbf0eeac533942559d777692599599dbca85aee9d431591dcf93382dcf316",
    "summary.schema.json": "2a86253e08b9d1c6eebeea61f6b118fa798c68f7cc4587534b26c870a86d33fe",
    "genres.json": "9e28f1e2f53391b6647bef6e8b61c13b4397cdc86282f45fb38dec951596bd7f",
    "keys.json": "448b065be3da0c728f0c2e60793a030fdad146d32033988f8717879b6f01ce4f",
}


def _bytes(name: str) -> bytes:
    return resources.files("mixlab.run_contract").joinpath(name).read_bytes()


def _serialise_config() -> dict[str, object]:
    # Copied from soltech-consulting/mixlab-multi-tenant tools/s6/count_pools.py:49-61 @ 3c376f7
    # (the function that wrote genres.json), with the imports moved to module level.
    return {
        "version": 1,
        "source": "MixLab src/mixlab/config.py:14-64 @ 5b33bda",
        "genreMap": {label: list(tags) for label, tags in GENRE_MAP.items()},
        "customGenres": {
            label: {"genres": list(cfg["genres"]), "bpmRange": list(cfg["bpm_range"]) if cfg["bpm_range"] else None}
            for label, cfg in CUSTOM_GENRES.items()
        },
        "ignoredGenres": sorted(IGNORED_GENRES),
    }


@pytest.mark.parametrize("name", sorted(_FILES))
def test_run_contract_hashes_pinned(name: str) -> None:
    assert hashlib.sha256(_bytes(name)).hexdigest() == _FILES[name]


def test_run_contract_load_returns_decoded_object() -> None:
    assert run_contract.load("keys.json")["version"] == 1


def test_flags_json_equals_flag_specs() -> None:
    flags = cast("list[dict[str, object]]", run_contract.load("flags.json")["flags"])
    from_file = [
        (f["wireKey"], f["cliFlag"], f["kind"], sorted(cast("list[str]", f.get("allowed", [])))) for f in flags
    ]
    from_engine = [(s.wire_key, s.cli_flag, s.kind, sorted(s.allowed or [])) for s in _FLAG_SPECS]
    assert from_file == from_engine


def test_genres_json_equals_config() -> None:
    expected = (json.dumps(_serialise_config(), indent=2) + "\n").encode("utf-8")
    assert _bytes("genres.json") == expected


def test_keys_json_table_is_complete() -> None:
    keys = run_contract.load("keys.json")
    codes = keys["codes"]
    assert isinstance(codes, dict)
    classical = re.compile(str(keys["classicalGrammar"]), re.IGNORECASE)
    camelot = re.compile(str(keys["camelotForm"]), re.IGNORECASE)
    assert sorted(codes) == sorted(f"{n}{r}" for n in range(1, 13) for r in "AB")
    for code, spellings in codes.items():
        assert camelot.match(code)
        assert isinstance(spellings, list)
        assert spellings
        for spelling in spellings:
            assert classical.match(str(spelling)), (code, spelling)


def test_bpm_grammar_fixture_matches_preflight_regex() -> None:
    text = (Path(__file__).parent / "fixtures" / "run_contract" / "bpm_grammar.txt").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.count("\n") == 1


_S6_STANDARD = Path(__file__).parent / "fixtures" / "run_contract" / "s6-standard.xml"
_STUB_ENV_CLEARED = (
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "CATALOG_API_URL",
    "DISCORD_BOT_TOKEN",
    "MIXLAB_STAGE0_MODEL",
    "MIXLAB_STAGE2_MODEL",
    "MIXLAB_TOKEN_BUDGET",
)


@pytest.fixture
def stub_run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A stub run on s6-standard.xml under MIXLAB_PLAYED_SOURCE=file, everything under tmp_path."""
    for var in _STUB_ENV_CLEARED:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MIXLAB_LLM_STUB", "1")
    monkeypatch.setenv("MIXLAB_PLAYED_SOURCE", "file")
    monkeypatch.setenv("MIXLAB_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("MIXLAB_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("MIXLAB_ENGINE_DIGEST", "sha256:" + "ab" * 32)
    monkeypatch.setenv("MIXLAB_PROTOCOL_VERSION", "2")
    xml_path = tmp_path / "rekordbox.xml"
    xml_path.write_bytes(_S6_STANDARD.read_bytes())
    monkeypatch.setattr("mixlab.__main__._XML_PATH", xml_path)
    monkeypatch.setattr("mixlab.__main__._HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.chdir(tmp_path)
    yield
    for var in ("MIXLAB_LLM_STUB", "MIXLAB_PLAYED_SOURCE", "MIXLAB_API_MODE"):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(modes)
    llm.RECORDER.reset()
    llm.TOKEN_BUDGET.reset()


def _written_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> dict[str, object]:
    """Run main() with argv and return the summary.json it wrote, read back from the file."""
    before = set((tmp_path / "reports").glob("*.json"))
    monkeypatch.setattr("sys.argv", ["mixlab", *argv])
    with patch("mixlab.__main__.load_dotenv"), patch("mixlab.__main__.send_report", AsyncMock()):
        main()
    (path,) = set((tmp_path / "reports").glob("*.json")) - before
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.usefixtures("stub_run_env")
def test_summary_v2_validates_against_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = run_contract.load("summary.schema.json")
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)
    genre_run = ["--genre", "techno", "--mode", "all"]
    playlist_run = ["--playlist", "Seeds/Four", "--mode", "all"]

    v1 = _written_summary(monkeypatch, tmp_path, genre_run)
    assert v1["schemaVersion"] == 1
    validator.validate(v1)

    monkeypatch.setenv("MIXLAB_API_MODE", "tenant")
    for argv in (genre_run, playlist_run):
        summary = _written_summary(monkeypatch, tmp_path, argv)
        assert summary["schemaVersion"] == 2
        assert "historyEntry" in summary
        validator.validate(summary)
        without_calls = {key: value for key, value in summary.items() if key != "llmCalls"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(without_calls, schema)
