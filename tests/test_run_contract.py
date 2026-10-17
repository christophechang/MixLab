from __future__ import annotations

import hashlib
import json
import re
from importlib import resources
from pathlib import Path
from typing import cast

import pytest

from mixlab import run_contract
from mixlab.config import CUSTOM_GENRES, GENRE_MAP, IGNORED_GENRES
from mixlab.worker import _FLAG_SPECS

_FILES = {
    "flags.json": "9fccbf0eeac533942559d777692599599dbca85aee9d431591dcf93382dcf316",
    "summary.schema.json": "740d2990bddfd67ecdd64dd2eb2f69766f3931f2693999a5a4255a84d14ef018",
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
