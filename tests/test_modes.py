from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from mixlab import modes

_VARS = ("MIXLAB_API_MODE", "MIXLAB_LLM_STUB", "MIXLAB_PLAYED_SOURCE", "MIXLAB_KEY_NOTATION")


@pytest.fixture(autouse=True)
def _restore_modes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    importlib.reload(modes)


def test_modes_defaults_with_nothing_set() -> None:
    importlib.reload(modes)

    assert modes.TENANT_MODE is False
    assert modes.STUB_MODE is False
    assert modes.played_source() == "catalog"
    assert modes.key_notation() == "camelot"


def test_modes_tenant_and_stub_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_MODE", "tenant")
    monkeypatch.setenv("MIXLAB_LLM_STUB", "1")
    monkeypatch.setenv("MIXLAB_PLAYED_SOURCE", "file")
    monkeypatch.setenv("MIXLAB_KEY_NOTATION", "any")
    importlib.reload(modes)

    assert modes.TENANT_MODE is True
    assert modes.STUB_MODE is True
    assert modes.played_source() == "file"
    assert modes.key_notation() == "any"


def test_modes_other_flag_values_are_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXLAB_API_MODE", "TENANT")
    monkeypatch.setenv("MIXLAB_LLM_STUB", "true")
    importlib.reload(modes)

    assert modes.TENANT_MODE is False
    assert modes.STUB_MODE is False


def test_modes_invalid_values_warn_and_use_defaults(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MIXLAB_PLAYED_SOURCE", "database")
    monkeypatch.setenv("MIXLAB_KEY_NOTATION", "open")
    importlib.reload(modes)

    err = capsys.readouterr().err
    assert modes.played_source() == "catalog"
    assert modes.key_notation() == "camelot"
    assert err.count("MIXLAB_PLAYED_SOURCE") == 1
    assert err.count("MIXLAB_KEY_NOTATION") == 1


def test_modes_imports_nothing_of_mixlab() -> None:
    tree = ast.parse(Path(modes.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("mixlab") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or "").startswith("mixlab")
    held = [name for name, value in vars(modes).items() if getattr(value, "__name__", "").startswith("mixlab")]
    assert held == []
