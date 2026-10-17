"""Run-mode switches, read from the environment once at import.

A leaf module: it imports nothing of ``mixlab`` so every other module may import it.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

PlayedSource = Literal["catalog", "file"]
KeyNotation = Literal["camelot", "any"]


def _read_played_source() -> PlayedSource:
    value = os.environ.get("MIXLAB_PLAYED_SOURCE", "catalog")
    if value == "file":
        return "file"
    if value != "catalog":
        print(f"warning: MIXLAB_PLAYED_SOURCE={value!r} is not 'catalog' or 'file'; using 'catalog'", file=sys.stderr)
    return "catalog"


def _read_key_notation() -> KeyNotation:
    value = os.environ.get("MIXLAB_KEY_NOTATION", "camelot")
    if value == "any":
        return "any"
    if value != "camelot":
        print(f"warning: MIXLAB_KEY_NOTATION={value!r} is not 'camelot' or 'any'; using 'camelot'", file=sys.stderr)
    return "camelot"


TENANT_MODE: bool = os.environ.get("MIXLAB_API_MODE") == "tenant"
STUB_MODE: bool = os.environ.get("MIXLAB_LLM_STUB") == "1"
_PLAYED_SOURCE: PlayedSource = _read_played_source()
_KEY_NOTATION: KeyNotation = _read_key_notation()


def played_source() -> PlayedSource:
    """Where played history comes from: the catalog API (default) or the uploaded file."""
    return _PLAYED_SOURCE


def key_notation() -> KeyNotation:
    """How the reader reads Tonality: copied as written (default) or normalised to Camelot as the API does."""
    return _KEY_NOTATION
