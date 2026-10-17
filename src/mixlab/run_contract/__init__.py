"""The run contract's data files, shipped as package data.

Copied byte for byte from ``tools/run-contract/`` in ``soltech-consulting/mixlab-multi-tenant``
(see ``tests/fixtures/run_contract/README.md`` for the commit).
"""

from __future__ import annotations

import json
from importlib import resources

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def load(name: str) -> dict[str, JsonValue]:
    """Return the named JSON file of the run contract as a decoded object."""
    text = resources.files(__name__).joinpath(name).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{name} is not a JSON object")
    return data
