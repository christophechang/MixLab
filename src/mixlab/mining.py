"""Conjunction mining — deterministic 'found' directions (spec 2026-08-08).

Pure and deterministic: no I/O, no RNG. Extracts atomic predicates from a
genre pool, scans cross-kind pairs, gates on support/lift, prunes subsumed
pairs, and shortlists by lift for scoring in directions._score_field.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from mixlab.clustering import camelot_compatible
from mixlab.models import Track

_GATE_DEFAULT = 8
_GATE_REMIXER = 5
_COVERAGE_CAP = 0.70  # documentation only — enforced via integer arithmetic below to avoid
# float rounding at exact-boundary pool sizes (e.g. 63/90 == 0.70 exactly, but
# 0.70 * 90 == 62.99999999999999 in IEEE-754 double precision).


@dataclass(frozen=True)
class Predicate:
    kind: str
    value: str
    namable: bool
    track_ids: frozenset[str]


def _gate(kind: str) -> int:
    return _GATE_REMIXER if kind == "remixer" else _GATE_DEFAULT


def extract_predicates(pool: list[Track]) -> list[Predicate]:
    """All qualifying atomic predicates over ``pool``, sorted (kind, value)."""
    if not pool:
        return []
    groups: dict[tuple[str, str, bool], set[str]] = {}

    def _add(kind: str, value: str, namable: bool, track: Track) -> None:
        groups.setdefault((kind, value, namable), set()).add(track.track_id)

    for t in pool:
        if t.label:
            _add("label", t.label, True, t)
        if t.year is not None and t.year > 0:
            lo = t.year - t.year % 5
            _add("era", f"{lo}-{lo + 4}", True, t)
        for tag in t.tags:
            _add("tag", tag.lower(), True, t)
        if t.remixer:
            _add("remixer", t.remixer, True, t)
        if t.energy is not None:
            band = "low" if t.energy <= 4 else "high" if t.energy >= 6 else "mid"
            _add("energy", band, True, t)

    # Mechanical kinds — minable, never namable.
    from mixlab.directions import _tempo_regimes  # local import avoids cycle at module load

    usable = sorted((t for t in pool if t.bpm > 0), key=lambda t: (t.bpm, t.track_id))
    for regime in _tempo_regimes(usable):
        peak = round(statistics.median(t.bpm for t in regime))
        for t in regime:
            _add("bpm_regime", str(peak), False, t)
    for key in sorted({t.camelot_key for t in pool if t.camelot_key}):
        hood = [t for t in pool if camelot_compatible(key, t.camelot_key)]
        for t in hood:
            _add("key_hood", key, False, t)

    pool_size = len(pool)
    out = [
        Predicate(kind=k, value=v, namable=n, track_ids=frozenset(ids))
        for (k, v, n), ids in groups.items()
        if len(ids) >= _gate(k) and 10 * len(ids) <= 7 * pool_size
    ]
    return sorted(out, key=lambda p: (p.kind, p.value))
