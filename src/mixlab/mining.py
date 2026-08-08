"""Conjunction mining — deterministic 'found' directions (spec 2026-08-08).

Pure and deterministic: no I/O, no RNG. Extracts atomic predicates from a
genre pool, scans cross-kind pairs, gates on support/lift, prunes subsumed
pairs, and shortlists by lift for scoring in directions._shape_field.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mixlab.clustering import camelot_compatible
from mixlab.models import Track

if TYPE_CHECKING:
    from mixlab.directions import Direction  # avoids the module-load cycle at runtime

_GATE_DEFAULT = 8
_GATE_REMIXER = 5
# A predicate covering more than 70% of the pool is excluded from pairing — it rides
# along with any partner and mines nothing. Held as an exact fraction and applied as
# `_COVERAGE_CAP_DEN * support <= _COVERAGE_CAP_NUM * pool_size` rather than a float
# 0.70, because float rounding misjudges exact-boundary pool sizes (63/90 == 0.70
# exactly, but 0.70 * 90 == 62.99999999999999 in IEEE-754 double precision).
_COVERAGE_CAP_NUM = 7
_COVERAGE_CAP_DEN = 10

# Must equal directions.MIN_DIRECTION_POOL — a pair too small to build is not worth
# mining. Deliberately duplicated rather than imported: directions imports this module
# at module load (it owns the combined named+mined field), so a module-level
# `from mixlab.directions import MIN_DIRECTION_POOL` here would close the import cycle.
# The lazy in-function imports below exist for the same reason, but a module-level
# constant cannot use that escape hatch. tests/test_mining.py asserts the two stay equal.
MIN_SUPPORT = 15
MIN_LIFT = 1.3
_SUBSUME_JACCARD = 0.9
SHORTLIST = 12


@dataclass(frozen=True)
class Predicate:
    kind: str
    value: str
    namable: bool
    track_ids: frozenset[str]


@dataclass(frozen=True)
class MinedPair:
    a: Predicate
    b: Predicate  # (a.kind, a.value) < (b.kind, b.value)
    support: int
    lift: float
    member_ids: tuple[str, ...]


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
        if len(ids) >= _gate(k) and _COVERAGE_CAP_DEN * len(ids) <= _COVERAGE_CAP_NUM * pool_size
    ]
    return sorted(out, key=lambda p: (p.kind, p.value))


def _jaccard_sets(a: frozenset[str], b: frozenset[str]) -> float:
    """Set overlap, used for the subsumption check in :func:`scan_pairs`.

    Empty-vs-empty is 0.0 here, where ``directions._jaccard`` returns 1.0. The
    questions differ: that one asks "are these two candidate pools the same set of
    tracks" (two empty pools are), this one asks "is this pair's member set just
    its parent predicate restated" — and an empty overlap restates nothing. Don't
    unify the two without re-reading both call sites.
    """
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def scan_pairs(predicates: list[Predicate], pool_size: int) -> list[MinedPair]:
    """Gated, pruned, shortlisted cross-kind conjunctions. Deterministic.

    Gates: cross-kind only, support >= :data:`MIN_SUPPORT`, at least one namable
    side, lift >= :data:`MIN_LIFT`. A pair whose member set is near-identical to
    either parent (Jaccard > :data:`_SUBSUME_JACCARD`) is subsumed and dropped —
    it says nothing the single predicate did not already say. Returns at most
    :data:`SHORTLIST` pairs, strongest lift first.
    """
    if pool_size <= 0:
        return []
    ordered = sorted(predicates, key=lambda p: (p.kind, p.value))
    pairs: list[MinedPair] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if a.kind == b.kind or not (a.namable or b.namable):
                continue
            members = a.track_ids & b.track_ids
            if len(members) < MIN_SUPPORT:
                continue
            lift = len(members) * pool_size / (len(a.track_ids) * len(b.track_ids))
            if lift < MIN_LIFT:
                continue
            if (
                _jaccard_sets(members, a.track_ids) > _SUBSUME_JACCARD
                or _jaccard_sets(members, b.track_ids) > _SUBSUME_JACCARD
            ):
                continue
            pairs.append(MinedPair(a=a, b=b, support=len(members), lift=lift, member_ids=tuple(sorted(members))))
    pairs.sort(key=lambda p: (-p.lift, p.a.value, p.b.value, p.member_ids))
    return pairs[:SHORTLIST]


_MECH_NOUN = {"bpm_regime": "tempo", "key_hood": "harmonic"}


def _kind_noun(kind: str) -> str:
    return _MECH_NOUN.get(kind, kind)


def mine_pool(pool: list[Track]) -> list[Direction]:
    """Extract -> scan -> materialise. feasibility stays 0.0 for the field scorer."""
    from mixlab.directions import (  # local import avoids the module-load cycle
        MAX_DIRECTION_POOL,
        Direction,
        _freshness,
        _log_lift,
        _path_feasible,
        _rank,
    )

    by_id = {t.track_id: t for t in pool}
    out: list[Direction] = []
    for pair in scan_pairs(extract_predicates(pool), len(pool)):
        members = [by_id[i] for i in pair.member_ids]
        shipped = _rank(members)[:MAX_DIRECTION_POOL]
        ok, _ratio = _path_feasible(shipped)
        if not ok:
            continue
        namables = [p for p in (pair.a, pair.b) if p.namable]
        mech = [p for p in (pair.a, pair.b) if not p.namable]
        if len(namables) == 2:
            title = f"Found: {namables[0].value} × {namables[1].value}"
            where = f"where {namables[0].value} meets {namables[1].value}"
        else:
            unit = " BPM" if mech[0].kind == "bpm_regime" else ""
            title = f"Found: {namables[0].value}"
            where = (
                f"where {namables[0].value} clusters in one "
                f"{_kind_noun(mech[0].kind)} pocket (around {mech[0].value}{unit})"
            )
        sel = f", of which the {len(shipped)} most central are selected" if pair.support > len(shipped) else ""
        anchors = "; ".join(f"{t.artist} — {t.title}" for t in shipped[:3])
        brief = (
            f"FOUND SET. Play the corner of the crate {where} as a scene, not a "
            f"coincidence — treat the conjunction as the thesis and let its shared "
            f"sound carry the set. Evidence: {pair.support} tracks sit in this "
            f"overlap, {pair.lift:.1f}x denser than chance{sel}. Anchors: {anchors}."
        )
        mood = f"{_kind_noun(pair.a.kind)} x {_kind_noun(pair.b.kind)}"
        out.append(
            Direction(
                direction_type="found",
                title=title,
                mood=mood,
                track_ids=[t.track_id for t in shipped],
                brief=brief,
                feasibility=0.0,
                identity=round(_log_lift(pair.lift), 4),
                freshness=round(_freshness(shipped, pool), 4),
            )
        )
    return out
