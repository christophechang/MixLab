"""Deterministic set ordering — beam-search sequencing, order scoring, swap hints.

This is the core solver behind the Mix Engine (#60). Given a set of tracks it can:

- ``optimal_order`` — search for a strong play order via bounded beam search over the
  precomputed pairwise transition matrix, with optional pinned opener/closer;
- ``score_order`` — score an *existing* order on the exact same scale, so a
  human-chosen sequence and a solver-chosen one are directly comparable;
- ``suggest_swaps`` — propose a small number of position swaps that measurably improve
  an order without ever touching the opener or closer.

Everything here is pure: no I/O, no LLM, no randomness. The single source of truth for
"how good is this order" is :func:`_build_plan`; both the search and ``score_order``
route through it, so the two can never drift onto different scales. The scoring blends
the mean transition score (via :func:`mixlab.transitions.score_transition`), an
arc-fit term that mirrors ``transitions.trace_arc``'s conservative energy signatures,
and a small endpoint-fit term that rewards a soft open and a soft close.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from mixlab.models import ArcType, Track
from mixlab.transitions import TransitionEdge, relation_label, score_transition

# Search is exact-ish for small sets; beyond this the beam is neither necessary nor
# cheap enough to justify, and the caller should pre-cluster.
_MAX_TRACKS = 40

# Arc/endpoint terms need at least this many energy readings to mean anything, matching
# ``transitions.trace_arc``.
_MIN_ARC_ENERGIES = 4

# A beam-search state: (order so far, set of used ids, running edge-score sum).
_State = tuple[tuple[str, ...], frozenset[str], float]


@dataclass(frozen=True)
class Swap:
    """A single position swap suggestion within an order (0-indexed positions)."""

    pos_a: int
    pos_b: int
    gain: float  # total_score improvement the swap yields
    reason: str  # one-line human explanation built from edge labels


@dataclass
class SequencePlan:
    """A scored play order. ``total_score`` is on a fixed 0..1-ish scale shared by every
    entry point so orders from different sources are directly comparable."""

    order: list[str]  # track ids in play order
    total_score: float  # edge_mean * 0.7 + arc_fit * 0.2 + endpoint_fit * 0.1
    edge_scores: list[float]  # len(order) - 1
    arc_fit: float  # 0..1
    endpoint_fit: float  # 0..1
    notes: list[str] = field(default_factory=list)  # per-transition observations worth surfacing


# ---------------------------------------------------------------------------
# Shared scoring primitives
# ---------------------------------------------------------------------------


def _thirds(energies: list[int]) -> tuple[float, float]:
    """Return (first-third mean, last-third mean) — same split as transitions._thirds."""
    third = max(1, len(energies) // 3)
    return statistics.fmean(energies[:third]), statistics.fmean(energies[-third:])


def _collect_energies(tracks: list[Track]) -> list[int]:
    """Energies of tracks that carry one, in order. Energy-less tracks are skipped."""
    return [t.energy for t in tracks if t.energy is not None]


def _arc_fit(energies: list[int], arc_type: ArcType | None) -> float:
    """Fraction-of-ideal fit of an energy sequence against a declared arc.

    Neutral 0.7 when there is no arc or too little energy data to judge. The per-arc
    signatures mirror ``transitions.trace_arc``'s conservative philosophy: only arcs
    with a reliable energy fingerprint get a real score, everything else stays neutral.
    """
    if arc_type is None or len(energies) < _MIN_ARC_ENERGIES:
        return 0.7

    lo, hi = min(energies), max(energies)

    if arc_type == "progressive-build":
        first_mean, last_mean = _thirds(energies)
        delta = last_mean - first_mean
        return round(min(1.0, max(0.0, (delta + 0.5) / 1.5)), 4)

    if arc_type == "build-and-drop":
        peak = hi
        rise = peak > energies[0]
        drop = energies[-1] <= peak - 2
        if rise and drop:
            return 1.0
        if rise:
            return 0.5
        return 0.3

    if arc_type in ("wave", "double-peak"):
        deltas = [energies[i + 1] - energies[i] for i in range(len(energies) - 1)]
        has_rise = any(d > 0 for d in deltas)
        has_fall = any(d < 0 for d in deltas)
        return 1.0 if (has_rise and has_fall) else 0.3

    if arc_type in ("plateau", "sustained-pressure"):
        spread = hi - lo
        if spread <= 2:
            return 1.0
        if spread >= 5:
            return 0.2
        # Linear from 1.0 at spread 2 down to 0.2 at spread 5.
        return round(1.0 - (spread - 2) / 3.0 * 0.8, 4)

    if arc_type == "front-loaded":
        first_mean, last_mean = _thirds(energies)
        return 1.0 if first_mean >= last_mean else 0.4

    # dark-to-light, light-to-dark, narrative, abstract-journey — no reliable signal.
    return 0.7


def _endpoint_fit(tracks: list[Track]) -> float:
    """Reward a soft open and a soft close; unknown energy is a small hedge, not a miss."""
    score = 0.5
    if tracks:
        first = tracks[0].energy
        last = tracks[-1].energy
        if first is None:
            score += 0.1
        elif first <= 4:
            score += 0.25
        if last is None:
            score += 0.1
        elif last <= 5:
            score += 0.25
    return round(min(1.0, score), 4)


def _edges_for_order(tracks: list[Track]) -> list[TransitionEdge]:
    return [score_transition(tracks[i], tracks[i + 1]) for i in range(len(tracks) - 1)]


def _is_notable(edge: TransitionEdge) -> bool:
    """Whether a transition is worth calling out: a tight blend, a big key jump, a clash."""
    label = edge.blend_label
    return (
        label.endswith("tight")
        or "cut or manual loop" in label
        or edge.camelot_distance > 4
        or edge.tempo_relation == "incompatible"
    )


def _edge_detail(edge: TransitionEdge, a: Track, b: Track) -> str:
    """Human phrase describing why a transition is notable, from relation/blend labels."""
    parts: list[str] = []
    if edge.tempo_relation == "incompatible":
        parts.append("tempo incompatible")
    rel = relation_label(edge, a, b)
    if rel != "straight":
        parts.append(rel)
    if edge.camelot_distance > 4:
        parts.append("unparseable key" if edge.camelot_distance == 999 else f"camelot jump {edge.camelot_distance}")
    if edge.blend_label:
        parts.append(edge.blend_label)
    return "; ".join(parts) if parts else rel


def _build_plan(tracks: list[Track], arc_type: ArcType | None) -> SequencePlan:
    """The single scoring path shared by search and ``score_order``."""
    edges = _edges_for_order(tracks)
    edge_scores = [e.score for e in edges]
    edge_mean = statistics.fmean(edge_scores) if edge_scores else 0.0
    arc_fit = _arc_fit(_collect_energies(tracks), arc_type)
    endpoint_fit = _endpoint_fit(tracks)
    total = round(edge_mean * 0.7 + arc_fit * 0.2 + endpoint_fit * 0.1, 4)

    notes: list[str] = []
    for i, edge in enumerate(edges):
        if _is_notable(edge):
            notes.append(f"{i}->{i + 1}: {_edge_detail(edge, tracks[i], tracks[i + 1])}")

    return SequencePlan(
        order=[t.track_id for t in tracks],
        total_score=total,
        edge_scores=edge_scores,
        arc_fit=arc_fit,
        endpoint_fit=endpoint_fit,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def score_order(
    track_ids: list[str],
    tracks_by_id: dict[str, Track],
    arc_type: ArcType | None = None,
) -> SequencePlan:
    """Score an existing order on the shared scale.

    Unknown ids are silently dropped (caller's responsibility to pre-filter if it cares).
    Fewer than two known ids yields a plan whose edge component is 0.0 without crashing.
    """
    tracks = [tracks_by_id[tid] for tid in track_ids if tid in tracks_by_id]
    return _build_plan(tracks, arc_type)


def optimal_order(
    tracks: list[Track],
    *,
    arc_type: ArcType | None = None,
    opener_id: str | None = None,
    closer_id: str | None = None,
    beam_width: int = 64,
) -> SequencePlan:
    """Beam-search a strong play order over ``tracks``.

    The full pairwise transition matrix is computed once (n² scores). Beam states are
    ``(order, used, edge_sum)``; the beam is pruned each step by ``(edge_sum desc,
    order tuple asc)`` so the tie-break is total and the search fully deterministic.

    A pinned ``opener_id`` seeds the single start state; otherwise every track (bar a
    pinned closer) is seeded as a possible start. A pinned ``closer_id`` is held out of
    every expansion until the order is one short of full length, so it can only land in
    the final slot. At full length each complete candidate is scored through the shared
    scorer (which applies the arc and endpoint terms) and the best ``(total_score,
    order)`` wins.
    """
    n = len(tracks)
    if n < 2:
        raise ValueError(f"optimal_order needs at least 2 tracks, got {n}")
    if n > _MAX_TRACKS:
        raise ValueError(f"optimal_order supports at most {_MAX_TRACKS} tracks, got {n}")

    by_id = {t.track_id: t for t in tracks}
    ids = [t.track_id for t in tracks]
    if opener_id is not None and opener_id not in by_id:
        raise ValueError(f"opener_id {opener_id!r} is not in the track set")
    if closer_id is not None and closer_id not in by_id:
        raise ValueError(f"closer_id {closer_id!r} is not in the track set")

    # Precompute every directed edge score exactly once.
    edge_score: dict[tuple[str, str], float] = {}
    for a in tracks:
        for b in tracks:
            if a.track_id != b.track_id:
                edge_score[(a.track_id, b.track_id)] = score_transition(a, b).score

    starts = [opener_id] if opener_id is not None else [tid for tid in ids if tid != closer_id]
    beam: list[_State] = [((s,), frozenset({s}), 0.0) for s in starts]

    for _depth in range(1, n):
        candidates: list[_State] = []
        for order, used, esum in beam:
            last = order[-1]
            at_penultimate = len(order) == n - 1
            for tid in ids:
                if tid in used:
                    continue
                # A pinned closer may only be placed as the final track.
                if closer_id is not None and tid == closer_id and not at_penultimate:
                    continue
                candidates.append((order + (tid,), used | {tid}, esum + edge_score[(last, tid)]))
        # Prune deterministically: highest edge_sum first, order tuple as total tie-break.
        candidates.sort(key=lambda st: (-st[2], st[0]))
        beam = candidates[:beam_width]

    best: SequencePlan | None = None
    for order, _used, _esum in beam:
        plan = _build_plan([by_id[tid] for tid in order], arc_type)
        if (
            best is None
            or plan.total_score > best.total_score
            or (plan.total_score == best.total_score and tuple(plan.order) < tuple(best.order))
        ):
            best = plan
    assert best is not None  # beam is non-empty because n >= 2
    return best


def suggest_swaps(
    track_ids: list[str],
    tracks_by_id: dict[str, Track],
    arc_type: ArcType | None = None,
    *,
    max_swaps: int = 2,
    min_gain: float = 0.05,
) -> list[Swap]:
    """Greedily propose up to ``max_swaps`` improving position swaps.

    The opener and closer positions are sacred — never moved. Each round tries every
    interior pair ``(i, j)`` and applies the single best swap whose ``total_score`` gain
    is at least ``min_gain``, then repeats. Ties break by ``(gain desc, i asc, j asc)``.
    Returns ``[]`` when no swap clears the bar.
    """
    current = [tid for tid in track_ids if tid in tracks_by_id]
    swaps: list[Swap] = []

    for _ in range(max_swaps):
        n = len(current)
        if n < 4:  # need at least two interior positions to swap
            break
        baseline = score_order(current, tracks_by_id, arc_type).total_score

        best_gain: float | None = None
        best_ij: tuple[int, int] | None = None
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                swapped = list(current)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                gain = round(score_order(swapped, tracks_by_id, arc_type).total_score - baseline, 4)
                if gain < min_gain:
                    continue
                # Ascending (i, j) iteration + strict-greater replace keeps the lowest
                # (i, j) among equal gains, satisfying the tie-break.
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_ij = (i, j)

        if best_gain is None or best_ij is None:
            break

        i, j = best_ij
        pre_tracks = [tracks_by_id[tid] for tid in current]
        reason = _swap_reason(pre_tracks, i, j)
        current[i], current[j] = current[j], current[i]
        swaps.append(Swap(pos_a=i, pos_b=j, gain=best_gain, reason=reason))

    return swaps


def _swap_reason(pre_tracks: list[Track], i: int, j: int) -> str:
    """One-line reason naming the worst pre-swap edge the swap addresses."""
    n = len(pre_tracks)
    # Edges touching either moved position (edge p connects positions p and p+1).
    affected = sorted({p for p in (i - 1, i, j - 1, j) if 0 <= p <= n - 2})
    worst_pos = min(affected, key=lambda p: score_transition(pre_tracks[p], pre_tracks[p + 1]).score)
    a, b = pre_tracks[worst_pos], pre_tracks[worst_pos + 1]
    detail = _edge_detail(score_transition(a, b), a, b)
    return f"resolves {worst_pos}->{worst_pos + 1} ({detail})"
