"""Pure tempo/transition mechanics — no I/O, no LLM.

MixLab historically modelled tempo compatibility as absolute-BPM proximity. That
breaks for half/double-time mixing (86↔172), which is foundational for the
DnB/jungle/UKG material in this library: a locked halftime blend would otherwise
read as a massive BPM-jump violation. This module gives the rest of the pipeline a
vocabulary of *tempo relations* (straight, halftime, double, 3:4, 4:3) plus a
composite mixability score, and a per-track arc tracer that verifies a declared
``arc_type`` against the actual energy sequence.

Everything here is deterministic and side-effect free so it can be reused by the
Stage 2 prompt surface, the validators, and the practicality scorer without pulling
in network or model dependencies.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Literal

from mixlab.clustering import camelot_distance
from mixlab.models import MixConcept, Track

TempoRelation = Literal["straight", "halftime", "double", "three_four", "four_three", "incompatible"]

# ±6% pitch adjustment is the practical ceiling on a CDJ/turntable before the
# material audibly warps. A ratio move whose residual stretch exceeds this window is
# not actually mixable, so it is treated as incompatible.
PITCH_WINDOW = 0.06

# Camelot key parser: number 1–12 plus ring letter A/B.
_CAMELOT_RE = re.compile(r"^(\d{1,2})([AB])$", re.IGNORECASE)

# Ordered candidate ratios. Straight is tried first so an exact tempo match always
# wins; the named ratio moves follow.
_RATIOS: tuple[tuple[TempoRelation, float], ...] = (
    ("straight", 1.0),
    ("halftime", 0.5),
    ("double", 2.0),
    ("three_four", 0.75),
    ("four_three", 4.0 / 3.0),
)

_COMPATIBLE_NON_STRAIGHT: frozenset[TempoRelation] = frozenset({"halftime", "double", "three_four", "four_three"})


@dataclass(frozen=True)
class TransitionEdge:
    from_id: str
    to_id: str
    tempo_relation: TempoRelation
    tempo_stretch: float  # fractional pitch adjustment needed after ratio, e.g. 0.031 = 3.1%
    camelot_distance: int  # from clustering.camelot_distance
    is_energy_lift: bool  # +1 Camelot same ring (with wrap 12→1), the classic energy boost move
    energy_delta: int | None  # b.energy - a.energy when both present
    tag_overlap: float  # Jaccard over lowercased tag sets; 0.0 when either side has no tags
    score: float  # 0..1 composite mixability
    # Mix-point headroom (#59). None when either track lacks the cue data needed to
    # compute it — the composite score then falls back to the pre-#59 weights exactly.
    blend_headroom: float | None = None
    blend_label: str = ""


def _parse_camelot(key: str) -> tuple[int, str] | None:
    m = _CAMELOT_RE.match(key)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper()


def tempo_relation(a_bpm: float, b_bpm: float) -> tuple[TempoRelation, float]:
    """Classify the tempo move from ``a_bpm`` to ``b_bpm``.

    Tries straight (1:1), halftime, double, 3:4, 4:3 in order. For each ratio the
    residual stretch is ``|b_bpm - a_bpm*ratio| / (a_bpm*ratio)``; the first ratio
    whose stretch is within :data:`PITCH_WINDOW` wins. When none fit, returns
    ``("incompatible", straight_stretch)``. Non-positive tempos are incompatible.
    """
    if a_bpm <= 0 or b_bpm <= 0:
        return "incompatible", 1.0
    straight_stretch = abs(b_bpm - a_bpm) / a_bpm
    for relation, ratio in _RATIOS:
        target = a_bpm * ratio
        stretch = abs(b_bpm - target) / target
        if stretch <= PITCH_WINDOW:
            return relation, stretch
    return "incompatible", straight_stretch


def _tempo_component(relation: TempoRelation, stretch: float) -> float:
    """Tempo contribution to the composite score in [0, 1]."""
    if relation == "incompatible":
        return 0.0
    # 1.0 at stretch 0, decaying linearly to 0.6 at the pitch window.
    clamped = min(stretch, PITCH_WINDOW)
    base = 1.0 - (clamped / PITCH_WINDOW) * 0.4
    if relation in _COMPATIBLE_NON_STRAIGHT:
        return base * 0.9
    return base


def _harmonic_component(distance: int) -> float:
    if distance == 999:
        return 0.3
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.9
    if distance == 2:
        return 0.6
    if distance == 3:
        return 0.4
    return 0.15


def _energy_component(delta: int | None) -> float:
    if delta is None:
        return 0.7
    abs_delta = abs(delta)
    if abs_delta <= 1:
        return 1.0
    if abs_delta == 2:
        return 0.8
    return 0.5


def _tag_component(a_tags: list[str], b_tags: list[str], overlap: float) -> float:
    if a_tags and b_tags:
        return 0.5 + 0.5 * overlap
    return 0.7


def _tag_overlap(a_tags: list[str], b_tags: list[str]) -> float:
    a_set = {t.lower() for t in a_tags}
    b_set = {t.lower() for t in b_tags}
    if not a_set or not b_set:
        return 0.0
    union = a_set | b_set
    return len(a_set & b_set) / len(union)


def blend_headroom(a: Track, b: Track) -> tuple[float | None, str]:
    """Score how much runway the outgoing track's outro gives the incoming track's intro.

    ``a`` is the outgoing track (its outro bars are the supply), ``b`` is the incoming
    track (its intro bars are the demand). Returns ``(None, "")`` whenever either track
    lacks the mix-point data needed to compute a ratio — callers then fall back to the
    pre-#59 scoring weights untouched.

    A short outro is not a dead transition, just a tighter one — the label always
    frames it as something a DJ can work around (a manual loop or a cut), never as
    "unmixable".
    """
    if a.mix_points is None or b.mix_points is None:
        return None, ""
    outro = a.mix_points.outro_bars
    intro = b.mix_points.intro_bars
    if outro is None or intro is None:
        return None, ""

    ratio = outro / max(intro, 1.0)
    if ratio >= 1.5:
        score = 1.0
    elif ratio >= 1.0:
        score = 0.7 + (ratio - 1.0) / 0.5 * 0.3
    elif ratio >= 0.5:
        score = 0.3 + (ratio - 0.5) / 0.5 * 0.4
    else:
        score = 0.15

    loop_bonus = False
    if a.duration_secs is not None and a.mix_points.loops:
        final_third_start = a.duration_secs * (2.0 / 3.0)
        loop_bonus = any(loop_start >= final_third_start for loop_start, _loop_end in a.mix_points.loops)
    if loop_bonus:
        score = min(1.0, score + 0.15)

    score = round(score, 4)

    if score >= 0.7:
        suffix = ""
    elif score >= 0.3:
        suffix = " — tight"
    else:
        suffix = " — cut or manual loop likely"

    label = f"{outro:.0f} bars out / {intro:.0f} in{suffix}"
    if loop_bonus:
        label += " (loop zone available)"

    return score, label


def _is_energy_lift(a_key: str, b_key: str) -> bool:
    """Same-ring +1 Camelot move (wrapping 12→1) — the classic energy-boost blend."""
    pa = _parse_camelot(a_key)
    pb = _parse_camelot(b_key)
    if pa is None or pb is None:
        return False
    num_a, ring_a = pa
    num_b, ring_b = pb
    if ring_a != ring_b:
        return False
    return num_b == (num_a % 12) + 1


def score_transition(a: Track, b: Track) -> TransitionEdge:
    """Score the ordered transition a → b, returning a fully-populated edge."""
    relation, stretch = tempo_relation(a.bpm, b.bpm)
    cam_dist = camelot_distance(a.camelot_key, b.camelot_key)
    energy_delta = b.energy - a.energy if a.energy is not None and b.energy is not None else None
    overlap = _tag_overlap(a.tags, b.tags)

    tempo = _tempo_component(relation, stretch)
    harmonic = _harmonic_component(cam_dist)
    energy = _energy_component(energy_delta)
    tags = _tag_component(a.tags, b.tags, overlap)
    headroom, blend_label = blend_headroom(a, b)

    # Hard zero when tempo is incompatible — no amount of harmonic/energy fit rescues
    # a transition that cannot actually be beat-matched.
    if relation == "incompatible":
        score = 0.0
    elif headroom is not None:
        # Mix-point data available (#59): fold blend headroom into the composite and
        # rebalance weights to make room for it.
        score = tempo * 0.35 + harmonic * 0.25 + headroom * 0.20 + energy * 0.10 + tags * 0.10
    else:
        # No mix-point data on one or both sides — keep the pre-#59 weights byte-stable
        # for cueless libraries.
        score = tempo * 0.4 + harmonic * 0.3 + energy * 0.15 + tags * 0.15

    return TransitionEdge(
        from_id=a.track_id,
        to_id=b.track_id,
        tempo_relation=relation,
        tempo_stretch=round(stretch, 4),
        camelot_distance=cam_dist,
        is_energy_lift=_is_energy_lift(a.camelot_key, b.camelot_key),
        energy_delta=energy_delta,
        tag_overlap=round(overlap, 4),
        score=round(score, 4),
        blend_headroom=headroom,
        blend_label=blend_label,
    )


def _is_boring_edge(edge: TransitionEdge) -> bool:
    """A plain edge Stage 2 already assumes is fine — carries no non-obvious mechanism."""
    return (
        edge.tempo_relation == "straight"
        and edge.camelot_distance <= 1
        and (edge.energy_delta is None or abs(edge.energy_delta) <= 1)
    )


def strong_edges(
    tracks: list[Track],
    *,
    top_n: int = 8,
    min_score: float = 0.55,
) -> list[TransitionEdge]:
    """Return the strongest, most *informative* transition edges over ``tracks``.

    Scores every ordered pair (a ≠ b), drops edges below ``min_score`` and plain
    "boring" edges (straight tempo, adjacent key, flat energy) that tell Stage 2
    nothing it does not already assume. Symmetric pairs are deduped keeping the
    higher-scoring direction (ties broken lexicographically). Deterministic.
    """
    edges: list[TransitionEdge] = []
    for a in tracks:
        for b in tracks:
            if a.track_id == b.track_id:
                continue
            edge = score_transition(a, b)
            if edge.score < min_score:
                continue
            if _is_boring_edge(edge):
                continue
            edges.append(edge)

    edges.sort(key=lambda e: (-e.score, e.from_id, e.to_id))

    deduped: list[TransitionEdge] = []
    seen: set[frozenset[str]] = set()
    for edge in edges:
        pair = frozenset({edge.from_id, edge.to_id})
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(edge)

    return deduped[:top_n]


def relation_label(edge: TransitionEdge, a: Track, b: Track) -> str:
    """Human-readable mechanism string for prompts, e.g. ``halftime lock 172→86``.

    Requires the two tracks because the edge itself does not carry BPMs/keys — the
    caller always has them on hand.
    """
    parts: list[str] = []
    ratio_labels: dict[TempoRelation, str] = {
        "halftime": "halftime lock",
        "double": "double-time lock",
        "three_four": "3:4 shuffle",
        "four_three": "4:3 push",
    }
    if edge.tempo_relation in ratio_labels:
        parts.append(f"{ratio_labels[edge.tempo_relation]} {round(a.bpm)}→{round(b.bpm)}")
    if edge.is_energy_lift:
        parts.append(f"energy lift {a.camelot_key}→{b.camelot_key}")
    if not parts:
        return "straight"
    return ", ".join(parts)


# arc_type values with a reliable energy signature. Everything else (dark-to-light,
# light-to-dark, narrative, abstract-journey) is never flagged.
_MIN_ARC_ENERGIES = 4


def _thirds(energies: list[int]) -> tuple[float, float]:
    """Return (first-third mean, last-third mean) of an energy sequence."""
    third = max(1, len(energies) // 3)
    first_mean = statistics.fmean(energies[:third])
    last_mean = statistics.fmean(energies[-third:])
    return first_mean, last_mean


def trace_arc(concept: MixConcept, tracks_by_id: dict[str, Track]) -> str | None:
    """Verify a concept's declared ``arc_type`` against its actual energy sequence.

    Returns ``None`` when the arc_type is absent, when fewer than four tracks carry
    energy, or when the sequence matches the declared arc. Otherwise returns a
    precise, factual mismatch message. Conservative by design — only arcs with a
    clear energy signature are checked, and each check prefers ``None`` over a false
    accusation.
    """
    arc = concept.arc_type
    if arc is None:
        return None

    energies = [
        t.energy for tid in concept.track_ids if (t := tracks_by_id.get(tid)) is not None and t.energy is not None
    ]
    if len(energies) < _MIN_ARC_ENERGIES:
        return None

    lo, hi = min(energies), max(energies)

    if arc == "progressive-build":
        # Must keep climbing: the final third should sit clearly above the first.
        first_mean, last_mean = _thirds(energies)
        if last_mean < first_mean + 0.5:
            return f"declared progressive-build; energy falls {energies[0]}→{energies[-1]} instead of building"
        return None

    if arc == "build-and-drop":
        # Builds to a peak then releases — the load-bearing signature is the drop, not a
        # rising last third (the drop itself lands in the final third of an arch). Flag only
        # when the set never actually comes down off its peak.
        peak = hi
        final = energies[-1]
        if final > peak - 2:
            return f"declared build-and-drop; final energy {final} never drops from peak {peak}"
        return None

    if arc in ("wave", "double-peak"):
        non_decreasing = all(energies[i + 1] >= energies[i] for i in range(len(energies) - 1))
        non_increasing = all(energies[i + 1] <= energies[i] for i in range(len(energies) - 1))
        if hi > lo and non_decreasing:
            return f"declared {arc}; energy is monotonic rising {lo}→{hi}"
        if hi > lo and non_increasing:
            return f"declared {arc}; energy is monotonic falling {hi}→{lo}"
        return None

    if arc in ("plateau", "sustained-pressure"):
        if hi - lo >= 4:
            return f"declared {arc}; energy spans {lo}→{hi}"
        return None

    if arc == "front-loaded":
        first_mean, last_mean = _thirds(energies)
        if first_mean < last_mean:
            return (
                f"declared front-loaded; first-third energy {round(first_mean, 1)} "
                f"below last-third {round(last_mean, 1)}"
            )
        return None

    # dark-to-light, light-to-dark, narrative, abstract-journey — no reliable signal.
    return None
