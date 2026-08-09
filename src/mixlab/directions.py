"""Concept directions — cross-strata creative briefs (#53).

The deterministic Stage 1 partitioner (:mod:`mixlab.clustering`) slices a genre pool
into BPM strata, and each stratum becomes one :class:`~mixlab.models.MixCanvas`. That
makes tempo the only generative axis: mood journeys, era dialogues, label spotlights,
and artist threads that deliberately span strata are structurally impossible.

This module adds a second, cross-strata axis. It enumerates a fixed vocabulary of
CANDIDATE directions over the *whole* genre-scoped pool, proposes only the ones the
material actually supports (each builder returns ``Direction | None``), scores each for
feasibility, seed-rotates the pick so different days surface different angles, and
materialises the survivors as :class:`~mixlab.models.MixCanvas` objects carrying a
DIRECTION BRIEF for Stage 2.

Alongside the named vocabulary sits an unnamed one: :mod:`mixlab.mining` mines the
pool for statistically dense predicate conjunctions the vocabulary has no word for
(``found`` rows). Both axes go through the same scorer and the same dedupe, so a
found set competes with a label spotlight on equal terms — see :func:`_combined_field`.

Everything here is pure and deterministic: same pool + same seed → byte-identical
output. No I/O beyond the one-line-per-direction stdout note emitted by
:func:`generate_directions` (an intentional application-layer convenience).
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field

from mixlab import mining
from mixlab.clustering import _centrality_rank, build_mix_canvas, camelot_compatible
from mixlab.models import KeyGroup, MixCanvas, MixConcept, Track
from mixlab.transitions import score_transition, tempo_relation

MIN_DIRECTION_POOL = 15
MAX_DIRECTION_POOL = 25

# Derives each builder's own deterministic RNG stream from the run seed. The per-type
# offset keeps the six builders from drawing correlated choices off the same seed.
_SEED_K = 1000003

# Fraction of BPM-sorted adjacent pairs that must be tempo-compatible for a candidate
# pool to be considered a feasible mix path.
_PATH_FEASIBILITY_THRESHOLD = 0.80

# Contrasting mood-tag poles for the mood-journey direction. Matched case-insensitively
# against Track.tags in either direction (a dark↔euphoric pair fires whether the pole
# order in the pool is dark-heavy or euphoric-heavy).
_MOOD_POLES: list[tuple[str, str]] = [
    ("dark", "euphoric"),
    ("dark", "happy"),
    ("brooding", "carnival"),
    ("melancholic", "energetic"),
    ("dreamy", "driving"),
    ("soulful", "aggressive"),
]

_MOOD_POLE_MIN = 5  # tracks required per pole for a viable mood journey
_ERA_PER_SIDE_MIN = 8
_ERA_MIN_GAP = 8
_ERA_YEAR_COVERAGE = 0.60
_LABEL_MIN_COUNT = 8
_ENERGY_COVERAGE = 0.70
_FRESH_MIN_COUNT = 10

# Field-scorer weights (see _score_field). They sum to 1.0, so feasibility stays in
# [0, 1] and the three components are directly comparable across pools.
_W_FRESHNESS = 0.25
_W_IDENTITY = 0.45
_W_DISTINCT = 0.30
_DEDUPE_JACCARD = 0.6

# Mined ("found") rows are open-ended — the miner shortlists up to twelve pairs — so
# they are capped after scoring, before they can crowd out the named vocabulary.
_MAX_MINED_PER_POOL = 3
_MINED_TYPE = "found"


@dataclass(frozen=True)
class Direction:
    direction_type: str
    title: str
    mood: str
    track_ids: list[str]
    brief: str
    feasibility: float
    thread_artist: str = ""
    identity: float = 0.0
    freshness: float = 0.0
    # Defining subsets (see models.KeyGroup) — what a curated concept must retain
    # for the direction's promise to hold. Empty for types whose canvas is
    # homogeneous (fresh_crate) or already validator-covered (genre_traverse).
    key_groups: list[KeyGroup] = dataclass_field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rank(tracks: list[Track]) -> list[Track]:
    """Centrality-rank, tolerating the empty list (which ``_centrality_rank`` cannot)."""
    return _centrality_rank(tracks) if tracks else []


def _dedupe(tracks: list[Track]) -> list[Track]:
    """De-duplicate by track_id, preserving first-seen order."""
    seen: set[str] = set()
    out: list[Track] = []
    for t in tracks:
        if t.track_id not in seen:
            seen.add(t.track_id)
            out.append(t)
    return out


def _has_tag(track: Track, pole: str) -> bool:
    return any(tag.lower() == pole for tag in track.tags)


def _with_tag(pool: list[Track], pole: str) -> list[Track]:
    return [t for t in pool if _has_tag(t, pole)]


def _path_feasible(tracks: list[Track]) -> tuple[bool, float]:
    """Fraction of BPM-sorted adjacent pairs whose tempo relation is not incompatible.

    Returns ``(ok, ratio)`` where ``ok`` is ``ratio >= _PATH_FEASIBILITY_THRESHOLD``.
    A single-track (or empty) pool is trivially feasible.
    """
    if len(tracks) < 2:
        return True, 1.0
    ordered = sorted(tracks, key=lambda t: (t.bpm, t.track_id))
    total = len(ordered) - 1
    ok = 0
    for a, b in zip(ordered, ordered[1:], strict=False):
        relation, _ = tempo_relation(a.bpm, b.bpm)
        if relation != "incompatible":
            ok += 1
    ratio = ok / total
    return ratio >= _PATH_FEASIBILITY_THRESHOLD, ratio


def _finalise(
    *,
    direction_type: str,
    title: str,
    mood: str,
    chosen: list[Track],
    brief: str,
    signal: float,
    pool: list[Track],
    thread_artist: str = "",
    key_groups: list[KeyGroup] | None = None,
) -> Direction | None:
    """Cap and gate a candidate pool, emitting an *unscored* Direction.

    Gates only: returns ``None`` when the pool falls below :data:`MIN_DIRECTION_POOL`
    after de-dup and capping, or when the BPM-sorted ordering is not path-feasible.
    Pool size and path feasibility used to be 70% of the score, which saturated —
    every surviving candidate is full-size and path-feasible by construction, so they
    carried no information. They are pass/fail here and ``feasibility`` is left at 0.0
    for :func:`_shape_field`, which owns ranking over the whole candidate field.
    """
    chosen_capped = _dedupe(chosen)[:MAX_DIRECTION_POOL]
    if len(chosen_capped) < MIN_DIRECTION_POOL:
        return None
    ok, _ratio = _path_feasible(chosen_capped)
    if not ok:
        return None
    return Direction(
        key_groups=_clamp_key_groups(key_groups or [], {t.track_id for t in chosen_capped}),
        direction_type=direction_type,
        title=title,
        mood=mood,
        track_ids=[t.track_id for t in chosen_capped],
        brief=brief,
        feasibility=0.0,
        thread_artist=thread_artist,
        identity=round(min(max(signal, 0.0), 1.0), 4),
        freshness=round(_freshness(chosen_capped, pool), 4),
    )


def _clamp_key_groups(groups: list[KeyGroup], shipped_ids: set[str]) -> list[KeyGroup]:
    """Intersect each group with what actually ships and clamp ``required`` to match.

    The dedupe/cap in :func:`_finalise` (or collection drift, for spec-supplied
    groups) can drop key tracks; a requirement above what is present would be
    unsatisfiable by construction. Groups left empty vanish.
    """
    out: list[KeyGroup] = []
    for g in groups:
        ids = [tid for tid in dict.fromkeys(g.track_ids) if tid in shipped_ids]
        required = min(g.required, len(ids))
        if ids and required >= 1:
            out.append(KeyGroup(label=g.label, required=required, track_ids=ids))
    return out


def _balance(a: int, b: int) -> float:
    """Symmetric balance signal: 1.0 when the two counts are equal, → 0 as they diverge."""
    hi = max(a, b)
    return min(a, b) / hi if hi else 0.0


def _freshness(chosen: list[Track], pool: list[Track]) -> float:
    """Median date_added count-percentile of ``chosen`` within ``pool``.

    Rank-based (ISO strings sort lexicographically — no date parsing), so it
    cannot saturate under an all-unplayed pool. Empty date_added sorts oldest.
    """
    if len(pool) < 2:
        return 0.5
    ordered = sorted(pool, key=lambda t: (t.date_added, t.track_id))
    pct = {t.track_id: i / (len(ordered) - 1) for i, t in enumerate(ordered)}
    return statistics.median(pct[t.track_id] for t in chosen)


def _log_lift(ratio: float) -> float:
    """Common identity scale for concentration ratios: 1x→0, 2x→1/3, 8x+→1."""
    return min(math.log2(max(ratio, 1.0)) / 3.0, 1.0)


def _jaccard(a: list[str], b: list[str]) -> float:
    """Track-id overlap of two candidate pools: |A∩B| / |A∪B|.

    Two empty pools are *identical*, not disjoint, so they score 1.0 — the metric
    is total over any pair of id lists rather than depending on callers to filter
    empty pools out first. (Builders and the miner both floor their pools at
    :data:`MIN_DIRECTION_POOL`, so this is a contract guard, not a live case.)
    """
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    if not union:
        return 1.0
    return len(sa & sb) / union


def _rank_key(direction: Direction) -> tuple[float, str, str, tuple[str, ...], str, str, str, float, float]:
    """Pass-1 ordering: best distinctiveness-free score first, then total tie-breaks.

    Every field except ``feasibility`` (uniformly 0.0 on input — the field scorer
    computes it) participates, so the key is a total order over the candidates and
    the survivor of a dedupe never depends on input order. The trailing components
    matter for mined rows in particular: they all share ``direction_type ==
    "found"`` and two pairs over the same members can share a title too, which
    ``(score, type, title, track_ids)`` alone cannot separate.
    """
    score = _W_FRESHNESS * direction.freshness + _W_IDENTITY * direction.identity
    return (
        -score / (_W_FRESHNESS + _W_IDENTITY),
        direction.direction_type,
        direction.title,
        tuple(sorted(direction.track_ids)),
        direction.mood,
        direction.brief,
        direction.thread_artist,
        direction.identity,
        direction.freshness,
    )


def _final_key(direction: Direction) -> tuple[float, str, str, tuple[str, ...], str, str, str, float, float]:
    """Pass-2 ordering: best feasibility first, then :func:`_rank_key`'s total tie-break.

    ``(-feasibility, direction_type)`` alone is NOT a total order over this field:
    mined rows all carry ``direction_type == "found"`` until :func:`_shape_field`
    renumbers them, so two equally-scored mined rows would be separated only by
    sort stability. Reusing the rank key's trailing components makes the order
    total and the output order-independent for real.
    """
    return (-direction.feasibility, *_rank_key(direction)[1:])


def _rank_and_dedupe(candidates: list[Direction]) -> list[Direction]:
    """Pass 1 of the two-pass scorer: rank on the distinctiveness-free score, drop clones.

    Walks :func:`_rank_key` order (``(0.25·freshness + 0.45·identity) / 0.70``) and
    keeps a candidate only when it overlaps every already-kept candidate at Jaccard
    <= :data:`_DEDUPE_JACCARD`. Returns the survivors **in rank order**, still
    unscored (``feasibility`` 0.0) — callers may drop further rows (see
    :func:`_shape_field`'s mined cap) before pass 2 measures distinctiveness.
    """
    kept: list[Direction] = []
    for cand in sorted(candidates, key=_rank_key):
        if all(_jaccard(cand.track_ids, k.track_ids) <= _DEDUPE_JACCARD for k in kept):
            kept.append(cand)
    return kept


def _score_final(field: list[Direction]) -> list[Direction]:
    """Pass 2 of the two-pass scorer: distinctiveness and feasibility over ``field``.

    Distinctiveness is measured against **exactly the rows passed in** and nothing
    else, so the caller decides what the field is. :func:`_shape_field` passes the
    post-dedupe *post-cap* field — the rows the operator actually sees — which is
    what the design calls for: measuring against rows that get dropped afterwards
    would let a capped-away row reorder the ones that ship.

    Two passes rather than one because distinctiveness depends on which candidates
    survive and survival depends on rank; scoring them together would be circular.
    Deterministic: total sort key (:func:`_final_key`), order-independent output.
    """
    out: list[Direction] = []
    for cand in field:
        others = [k for k in field if k is not cand]
        # A lone survivor has nothing to be distinct from: neutral 0.5 rather than a
        # free 1.0, so a one-direction pool cannot out-score a contested field.
        distinct = 1.0 - max(_jaccard(cand.track_ids, k.track_ids) for k in others) if others else 0.5
        feasibility = _W_FRESHNESS * cand.freshness + _W_IDENTITY * cand.identity + _W_DISTINCT * distinct
        out.append(replace(cand, feasibility=round(feasibility, 4)))
    return sorted(out, key=_final_key)


def _score_field(candidates: list[Direction]) -> list[Direction]:
    """Both passes over ``candidates`` with no cap in between: dedupe, then score.

    The uncapped composition — distinctiveness is measured against every candidate
    that survived the clone dedupe. That is the right field only when the caller
    ships all of them; the production path caps mined rows first and therefore calls
    :func:`_rank_and_dedupe` and :func:`_score_final` itself (see :func:`_shape_field`).
    """
    return _score_final(_rank_and_dedupe(candidates))


# ---------------------------------------------------------------------------
# Direction builders (type index in comment drives the per-builder RNG stream)
# ---------------------------------------------------------------------------


def _build_mood_journey(pool: list[Track], *, seed: int) -> Direction | None:
    """Contrasting mood poles bridged by neutral material (type index 0)."""
    viable: list[tuple[str, str]] = []
    for start, end in _MOOD_POLES:
        if len(_with_tag(pool, start)) >= _MOOD_POLE_MIN and len(_with_tag(pool, end)) >= _MOOD_POLE_MIN:
            viable.append((start, end))
    if not viable:
        return None
    viable.sort()
    rng = random.Random(seed * _SEED_K + 0)
    start_pole, end_pole = rng.choice(viable)

    start_tracks = _with_tag(pool, start_pole)
    end_tracks = _with_tag(pool, end_pole)
    start_ranked = _rank(start_tracks)[:10]
    end_ranked = _rank(end_tracks)[:10]

    start_med = statistics.median(t.bpm for t in start_tracks)
    end_med = statistics.median(t.bpm for t in end_tracks)
    lo, hi = sorted((start_med, end_med))
    bridge_pool = [t for t in pool if not _has_tag(t, start_pole) and not _has_tag(t, end_pole) and lo <= t.bpm <= hi]
    bridge_ranked = _rank(bridge_pool)[:5]

    chosen = start_ranked + bridge_ranked + end_ranked
    signal = _balance(len(start_tracks), len(end_tracks))
    brief = (
        f"This set is a mood journey from '{start_pole}' to '{end_pole}'. Open in the {start_pole} pole, "
        f"then move deliberately through the neutral bridge tracks, and land in the {end_pole} pole. "
        f"The emotional travel is the thesis — sequence it as {start_pole} -> bridge -> {end_pole} so the "
        f"room feels the transformation rather than a sudden switch. Let the bridge do the tonal work that "
        f"makes the arrival earned."
    )
    return _finalise(
        direction_type="mood_journey",
        title=f"Mood journey: {start_pole} -> {end_pole}",
        mood=f"{start_pole} to {end_pole}",
        chosen=chosen,
        brief=brief,
        signal=signal,
        pool=pool,
        key_groups=[
            KeyGroup(f"'{start_pole}' pole", min(2, len(start_ranked)), [t.track_id for t in start_ranked]),
            KeyGroup(f"'{end_pole}' pole", min(2, len(end_ranked)), [t.track_id for t in end_ranked]),
        ],
    )


def _era_split_local(pool: list[Track]) -> tuple[list[Track], list[Track]] | None:
    """Split the pool at its largest year gap. Returns (old, new) or None.

    Local to this module (per #53 design) so it can enforce the direction-specific
    per-side floor (``_ERA_PER_SIDE_MIN``) rather than clustering's shortlist floor.
    """
    dated = [(t.year, t) for t in pool if t.year is not None and t.year > 0]
    if len(dated) < _ERA_YEAR_COVERAGE * len(pool):
        return None
    dated.sort(key=lambda p: (p[0], p[1].track_id))
    if len(dated) < 2:
        return None
    best_gap = -1
    gap_idx = 0
    for i in range(len(dated) - 1):
        gap = dated[i + 1][0] - dated[i][0]
        if gap > best_gap:
            best_gap = gap
            gap_idx = i
    if best_gap < _ERA_MIN_GAP:
        return None
    gap_start = dated[gap_idx][0]
    old = [t for yr, t in dated if yr <= gap_start]
    new = [t for yr, t in dated if yr > gap_start]
    if len(old) < _ERA_PER_SIDE_MIN or len(new) < _ERA_PER_SIDE_MIN:
        return None
    return old, new


def _build_era_dialogue(pool: list[Track], *, seed: int) -> Direction | None:
    """Old-vs-new conversation across a year gap (type index 1)."""
    split = _era_split_local(pool)
    if split is None:
        return None
    old, new = split
    old_ranked = _rank(old)[:12]
    new_ranked = _rank(new)[:13]
    old_years = [t.year for t in old if t.year is not None]
    new_years = [t.year for t in new if t.year is not None]
    old_lo, old_hi = min(old_years), max(old_years)
    new_lo, new_hi = min(new_years), max(new_years)
    signal = _balance(len(old), len(new))
    brief = (
        f"This set is an era dialogue: {old_lo}-{old_hi} material in conversation with {new_lo}-{new_hi}. "
        f"Alternate eras deliberately — don't play all the old records then all the new. Use the contrast in "
        f"production era (drum sound, low-end weight, mix polish) as the narrative device, letting each side "
        f"comment on the other. The dialogue only works if the room can hear both voices."
    )
    return _finalise(
        direction_type="era_dialogue",
        title=f"Era dialogue: {old_lo}-{old_hi} vs {new_lo}-{new_hi}",
        mood=f"{old_lo}-{old_hi} vs {new_lo}-{new_hi}",
        chosen=old_ranked + new_ranked,
        brief=brief,
        signal=signal,
        pool=pool,
        key_groups=[
            KeyGroup(f"{old_lo}-{old_hi} era", min(2, len(old_ranked)), [t.track_id for t in old_ranked]),
            KeyGroup(f"{new_lo}-{new_hi} era", min(2, len(new_ranked)), [t.track_id for t in new_ranked]),
        ],
    )


def _build_label_spotlight(pool: list[Track], *, seed: int, collection: list[Track] | None = None) -> Direction | None:
    """A single label's scene DNA, optionally braced by adjacent-key outsiders (type index 2)."""
    label_counts = Counter(t.label for t in pool if t.label)
    qualifying = sorted(label for label, count in label_counts.items() if count >= _LABEL_MIN_COUNT)
    if not qualifying:
        return None
    rng = random.Random(seed * _SEED_K + 2)
    label = rng.choice(qualifying)

    label_tracks = [t for t in pool if t.label == label]
    chosen = _rank(label_tracks)[:MAX_DIRECTION_POOL]

    if len(label_tracks) < 15:
        keys = [t.camelot_key for t in label_tracks]
        dominant_key = Counter(keys).most_common(1)[0][0] if keys else ""
        non_label = [t for t in pool if t.label != label]
        adjacent = [t for t in non_label if dominant_key and camelot_compatible(dominant_key, t.camelot_key)]
        chosen = chosen + _rank(adjacent)[:5]

    share_pool = label_counts[label] / len(pool) if pool else 0.0
    if collection:
        share_coll = sum(1 for t in collection if t.label == label) / len(collection)
        signal = _log_lift(share_pool / share_coll) if share_coll else 0.0
    else:
        signal = share_pool
    brief = (
        f"This set is a label spotlight on {label}. Treat the label as the scene DNA — the reason these "
        f"records share a sound. Lean into that house style rather than smoothing it out: the coherence is the "
        f"point. Where a non-label track appears, it should extend the label's aesthetic, not dilute it. Say in "
        f"the report what makes {label}'s sound identifiable."
    )
    label_shipped = [t for t in chosen if t.label == label]
    return _finalise(
        direction_type="label_spotlight",
        title=f"Label spotlight: {label}",
        mood=f"{label} scene DNA",
        chosen=chosen,
        brief=brief,
        signal=signal,
        pool=pool,
        key_groups=[
            KeyGroup(f"{label} catalogue", min(4, len(label_shipped)), [t.track_id for t in label_shipped]),
        ],
    )


def _build_artist_thread(pool: list[Track], *, seed: int) -> Direction | None:
    """One artist/remixer as the spine, 2-3 tracks as chapter markers (type index 3)."""
    name_tracks: dict[str, list[Track]] = {}
    for t in pool:
        names: set[str] = set()
        if t.artist:
            names.add(t.artist)
        if t.remixer:
            names.add(t.remixer)
        for name in names:
            name_tracks.setdefault(name, []).append(t)

    qualifying = sorted(name for name, ts in name_tracks.items() if 2 <= len(ts) <= 3)
    if not qualifying:
        return None
    rng = random.Random(seed * _SEED_K + 3)
    thread_artist = rng.choice(qualifying)
    thread_tracks = name_tracks[thread_artist]
    thread_ids = {t.track_id for t in thread_tracks}
    others = [t for t in pool if t.track_id not in thread_ids]

    def _best_score(candidate: Track) -> float:
        best = 0.0
        for anchor in thread_tracks:
            best = max(best, score_transition(anchor, candidate).score, score_transition(candidate, anchor).score)
        return best

    scored = [(_best_score(t), t) for t in others]
    if any(score > 0 for score, _ in scored):
        scored.sort(key=lambda st: (-st[0], st[1].track_id))
        companions = [t for _, t in scored[:22]]
        tightness = statistics.fmean(score for score, _ in scored[: len(companions)]) if companions else 0.0
    else:
        companions = _rank(others)[:22]
        tightness = 0.3

    brief = (
        f"This set threads {thread_artist} through the mix as its spine. Their {len(thread_tracks)} tracks are "
        f"the chapter markers — name them in the report and place them as structural pillars (opener, pivot, "
        f"peak, closer) that the surrounding records build toward and away from. Everything else is there to "
        f"frame the thread, not to compete with it."
    )
    return _finalise(
        direction_type="artist_thread",
        title=f"Artist thread: {thread_artist}",
        mood=f"{thread_artist} as spine",
        chosen=thread_tracks + companions,
        brief=brief,
        signal=min(max(tightness, 0.0), 1.0),
        pool=pool,
        thread_artist=thread_artist,
        # Every chapter marker is load-bearing — the brief names their count, so
        # dropping one falsifies the thesis (the "2 of 3 pillars" failure).
        key_groups=[
            KeyGroup(f"{thread_artist} spine", len(thread_tracks), [t.track_id for t in thread_tracks]),
        ],
    )


def _build_energy_shape_first(pool: list[Track], *, seed: int) -> Direction | None:
    """Balanced across energy bands to realise a declared arc (type index 4)."""
    with_energy = [t for t in pool if t.energy is not None]
    if len(with_energy) < _ENERGY_COVERAGE * len(pool):
        return None

    low = [t for t in with_energy if t.energy is not None and t.energy <= 4]
    high = [t for t in with_energy if t.energy is not None and t.energy >= 6]
    mid = [t for t in with_energy if t.energy is not None and 4 < t.energy < 6]

    realizable: list[str] = []
    if len(low) >= 4 and len(high) >= 4:
        realizable.extend(["wave", "double-peak"])
    if len(low) >= 5 and len(high) >= 5:
        realizable.append("dark-to-light")
    if not realizable:
        return None
    realizable.sort()
    rng = random.Random(seed * _SEED_K + 4)
    arc = rng.choice(realizable)

    bands = [band for band in (low, mid, high) if band]
    per_band = max(1, MAX_DIRECTION_POOL // len(bands))
    chosen: list[Track] = []
    for band in bands:
        chosen.extend(_rank(band)[:per_band])
    # Top up to the cap from any remaining energy-carrying tracks, most central first.
    if len(chosen) < MAX_DIRECTION_POOL:
        chosen_ids = {t.track_id for t in chosen}
        leftover = _rank([t for t in with_energy if t.track_id not in chosen_ids])
        chosen.extend(leftover[: MAX_DIRECTION_POOL - len(chosen)])

    signal = _balance(len(low), len(high))
    brief = (
        f"For this set the energy arc is the thesis: build it as a '{arc}' shape. The declared arc_type is "
        f"{arc}. Every placement should serve that curve — the low-energy tracks are the troughs and the "
        f"high-energy tracks the crests, and the sequence must make the shape audible, not merely present. "
        f"Choose an opener and closer that frame the arc's start and resolution."
    )
    chosen_ids_final = {t.track_id for t in chosen}
    low_shipped = [t for t in low if t.track_id in chosen_ids_final]
    high_shipped = [t for t in high if t.track_id in chosen_ids_final]
    return _finalise(
        direction_type="energy_shape_first",
        title=f"Energy shape: {arc}",
        mood=arc,
        chosen=chosen,
        brief=brief,
        signal=signal,
        pool=pool,
        # Troughs and crests only — an arc survives losing mid-band filler, but not
        # its low or high ends.
        key_groups=[
            KeyGroup("low-energy troughs", min(2, len(low_shipped)), [t.track_id for t in low_shipped]),
            KeyGroup("high-energy crests", min(2, len(high_shipped)), [t.track_id for t in high_shipped]),
        ],
    )


def _build_fresh_crate(pool: list[Track], *, seed: int) -> Direction | None:
    """Newest additions, braced by a few grounding anchors (type index 5).

    Recency is measured by count percentile over ``date_added`` (ISO strings sorted
    lexicographically — no date parsing), matching the #53 design note.
    """
    dated = [t for t in pool if t.date_added]
    if not dated:
        return None
    dated_sorted = sorted(dated, key=lambda t: (t.date_added, t.track_id))
    cutoff = int(len(dated_sorted) * 0.8)
    newest_slice = dated_sorted[cutoff:]
    if len(newest_slice) < _FRESH_MIN_COUNT:
        return None

    newest = dated_sorted[-20:]
    chosen_ids = {t.track_id for t in newest}
    anchor_candidates = [t for t in pool if t.track_id not in chosen_ids and ((t.rating or 0) > 0 or t.play_count > 0)]
    anchor_candidates.sort(key=lambda t: (-(t.rating or 0), -t.play_count, t.track_id))
    chosen = list(newest) + anchor_candidates[:5]

    # Identity = recency CONCENTRATION of the shipped set, not "does the pool carry
    # dates". The dated share (len(dated)/len(pool)) pinned at 1.0 on every real
    # pool — Rekordbox stamps DateAdded on ~every track — which is exactly the
    # saturation this scorer exists to remove. Rescaled from the median date-added
    # count-percentile: 0 at or below the pool's median age, 1.0 only when the
    # shipped set sits at the very newest end. ``chosen`` is what _finalise ships
    # (<= 25 tracks, no duplicates by construction), so the two agree.
    signal = max(0.0, 2 * (_freshness(chosen, pool) - 0.5))
    brief = (
        "This set is a fresh-crate debut showcase — the point is surfacing what just arrived. Frame the newest "
        "additions as discoveries worth their first play, and use the handful of grounding anchor tracks only to "
        "give the room something familiar to hold while the new material lands. Do not bury the debuts; they are "
        "the reason the set exists."
    )
    return _finalise(
        direction_type="fresh_crate",
        title="Fresh crate: newest additions",
        mood="debut showcase",
        chosen=chosen,
        brief=brief,
        signal=signal,
        pool=pool,
    )


_TRAVERSE_REGIME_MIN = 5  # tracks required for a regime to be a viable chapter
# Live finding (v1.8.3 still not firing on the real pool): gap-based regime splitting
# needs a >12 BPM hole between sorted neighbours, and a full collection's BPMs form a
# near-continuum — no holes, one giant "regime", builder bails. Regimes are now tempo
# DENSITY PEAKS: a smoothed 1-BPM histogram, peaks greedily picked by mass with a
# minimum separation, regime = tracks within a half-width of the peak. In-between
# material simply doesn't join a chapter, which is what a journey wants.
_TRAVERSE_PEAK_SEPARATION = 20.0  # min BPM distance between regime peaks
_TRAVERSE_REGIME_HALF_WIDTH = 8.0  # regime membership: within this of the peak
_TRAVERSE_SMOOTH_WINDOW = 3  # histogram smoothing: +-N BPM bins
_TRAVERSE_MIN_BRIDGES = 2  # verified ratio-bridge pairs required per regime hop
_TRAVERSE_MAX_CHAPTERS = 4  # journey length cap — quota per chapter stays >= 6 tracks

_TRAVERSE_RELATION_LABELS: dict[str, str] = {
    "halftime": "halftime lock",
    "double": "double-time lock",
    "three_four": "3:4 shuffle",
    "four_three": "4:3 push",
}


def _regime_label(regime: list[Track]) -> str:
    """Dominant genre tag plus BPM span, e.g. ``House 122-130``."""
    dominant = Counter(t.genre for t in regime if t.genre).most_common(1)
    genre = dominant[0][0] if dominant else "mixed"
    lo = min(t.bpm for t in regime)
    hi = max(t.bpm for t in regime)
    span = f"{lo:g}" if round(lo) == round(hi) else f"{lo:.0f}-{hi:.0f}"
    return f"{genre} {span}"


def _tempo_regimes(usable: list[Track]) -> list[list[Track]]:
    """Density-peak tempo regimes, ascending by peak BPM.

    Builds a 1-BPM histogram smoothed over ±:data:`_TRAVERSE_SMOOTH_WINDOW` bins,
    greedily picks peaks by smoothed mass (ties → lower BPM) with at least
    :data:`_TRAVERSE_PEAK_SEPARATION` between peaks, then forms each regime from the
    tracks within :data:`_TRAVERSE_REGIME_HALF_WIDTH` of its peak. Regimes are
    disjoint because the separation exceeds twice the half-width. Peaks whose
    membership falls under :data:`_TRAVERSE_REGIME_MIN` are dropped. Deterministic.
    """
    if not usable:
        return []
    counts = Counter(round(t.bpm) for t in usable)
    lo, hi = min(counts), max(counts)
    smoothed = {
        b: sum(counts.get(b + d, 0) for d in range(-_TRAVERSE_SMOOTH_WINDOW, _TRAVERSE_SMOOTH_WINDOW + 1))
        for b in range(lo, hi + 1)
    }
    peaks: list[int] = []
    for bpm_bin, mass in sorted(smoothed.items(), key=lambda kv: (-kv[1], kv[0])):
        if mass < _TRAVERSE_REGIME_MIN:
            break
        if all(abs(bpm_bin - p) >= _TRAVERSE_PEAK_SEPARATION for p in peaks):
            peaks.append(bpm_bin)
        if len(peaks) >= _TRAVERSE_MAX_CHAPTERS + 2:  # spares — chaining may skip some
            break
    regimes: list[list[Track]] = []
    for peak in sorted(peaks):
        members = [t for t in usable if abs(t.bpm - peak) <= _TRAVERSE_REGIME_HALF_WIDTH]
        if len(members) >= _TRAVERSE_REGIME_MIN:
            regimes.append(members)
    return regimes


def _build_genre_traverse(pool: list[Track], *, seed: int) -> Direction | None:
    """Cross-genre journey through distinct tempo regimes linked by ratio bridges (type index 6).

    Fires only when the pool's tempo-density peaks (see :func:`_tempo_regimes`) form
    two or more regimes chained by verified pitch-lock bridges — halftime/double-time/
    3:4/4:3 pairs within the ±6% window. Single-regime pools (every standard-genre
    run) return ``None``, so this direction only surfaces for cross-genre pools
    (``4x4``, ``traverse``). Regimes that cannot be bridged from the chain are skipped
    rather than forced.
    """
    usable = sorted((t for t in pool if t.bpm > 0), key=lambda t: (t.bpm, t.track_id))
    regimes = _tempo_regimes(usable)
    if len(regimes) < 2:
        return None

    # Chain regimes ascending by BPM; a regime joins only if it has enough verified
    # bridges from the current chain end. Camelot-compatible bridges sort first.
    # Live finding (three consecutive non-firings on the traverse pool): a chain
    # anchored to regimes[0] dies whenever the lowest regime bridges to nothing —
    # a ~77 BPM hip-hop block has no ratio partner even when 124↔170 bridges are
    # plentiful further up. Try every regime as the chain start and keep the
    # longest chain (ties → earliest start, deterministic).
    def _bridge_pairs(prev: list[Track], candidate: list[Track]) -> list[tuple[Track, Track]]:
        pairs = [
            (a, b)
            for a in prev
            for b in candidate
            if tempo_relation(a.bpm, b.bpm)[0] not in ("incompatible", "straight")
        ]
        pairs.sort(
            key=lambda p: (
                not camelot_compatible(p[0].camelot_key, p[1].camelot_key),
                p[0].track_id,
                p[1].track_id,
            )
        )
        return pairs

    chain: list[list[Track]] = []
    hops: list[list[tuple[Track, Track]]] = []
    best_key = (0, 0)
    for start in range(len(regimes)):
        cand_chain: list[list[Track]] = [regimes[start]]
        cand_hops: list[list[tuple[Track, Track]]] = []
        for candidate in regimes[start + 1 :]:
            if len(cand_chain) >= _TRAVERSE_MAX_CHAPTERS:
                break
            pairs = _bridge_pairs(cand_chain[-1], candidate)
            if len(pairs) >= _TRAVERSE_MIN_BRIDGES:
                cand_chain.append(candidate)
                cand_hops.append(pairs)
        # More chapters wins; ties break to the chain with more material (a
        # two-chapter journey over 16 tracks beats one over 14 that would then
        # die at the MIN_DIRECTION_POOL floor), then to the earliest start.
        cand_key = (len(cand_chain), sum(len(c) for c in cand_chain))
        if cand_key > best_key:
            best_key = cand_key
            chain = cand_chain
            hops = cand_hops
    if len(chain) < 2:
        return None

    # Seed-flip the journey direction: half the days climb, half descend.
    rng = random.Random(seed * _SEED_K + 6)
    descending = rng.random() < 0.5
    chapters = list(reversed(chain)) if descending else chain
    chapter_hops = [[(b, a) for a, b in pairs] for pairs in reversed(hops)] if descending else hops

    # Per-chapter selection: bridge endpoints first (they make the hops physically
    # possible), then centrality fill to an even quota.
    quota = max(_TRAVERSE_REGIME_MIN, MAX_DIRECTION_POOL // len(chapters))
    chosen: list[Track] = []
    for i, chapter in enumerate(chapters):
        endpoint_ids: list[str] = []
        if i > 0:
            endpoint_ids.extend(b.track_id for _a, b in chapter_hops[i - 1][:2])
        if i < len(chapter_hops):
            endpoint_ids.extend(a.track_id for a, _b in chapter_hops[i][:2])
        endpoints = [t for t in chapter if t.track_id in set(endpoint_ids)]
        fill = [t for t in _rank(chapter) if t.track_id not in {e.track_id for e in endpoints}]
        chosen.extend((endpoints + fill)[:quota])

    labels = [_regime_label(c) for c in chapters]
    journey = " → ".join(labels)
    bridge_lines: list[str] = []
    for i, pairs in enumerate(chapter_hops):
        for a, b in pairs[:2]:
            relation, _ = tempo_relation(a.bpm, b.bpm)
            mechanism = _TRAVERSE_RELATION_LABELS.get(relation, relation)
            bridge_lines.append(
                f"  hop {i + 1} ({labels[i]} → {labels[i + 1]}): {a.artist} — {a.title} ({a.bpm:g}) → "
                f"{b.artist} — {b.title} ({b.bpm:g}) via {mechanism}"
            )

    min_bridges = min(len(pairs) for pairs in chapter_hops)
    size_balance = min(len(c) for c in chapters) / max(len(c) for c in chapters)
    signal = 0.6 * min(min_bridges / 5.0, 1.0) + 0.4 * size_balance

    brief = (
        f"GENRE TRAVERSE. This set travels {journey} — one chapter per tempo regime, in this order. "
        f"Every regime change must be executed on a ratio bridge (pitch-locked halftime, double-time, "
        f"3:4 or 4:3 blend within the pitch window) — never a raw tempo ride. Verified bridges "
        f"(use these or equivalents from the pool):\n" + "\n".join(bridge_lines) + "\n"
        "Spend at least three tracks establishing each chapter before hopping, land every hop as a "
        "named chapter pivot with its mechanism, and keep the energy arc continuous across the "
        "boundary. The journey is the thesis: the room should feel it crossed genres without ever "
        "losing the pulse."
    )
    return _finalise(
        direction_type="genre_traverse",
        title=f"Genre traverse: {labels[0]} -> {labels[-1]}",
        mood=f"cross-genre journey in {len(chapters)} chapters",
        chosen=chosen,
        brief=brief,
        signal=signal,
        pool=pool,
    )


_BUILDERS = (
    _build_mood_journey,
    _build_era_dialogue,
    _build_label_spotlight,
    _build_artist_thread,
    _build_energy_shape_first,
    _build_fresh_crate,
    _build_genre_traverse,
)


def _named_candidates(pool: list[Track], *, seed: int, collection: list[Track] | None = None) -> list[Direction]:
    """Every surviving named-builder candidate, unsorted. Single enumeration
    point shared by the map path and the run path (previously duplicated)."""
    candidates: list[Direction] = []
    for builder in _BUILDERS:
        # Dispatch by name, not through the union-typed loop variable: only
        # label_spotlight takes `collection`, and mypy (rightly) rejects the
        # extra kwarg on the tuple's joined callable type.
        direction = (
            _build_label_spotlight(pool, seed=seed, collection=collection)
            if builder is _build_label_spotlight
            else builder(pool, seed=seed)
        )
        if direction is not None:
            candidates.append(direction)
    return candidates


def _is_mined(direction: Direction) -> bool:
    """True for a mined row, before (``found``) or after (``found_1``) renumbering."""
    return direction.direction_type.startswith(_MINED_TYPE)


def _shape_field(candidates: list[Direction]) -> list[Direction]:
    """The production pipeline: rank & dedupe, cap the mined rows, then score them.

    Order matters and is the whole point of splitting the scorer in two:

    1. :func:`_rank_and_dedupe` — pass 1, clones removed in rank order.
    2. **Shape the mined rows** by walking that rank order: keep at most
       :data:`_MAX_MINED_PER_POOL`, and at most one row per title (two pairs can
       name themselves the same thing — a tag pairs with both a tempo pocket and a
       key neighbourhood — so the higher-ranked wins). Named rows pass through
       untouched: they are a fixed vocabulary of seven and cap themselves.
    3. :func:`_score_final` — pass 2 over **exactly the surviving field**, so
       distinctiveness reflects the rows that ship. Measuring it before the cap
       would let a row that never ships reorder (or dedupe-drop) the ones that do.
    4. Renumber ``found`` -> ``found_1``/``found_2``/... in *final* rank order, so
       downstream consumers get stable, distinguishable direction types and
       ``found_1`` really is the best-scoring mined row.
    """
    survivors: list[Direction] = []
    seen_titles: set[str] = set()
    mined_kept = 0
    for cand in _rank_and_dedupe(candidates):  # pass-1 rank order
        if cand.direction_type == _MINED_TYPE:
            if mined_kept >= _MAX_MINED_PER_POOL or cand.title in seen_titles:
                continue
            mined_kept += 1
            seen_titles.add(cand.title)
        survivors.append(cand)

    out: list[Direction] = []
    mined_rank = 0
    for cand in _score_final(survivors):  # final (-feasibility, ...) order
        if cand.direction_type == _MINED_TYPE:
            mined_rank += 1
            cand = replace(cand, direction_type=f"{_MINED_TYPE}_{mined_rank}")
        out.append(cand)
    return out


def _field_parts(
    pool: list[Track], *, seed: int, collection: list[Track] | None = None
) -> tuple[list[Direction], list[Direction]]:
    """Every candidate source over ``pool``, pre-scoring: ``(named, mined)``.

    THE single assembly point for the candidate field — a new source of Directions
    is added here and nowhere else, so the map path and the run path cannot drift
    apart on what the field contains (adding a third part changes this signature,
    which breaks both callers loudly rather than silently skipping one).

    Returned as parts rather than one list because :func:`generate_directions`
    reports proposal counts per source in its run-log diagnostic, and those counts
    are measured *before* scoring drops anything.
    """
    return _named_candidates(pool, seed=seed, collection=collection), mining.mine_pool(pool)


def _combined_field(pool: list[Track], *, seed: int, collection: list[Track] | None = None) -> list[Direction]:
    """The whole candidate field over ``pool``: named builders plus mined pairs.

    Both axes go through the one scorer so a found set and a label spotlight are
    ranked on the same footing (and dedupe against each other — a mined pair that
    is really just a label's catalogue loses to, or beats, the label spotlight
    rather than shipping alongside it).
    """
    named, mined = _field_parts(pool, seed=seed, collection=collection)
    return _shape_field(named + mined)


def enumerate_directions(pool: list[Track], *, seed: int, collection: list[Track] | None = None) -> list[Direction]:
    """Every surviving Direction candidate over ``pool``, exhaustively.

    The library-map path (#40): unlike :func:`generate_directions` there is no
    seed-derived rotation, no ``max_directions`` cap, no materialisation into
    MixCanvas, and no run-log printing — the caller wants the full candidate
    field with feasibility scores, deterministically ordered. ``collection`` is the
    whole scoped library, used by builders that measure a pool's concentration
    against the library baseline (label_spotlight).

    Named and mined rows both appear; mined ones carry ``direction_type``
    ``found_1``..``found_{_MAX_MINED_PER_POOL}`` (the map shows the whole capped
    field, the run path picks at most one of them).
    """
    return sorted(
        _combined_field(pool, seed=seed, collection=collection),
        key=lambda d: (-d.feasibility, d.direction_type),
    )


def generate_directions(
    pool: list[Track],
    tracks_by_id: dict[str, Track],
    *,
    seed: int,
    max_directions: int = 3,
    collection: list[Track] | None = None,
) -> list[MixCanvas]:
    """Enumerate, score, seed-rotate, and materialise concept directions over ``pool``.

    Each builder proposes a :class:`Direction` only when the material supports it and
    the miner proposes conjunctions the vocabulary has no name for;
    :func:`_shape_field` then dedupes clones, scores the surviving field, and caps
    and renumbers the mined rows. Named survivors are sorted by
    ``(-feasibility, direction_type)``, then rotated by a seed-derived offset so
    different seeds/days surface different directions while the strongest ones appear
    often. Mined rows do not rotate: a run ships at most one found set, always the
    best-scoring, because they are the speculative axis. Up to ``max_directions``
    canvases are materialised (via ``build_mix_canvas``) carrying the direction's
    ``brief``/``direction_type``/``thread_artist``. ``collection`` is the whole scoped
    library, used by builders that measure a pool's concentration against the library
    baseline (label_spotlight) — pass it so the run path scores identically to the map.

    Prints one ``Direction: <type> — <title> (feasibility <f>)`` line per materialised
    direction to stdout — an intentional application-layer convenience so the run log
    records which directions fired without threading feasibility back through the caller.
    Deterministic: same pool + same seed → identical output.
    """
    proposals, mined_proposals = _field_parts(pool, seed=seed, collection=collection)
    # One diagnostic line so non-firing builders are visible in the run log —
    # genre_traverse silently returning None took three production runs to notice.
    # Measured pre-scoring: _shape_field can also drop a clone or cap a mined row,
    # and that is not the same failure as a builder never proposing anything.
    proposed = ", ".join(sorted(d.direction_type for d in proposals)) if proposals else "none"
    print(f"Directions proposed: {proposed} ({len(proposals)}/{len(_BUILDERS)} builders, {len(mined_proposals)} found)")
    field = _shape_field(proposals + mined_proposals)
    if not field:
        return []

    named = [d for d in field if not _is_mined(d)]
    mined = [d for d in field if _is_mined(d)]
    named.sort(key=lambda d: (-d.feasibility, d.direction_type))
    # A run ships at most one found set — and none at all when the caller asked for
    # no directions, which the unguarded mined[:1] would have overridden.
    picked = mined[:1] if max_directions >= 1 else []
    if named:
        rng = random.Random(seed)
        offset = rng.randrange(len(named))
        ordered = [named[(offset + i) % len(named)] for i in range(len(named))]
        picked += ordered[: max(max_directions - len(picked), 0)]

    result: list[MixCanvas] = []
    for direction in picked:
        concept = MixConcept(title=direction.title, mood=direction.mood, track_ids=list(direction.track_ids))
        canvas = build_mix_canvas(concept, tracks_by_id)
        canvas.brief = direction.brief
        canvas.direction_type = direction.direction_type
        canvas.thread_artist = direction.thread_artist
        canvas.key_groups = list(direction.key_groups)
        result.append(canvas)
        print(f"Direction: {direction.direction_type} — {direction.title} (feasibility {direction.feasibility:.2f})")
    return result


# ---------------------------------------------------------------------------
# Pinned direction spec (--direction-spec) — mixlab-web "Run this direction"
# ---------------------------------------------------------------------------


class DirectionSpecError(ValueError):
    """Raised when a ``--direction-spec`` payload cannot be materialised."""


_SPEC_REQUIRED_STRINGS = ("direction_type", "title", "brief")
_THREAD_TITLE_PREFIX = "Artist thread: "


def pinned_canvas_from_spec(spec_json: str, tracks_by_id: dict[str, Track]) -> MixCanvas:
    """Materialise a library-map direction entry as a pinned :class:`MixCanvas`.

    ``spec_json`` is the map payload's direction entry (see
    :func:`mixlab.library_map._direction_entry`) serialised by mixlab-web's
    "Run this direction" — machine-generated, so validation is strict and every
    failure raises :class:`DirectionSpecError` with an operator-actionable message.
    Unknown keys are ignored (the entry also carries ``feasibility``, and future
    map versions may add more).

    The tracks are pinned by id: ids that no longer resolve against the current
    collection (drift since the map was generated) are dropped, and fewer than
    :data:`MIN_DIRECTION_POOL` survivors is an error rather than a silent shrink —
    the whole point of a pinned direction is that the clicked tracks are the set.
    Survivors are capped at :data:`MAX_DIRECTION_POOL` in spec order, mirroring
    ``_finalise``. No feasibility gate is applied: the operator chose this
    direction explicitly, so path-feasibility scoring has no veto.
    """
    try:
        raw = json.loads(spec_json)
    except json.JSONDecodeError as exc:
        raise DirectionSpecError(f"--direction-spec is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DirectionSpecError("--direction-spec must be a JSON object (a map direction entry)")

    for key in _SPEC_REQUIRED_STRINGS:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DirectionSpecError(f"--direction-spec field {key!r} must be a non-empty string")
    mood = raw.get("mood", "")
    if not isinstance(mood, str):
        raise DirectionSpecError("--direction-spec field 'mood' must be a string")
    thread_artist = raw.get("thread_artist", "")
    if not isinstance(thread_artist, str):
        raise DirectionSpecError("--direction-spec field 'thread_artist' must be a string")

    track_ids = raw.get("track_ids")
    if (
        not isinstance(track_ids, list)
        or not track_ids
        or not all(isinstance(tid, str) for tid in track_ids)
    ):
        raise DirectionSpecError("--direction-spec field 'track_ids' must be a non-empty list of strings")

    resolved = [tid for tid in dict.fromkeys(track_ids) if tid in tracks_by_id]
    if len(resolved) < MIN_DIRECTION_POOL:
        raise DirectionSpecError(
            f"--direction-spec: only {len(resolved)} of {len(track_ids)} pinned tracks resolve against "
            f"the current collection (minimum {MIN_DIRECTION_POOL}) — the library analysis is stale; "
            "re-run the engine analysis and pick the direction again"
        )
    resolved = resolved[:MAX_DIRECTION_POOL]

    direction_type = raw["direction_type"].strip()
    title = raw["title"].strip()
    # Stale map payloads predate thread_artist in the wire entry (v1.14.x); the
    # artist-thread repeat-suppression validator still needs the spine's name, and
    # for artist_thread the title carries it verbatim.
    if not thread_artist and direction_type == "artist_thread" and title.startswith(_THREAD_TITLE_PREFIX):
        thread_artist = title[len(_THREAD_TITLE_PREFIX) :]

    raw_groups = raw.get("key_groups", [])
    if not isinstance(raw_groups, list):
        raise DirectionSpecError("--direction-spec field 'key_groups' must be a list")
    key_groups: list[KeyGroup] = []
    for i, g in enumerate(raw_groups):
        if (
            not isinstance(g, dict)
            or not isinstance(g.get("label"), str)
            or not isinstance(g.get("required"), int)
            or isinstance(g.get("required"), bool)
            or not isinstance(g.get("track_ids"), list)
            or not all(isinstance(tid, str) for tid in g["track_ids"])
        ):
            raise DirectionSpecError(
                f"--direction-spec key_groups[{i}] must be {{label: str, required: int, track_ids: [str]}}"
            )
        key_groups.append(KeyGroup(label=g["label"], required=g["required"], track_ids=list(g["track_ids"])))

    concept = MixConcept(title=title, mood=mood, track_ids=resolved)
    canvas = build_mix_canvas(concept, tracks_by_id)
    canvas.brief = raw["brief"].strip()
    canvas.direction_type = direction_type
    canvas.thread_artist = thread_artist
    canvas.pinned = True
    # Clamp against what actually resolved — drift can drop key tracks, and a
    # requirement above what is present would be unsatisfiable by construction.
    canvas.key_groups = _clamp_key_groups(key_groups, set(resolved))
    return canvas


# ---------------------------------------------------------------------------
# Track pool (--track-pool) — mixlab-web "Run this block"
# ---------------------------------------------------------------------------

_TRACK_POOL_LABEL_MAX_LEN = 200
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


class TrackPoolError(ValueError):
    """Raised when a ``--track-pool`` payload cannot be parsed."""


@dataclass(frozen=True)
class TrackPool:
    """A parsed ``--track-pool`` payload: an id list plus an optional display label."""

    track_ids: tuple[str, ...]
    label: str  # "" when absent


def parse_track_pool(raw: str) -> TrackPool:
    """Parse the ``--track-pool`` JSON: ``{"track_ids": [...], "label": "..."}``.

    ``raw`` is mixlab-web's "Run this block" payload — machine-generated, so
    validation is strict. Resolving the ids against the current collection (and
    the minimum-pool floor) is the caller's job, same division of labour as
    ``pinned_canvas_from_spec`` — this function only validates shape.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrackPoolError(f"--track-pool is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TrackPoolError("--track-pool must be a JSON object ({track_ids: [...], label: ...})")

    track_ids = data.get("track_ids")
    if not isinstance(track_ids, list) or not track_ids or not all(isinstance(tid, str) for tid in track_ids):
        raise TrackPoolError("--track-pool field 'track_ids' must be a non-empty list of strings")

    label = data.get("label", "")
    if not isinstance(label, str):
        raise TrackPoolError("--track-pool field 'label' must be a string")
    label = _sanitise_track_pool_label(label)

    return TrackPool(track_ids=tuple(track_ids), label=label)


def _sanitise_track_pool_label(label: str) -> str:
    """Neutralise an operator-supplied block label before it reaches stdout raw.

    The label is printed verbatim (the availability-table line, report context, history
    entry) and a worker resolves run artifacts by scanning stdout lines — a label
    containing e.g. ``"\\nRun summary: <path>"`` could hijack that scan. Strip control
    characters (including newlines) to a single space, collapse runs of whitespace, and
    cap the length so one payload can't blow up a printed line either.
    """
    label = _CONTROL_CHARS_RE.sub(" ", label)
    label = _WHITESPACE_RE.sub(" ", label).strip()
    return label[:_TRACK_POOL_LABEL_MAX_LEN]
