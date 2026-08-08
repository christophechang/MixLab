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

Everything here is pure and deterministic: same pool + same seed → byte-identical
output. No I/O beyond the one-line-per-direction stdout note emitted by
:func:`generate_directions` (an intentional application-layer convenience).
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass

from mixlab.clustering import _centrality_rank, build_mix_canvas, camelot_compatible
from mixlab.models import MixCanvas, MixConcept, Track
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
) -> Direction | None:
    """Cap/validate a candidate pool and compute its feasibility score.

    Returns ``None`` when the pool falls below :data:`MIN_DIRECTION_POOL` after de-dup
    and capping, or when the BPM-sorted ordering is not path-feasible. Feasibility is
    ``0.4·pool_fill + 0.3·path_ok_ratio + 0.3·signal_strength``.
    """
    chosen_capped = _dedupe(chosen)[:MAX_DIRECTION_POOL]
    if len(chosen_capped) < MIN_DIRECTION_POOL:
        return None
    ok, ratio = _path_feasible(chosen_capped)
    if not ok:
        return None
    pool_fill = min(len(chosen_capped) / MAX_DIRECTION_POOL, 1.0)
    signal_clamped = min(max(signal, 0.0), 1.0)
    feasibility = 0.4 * pool_fill + 0.3 * ratio + 0.3 * signal_clamped
    return Direction(
        direction_type=direction_type,
        title=title,
        mood=mood,
        track_ids=[t.track_id for t in chosen_capped],
        brief=brief,
        feasibility=round(feasibility, 4),
        thread_artist=thread_artist,
        identity=signal_clamped,
        freshness=round(_freshness(chosen_capped, pool), 4),
    )


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
    return _finalise(
        direction_type="label_spotlight",
        title=f"Label spotlight: {label}",
        mood=f"{label} scene DNA",
        chosen=chosen,
        brief=brief,
        signal=signal,
        pool=pool,
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
    return _finalise(
        direction_type="energy_shape_first",
        title=f"Energy shape: {arc}",
        mood=arc,
        chosen=chosen,
        brief=brief,
        signal=signal,
        pool=pool,
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

    signal = len(dated) / len(pool)
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
        if builder is _build_label_spotlight:
            direction = builder(pool, seed=seed, collection=collection)
        else:
            direction = builder(pool, seed=seed)
        if direction is not None:
            candidates.append(direction)
    return candidates


def enumerate_directions(pool: list[Track], *, seed: int) -> list[Direction]:
    """Every surviving Direction candidate over ``pool``, exhaustively.

    The library-map path (#40): unlike :func:`generate_directions` there is no
    seed-derived rotation, no ``max_directions`` cap, no materialisation into
    MixCanvas, and no run-log printing — the caller wants the full candidate
    field with feasibility scores, deterministically ordered.
    """
    candidates = _named_candidates(pool, seed=seed)
    candidates.sort(key=lambda d: (-d.feasibility, d.direction_type))
    return candidates


def generate_directions(
    pool: list[Track],
    tracks_by_id: dict[str, Track],
    *,
    seed: int,
    max_directions: int = 3,
) -> list[MixCanvas]:
    """Enumerate, score, seed-rotate, and materialise concept directions over ``pool``.

    Each builder proposes a :class:`Direction` only when the material supports it.
    Survivors are sorted by ``(-feasibility, direction_type)``, then rotated by a
    seed-derived offset so different seeds/days surface different directions while the
    strongest ones appear often. Up to ``max_directions`` are materialised as
    :class:`~mixlab.models.MixCanvas` objects (via ``build_mix_canvas``) carrying the
    direction's ``brief``/``direction_type``/``thread_artist``.

    Prints one ``Direction: <type> — <title> (feasibility <f>)`` line per materialised
    direction to stdout — an intentional application-layer convenience so the run log
    records which directions fired without threading feasibility back through the caller.
    Deterministic: same pool + same seed → identical output.
    """
    candidates = _named_candidates(pool, seed=seed)
    # One diagnostic line so non-firing builders are visible in the run log —
    # genre_traverse silently returning None took three production runs to notice.
    proposed = ", ".join(sorted(d.direction_type for d in candidates)) if candidates else "none"
    print(f"Directions proposed: {proposed} ({len(candidates)}/{len(_BUILDERS)} builders)")
    if not candidates:
        return []

    candidates.sort(key=lambda d: (-d.feasibility, d.direction_type))
    rng = random.Random(seed)
    offset = rng.randrange(len(candidates))
    ordered = [candidates[(offset + i) % len(candidates)] for i in range(len(candidates))]
    picked = ordered[:max_directions]

    result: list[MixCanvas] = []
    for direction in picked:
        concept = MixConcept(title=direction.title, mood=direction.mood, track_ids=list(direction.track_ids))
        canvas = build_mix_canvas(concept, tracks_by_id)
        canvas.brief = direction.brief
        canvas.direction_type = direction.direction_type
        canvas.thread_artist = direction.thread_artist
        result.append(canvas)
        print(f"Direction: {direction.direction_type} — {direction.title} (feasibility {direction.feasibility:.2f})")
    return result
