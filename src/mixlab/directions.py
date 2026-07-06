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
    thread_artist: str = "",
) -> Direction | None:
    """Cap/validate a candidate pool and compute its feasibility score.

    Returns ``None`` when the pool falls below :data:`MIN_DIRECTION_POOL` after de-dup
    and capping, or when the BPM-sorted ordering is not path-feasible. Feasibility is
    ``0.4·pool_fill + 0.3·path_ok_ratio + 0.3·signal_strength``.
    """
    pool = _dedupe(chosen)[:MAX_DIRECTION_POOL]
    if len(pool) < MIN_DIRECTION_POOL:
        return None
    ok, ratio = _path_feasible(pool)
    if not ok:
        return None
    pool_fill = min(len(pool) / MAX_DIRECTION_POOL, 1.0)
    signal_clamped = min(max(signal, 0.0), 1.0)
    feasibility = 0.4 * pool_fill + 0.3 * ratio + 0.3 * signal_clamped
    return Direction(
        direction_type=direction_type,
        title=title,
        mood=mood,
        track_ids=[t.track_id for t in pool],
        brief=brief,
        feasibility=round(feasibility, 4),
        thread_artist=thread_artist,
    )


def _balance(a: int, b: int) -> float:
    """Symmetric balance signal: 1.0 when the two counts are equal, → 0 as they diverge."""
    hi = max(a, b)
    return min(a, b) / hi if hi else 0.0


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
    signal = _balance(len(start_ranked), len(end_ranked))
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
    )


def _build_label_spotlight(pool: list[Track], *, seed: int) -> Direction | None:
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

    share = label_counts[label] / len(pool) if pool else 0.0
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
        signal=share,
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

    signal = min(1.0, len(newest_slice) / 15.0)
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
    )


_BUILDERS = (
    _build_mood_journey,
    _build_era_dialogue,
    _build_label_spotlight,
    _build_artist_thread,
    _build_energy_shape_first,
    _build_fresh_crate,
)


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
    candidates = [direction for builder in _BUILDERS if (direction := builder(pool, seed=seed)) is not None]
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
