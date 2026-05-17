"""Mix Canvas pipeline — pool partitioning, role inference, scoring, and selection.

This module turns a Stage 1 shortlist into a structured Mix Canvas: a deterministic
post-processing layer that gives Stage 2 a vocabulary of materials (BPM tiers, role
candidate pools, contrast assets, risk notes) rather than raw track lists.

Key entry points:
- ``partition_bpm_pools`` splits tracks into core/bridge/wildcard tiers around the
  cluster median BPM.
- ``build_mix_canvas`` wraps a Stage 1 ``MixConcept`` into a ``MixCanvas`` with role
  candidates, contrast assets, and risk notes attached.
- ``score_canvas`` produces a weighted scalar score across six dimensions plus two
  penalty terms (weakness_penalty, floor_multiplier).
- ``select_canvases`` greedily picks the top-N canvases with overlap-aware re-scoring.

See ``docs/architecture/mix-canvas.md`` for the full design rationale, weight
provenance, and pipeline overview.
"""

from __future__ import annotations

import math
import re
import statistics
import sys

from mixlab.config import CustomGenre
from mixlab.history import ConceptHistory, similarity_breakdown_to_history, similarity_to_history
from mixlab.models import (
    BpmPools,
    CanvasRoleCandidates,
    CanvasScore,
    ContrastAssets,
    MixCanvas,
    MixConcept,
    Track,
)

_BPM_SPREAD = 6.0

# Camelot wheel has keys 1–12, modes A (minor) and B (major).
_CAMELOT_RE = re.compile(r"^(\d{1,2})([AB])$", re.IGNORECASE)


def _camelot_number(key: str) -> int:
    m = _CAMELOT_RE.match(key)
    return int(m.group(1)) if m else 0


def camelot_compatible(a: str, b: str) -> bool:
    ma = _CAMELOT_RE.match(a)
    mb = _CAMELOT_RE.match(b)
    if not ma or not mb:
        return False
    num_a, mode_a = int(ma.group(1)), ma.group(2).upper()
    num_b, mode_b = int(mb.group(1)), mb.group(2).upper()
    # Same key
    if num_a == num_b and mode_a == mode_b:
        return True
    # ±1 adjacent, same mode (wraps 1↔12)
    if mode_a == mode_b and abs(num_a - num_b) in (1, 11):
        return True
    # Same number, opposite mode
    return bool(num_a == num_b and mode_a != mode_b)


def camelot_distance(key_a: str, key_b: str) -> int:
    """Return minimum Camelot wheel steps between two keys (0 = identical, 1 = adjacent).

    Adjacent = ±1 same ring (wraps 12↔1), or same number opposite ring.
    Returns 999 if either key is unparseable.
    """
    ma = _CAMELOT_RE.match(key_a)
    mb = _CAMELOT_RE.match(key_b)
    if not ma or not mb:
        return 999
    num_a, mode_a = int(ma.group(1)), ma.group(2).upper()
    num_b, mode_b = int(mb.group(1)), mb.group(2).upper()
    if num_a == num_b and mode_a == mode_b:
        return 0
    if num_a == num_b:  # same number, opposite ring
        return 1
    ring_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    if mode_a == mode_b:
        return ring_dist
    # Cross-ring: minimum path is same-ring distance + one ring crossing
    return ring_dist + 1


def group_by_genre(tracks: list[Track], genre_map: dict[str, list[str]]) -> dict[str, list[Track]]:
    buckets: dict[str, list[Track]] = {}
    for track in tracks:
        for _api_label, rb_genres in genre_map.items():
            if track.genre in rb_genres:
                buckets.setdefault(track.genre, []).append(track)
                break
    return buckets


def sort_by_camelot(tracks: list[Track]) -> list[Track]:
    if not tracks:
        return []

    remaining = list(tracks)
    sorted_tracks: list[Track] = [remaining.pop(0)]

    while remaining:
        last_key = sorted_tracks[-1].camelot_key
        compatible_idx = [i for i, t in enumerate(remaining) if camelot_compatible(last_key, t.camelot_key)]
        if compatible_idx:
            best_idx = min(compatible_idx, key=lambda i: remaining[i].bpm)
        else:
            best_idx = min(range(len(remaining)), key=lambda i: remaining[i].bpm)
        sorted_tracks.append(remaining.pop(best_idx))

    return sorted_tracks


def count_outlier_genres(
    all_tracks: list[Track],
    unplayed: list[Track],
    genre_map: dict[str, list[str]],
    ignored: frozenset[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Returns {rekordbox_genre_tag: (total, unplayed)} for tags not in GENRE_MAP, sorted by unplayed desc."""
    all_mapped = {g for genres in genre_map.values() for g in genres}
    skip = (ignored or frozenset()) | all_mapped
    unplayed_ids = {t.track_id for t in unplayed}
    result: dict[str, tuple[int, int]] = {}
    for track in all_tracks:
        if track.genre not in skip:
            total, avail = result.get(track.genre, (0, 0))
            result[track.genre] = (total + 1, avail + (1 if track.track_id in unplayed_ids else 0))
    return dict(sorted(result.items(), key=lambda x: x[1][1], reverse=True))


def count_available_by_genre(
    all_tracks: list[Track],
    unplayed: list[Track],
    genre_map: dict[str, list[str]],
) -> dict[str, tuple[int, int]]:
    """Returns {api_label: (total_in_collection, unplayed)} sorted by unplayed desc."""
    result: dict[str, tuple[int, int]] = {}
    for api_label, rb_genres in genre_map.items():
        rb_set = set(rb_genres)
        total = sum(1 for t in all_tracks if t.genre in rb_set)
        available = sum(1 for t in unplayed if t.genre in rb_set)
        if total > 0:
            result[api_label] = (total, available)
    return dict(sorted(result.items(), key=lambda x: x[1][1], reverse=True))


def resolve_genre_clusters(
    genre: str,
    clusters: dict[str, list[Track]],
    genre_map: dict[str, list[str]],
) -> dict[str, list[Track]]:
    """Match --genre against an API label (drum_and_bass) or a Rekordbox genre name (Drum & Bass)."""
    normalised = genre.lower().replace(" ", "_").replace("-", "_")
    if normalised in genre_map:
        rb_set = {g.lower() for g in genre_map[normalised]}
        return {k: v for k, v in clusters.items() if k.lower() in rb_set}
    # Fall back to direct Rekordbox genre name match (case-insensitive).
    return {k: v for k, v in clusters.items() if k.lower() == genre.lower()}


def partition_outliers(
    tracks: list[Track],
    genre_map: dict[str, list[str]],
) -> tuple[dict[str, list[Track]], list[Track]]:
    all_mapped: set[str] = {genre for genres in genre_map.values() for genre in genres}
    mapped_tracks = [t for t in tracks if t.genre in all_mapped]
    outliers = [t for t in tracks if t.genre not in all_mapped]
    clusters = group_by_genre(mapped_tracks, genre_map)
    return clusters, outliers


def filter_by_bpm(tracks: list[Track]) -> list[Track]:
    """Remove tracks whose BPM is more than _BPM_SPREAD from the cluster median."""
    if not tracks:
        return tracks
    median = statistics.median(t.bpm for t in tracks)
    return [t for t in tracks if abs(t.bpm - median) <= _BPM_SPREAD]


_BRIDGE_SPREAD = 12.0


def partition_bpm_pools(tracks: list[Track]) -> BpmPools:
    """Split tracks into core/bridge/wildcard tiers relative to pool BPM median."""
    if not tracks:
        return BpmPools(core=[], bridge=[], wildcard=[])
    median = statistics.median(t.bpm for t in tracks)
    core: list[Track] = []
    bridge: list[Track] = []
    wildcard: list[Track] = []
    for t in tracks:
        delta = abs(t.bpm - median)
        if delta <= _BPM_SPREAD:
            core.append(t)
        elif delta <= _BRIDGE_SPREAD:
            bridge.append(t)
        else:
            wildcard.append(t)
    return BpmPools(core=core, bridge=bridge, wildcard=wildcard)


def filter_by_bpm_range(tracks: list[Track], bpm_min: float, bpm_max: float) -> list[Track]:
    """Keep only tracks whose BPM falls within [bpm_min, bpm_max] inclusive."""
    return [t for t in tracks if bpm_min <= t.bpm <= bpm_max]


def build_custom_genre_pool(
    custom_genre_key: str,
    tracks: list[Track],
    custom_genres: dict[str, CustomGenre],
    genre_map: dict[str, list[str]],
) -> list[Track]:
    """Return all tracks belonging to a custom genre's sub-genres, with BPM range filter applied if defined."""
    cfg = custom_genres[custom_genre_key]

    # Collect all Rekordbox genre tags that belong to this custom genre.
    rb_tags: set[str] = set()
    for key in cfg["genres"]:
        for tag in genre_map.get(key, []):
            rb_tags.add(tag)

    pool = [t for t in tracks if t.genre in rb_tags]

    bpm_range = cfg["bpm_range"]
    if bpm_range is not None:
        pool = filter_by_bpm_range(pool, bpm_range[0], bpm_range[1])

    return pool


# ---------------------------------------------------------------------------
# Mix Canvas — role inference, contrast detection, risk notes, scoring
# ---------------------------------------------------------------------------

_VOCAL_TOKENS = frozenset({"feat.", "ft.", "feat", "ft", "vocal", "vocals", "w/"})
_OPENER_MAX_ENERGY = 3
_CLOSER_MAX_ENERGY = 4
_GROOVE_ENERGY_MIN = 3
_GROOVE_ENERGY_MAX = 5
_BUILDER_ENERGY_MIN = 4
_BUILDER_ENERGY_MAX = 6
_PEAK_ENERGY_MIN = 6
_GROOVE_BPM_TOLERANCE = 2.0


def _has_vocal_token(text: str) -> bool:
    lower = text.lower()
    return any(tok in lower for tok in _VOCAL_TOKENS)


def _energy_median(tracks: list[Track]) -> float | None:
    energies = [t.energy for t in tracks if t.energy is not None]
    return statistics.median(energies) if energies else None


def _infer_roles(tracks: list[Track], pools: BpmPools) -> CanvasRoleCandidates:
    if not tracks:
        return CanvasRoleCandidates(opener=[], groove_locker=[], builder=[], pivot=[], peak=[], closer=[])

    core_ids = {t.track_id for t in pools.core}
    bridge_ids = {t.track_id for t in pools.bridge}
    median_bpm = statistics.median(t.bpm for t in tracks)

    # Dominant Camelot key (most common among core pool)
    core_keys = [t.camelot_key for t in pools.core]
    dominant_camelot = max(set(core_keys), key=core_keys.count) if core_keys else ""

    opener: list[str] = []
    groove_locker: list[str] = []
    builder: list[str] = []
    pivot: list[str] = []
    peak: list[str] = []
    closer: list[str] = []

    for t in tracks:
        in_core = t.track_id in core_ids
        in_bridge = t.track_id in bridge_ids
        e = t.energy

        # Pivot: Camelot key ≥3 steps from dominant; prefer bridge
        if dominant_camelot and camelot_distance(t.camelot_key, dominant_camelot) >= 3:
            pivot.append(t.track_id)

        if e is not None:
            # Energy-based inference
            if e <= _OPENER_MAX_ENERGY:
                opener.append(t.track_id)
            if (
                in_core
                and _GROOVE_ENERGY_MIN <= e <= _GROOVE_ENERGY_MAX
                and abs(t.bpm - median_bpm) <= _GROOVE_BPM_TOLERANCE
            ):
                groove_locker.append(t.track_id)
            if in_core and _BUILDER_ENERGY_MIN <= e <= _BUILDER_ENERGY_MAX:
                builder.append(t.track_id)
            if in_core and e >= _PEAK_ENERGY_MIN:
                peak.append(t.track_id)
            if e <= _CLOSER_MAX_ENERGY:
                closer.append(t.track_id)
        else:
            # BPM-proxy fallback
            if in_bridge:
                opener.append(t.track_id)
            if in_core and abs(t.bpm - median_bpm) <= _GROOVE_BPM_TOLERANCE:
                groove_locker.append(t.track_id)
            # Peak: highest BPM quartile in core
            if in_core and t.bpm >= median_bpm + 2:
                peak.append(t.track_id)
            # Closer: lowest BPM in pool
            if t.bpm <= median_bpm - 2:
                closer.append(t.track_id)

    return CanvasRoleCandidates(
        opener=list(dict.fromkeys(opener)),
        groove_locker=list(dict.fromkeys(groove_locker)),
        builder=list(dict.fromkeys(builder)),
        pivot=list(dict.fromkeys(pivot)),
        peak=list(dict.fromkeys(peak)),
        closer=list(dict.fromkeys(closer)),
    )


def _detect_contrast(tracks: list[Track], dominant_camelot: str) -> ContrastAssets:
    energy_med = _energy_median(tracks)

    vocal: list[str] = []
    texture: list[str] = []
    darker: list[str] = []
    brighter: list[str] = []
    resets: list[str] = []

    for t in tracks:
        if _has_vocal_token(t.artist) or _has_vocal_token(t.title):
            vocal.append(t.track_id)

        if dominant_camelot and camelot_distance(t.camelot_key, dominant_camelot) >= 3:
            texture.append(t.track_id)

        if energy_med is not None and t.energy is not None:
            if t.energy < energy_med:
                darker.append(t.track_id)
            if t.energy > energy_med + 1:
                brighter.append(t.track_id)

    return ContrastAssets(
        vocal_moments=list(dict.fromkeys(vocal)),
        texture_changes=list(dict.fromkeys(texture)),
        darker_turns=list(dict.fromkeys(darker)),
        brighter_lifts=list(dict.fromkeys(brighter)),
        lower_pressure_resets=list(dict.fromkeys(resets)),
    )


def _generate_risk_notes(
    tracks: list[Track],
    pools: BpmPools,
    roles: CanvasRoleCandidates,
) -> list[str]:
    notes: list[str] = []

    if len(roles.opener) < 2:
        notes.append("weak opener pool")
    if len(roles.closer) < 2:
        notes.append("weak closer pool")

    if pools.core:
        core_bpms = [t.bpm for t in pools.core]
        if max(core_bpms) - min(core_bpms) > 10:
            notes.append("excessive BPM spread")

    if pools.core and tracks:
        core_keys = [t.camelot_key for t in pools.core]
        dominant = max(set(core_keys), key=core_keys.count)
        near_dominant = sum(1 for k in core_keys if camelot_distance(k, dominant) <= 1)
        if near_dominant / len(core_keys) > 0.60:
            notes.append("too-similar midsection")

    from collections import Counter

    artist_counts = Counter(t.artist for t in pools.core)
    if any(c >= 3 for c in artist_counts.values()):
        notes.append("over-repeated artist")

    label_counts = Counter(t.label for t in pools.core if t.label)
    if any(c >= 4 for c in label_counts.values()):
        notes.append("over-repeated label")

    energies = [t.energy for t in tracks if t.energy is not None]
    if energies and sum(1 for e in energies if e >= 6) / len(energies) > 0.75:
        notes.append("all high energy")

    return notes


# Era/label canvas dimensions (#20).
_ERA_MIN_YEAR_COVERAGE = 0.60
_ERA_SPAN_FULL = 3  # years — below this, era_coherence is 1.0
_ERA_SPAN_ZERO = 18  # years — above this, era_coherence is 0.0
_LABEL_SHARE_THRESHOLD = 0.40
_LABEL_MIN_COUNT = 5
_LABEL_SHARE_FLOOR_FOR_COHERENCE = 0.30  # coherence = (share - 0.30) / 0.40, clamped to [0, 1]
_LABEL_SHARE_FLOOR_DIVISOR = 0.40


def _compute_era_window(core: list[Track]) -> tuple[tuple[int, int] | None, float]:
    """Compute (era_window, era_coherence) for a core pool. Empty window when patchy.

    Coherence is 1.0 when the span is <= ``_ERA_SPAN_FULL`` years, decays linearly to
    0.0 by ``_ERA_SPAN_ZERO`` years. Year-coverage floor at ``_ERA_MIN_YEAR_COVERAGE``
    of the pool — below that the era signal is suppressed entirely.
    """
    if not core:
        return None, 0.0
    years = [t.year for t in core if t.year is not None and t.year > 0]
    if len(years) < _ERA_MIN_YEAR_COVERAGE * len(core):
        return None, 0.0
    lo, hi = min(years), max(years)
    span = hi - lo
    if span <= _ERA_SPAN_FULL:
        coherence = 1.0
    elif span >= _ERA_SPAN_ZERO:
        coherence = 0.0
    else:
        coherence = max(0.0, 1.0 - (span - _ERA_SPAN_FULL) / (_ERA_SPAN_ZERO - _ERA_SPAN_FULL))
    return (lo, hi), round(coherence, 4)


def _compute_dominant_label(core: list[Track]) -> tuple[str | None, float, float]:
    """Compute (dominant_label, label_share, label_coherence) for a core pool.

    Returns (None, 0.0, 0.0) when no label clears the share threshold or the absolute
    minimum count. Coherence ramps from 0.0 at the threshold up to 1.0 around 0.70+
    share so a near-monolithic canvas gets a stronger bonus than a barely-dominant one.
    """
    from collections import Counter

    labelled = [t.label for t in core if t.label]
    if not labelled:
        return None, 0.0, 0.0
    counts = Counter(labelled)
    top_label, top_count = counts.most_common(1)[0]
    share = top_count / len(labelled)
    if share < _LABEL_SHARE_THRESHOLD or top_count < _LABEL_MIN_COUNT:
        return None, 0.0, 0.0
    coherence = min(1.0, max(0.0, (share - _LABEL_SHARE_FLOOR_FOR_COHERENCE) / _LABEL_SHARE_FLOOR_DIVISOR))
    return top_label, round(share, 4), round(coherence, 4)


def _era_coherence_from_window(window: tuple[int, int] | None) -> float:
    """Coherence value for a precomputed (min_year, max_year) tuple."""
    if window is None:
        return 0.0
    span = window[1] - window[0]
    if span <= _ERA_SPAN_FULL:
        return 1.0
    if span >= _ERA_SPAN_ZERO:
        return 0.0
    return round(max(0.0, 1.0 - (span - _ERA_SPAN_FULL) / (_ERA_SPAN_ZERO - _ERA_SPAN_FULL)), 4)


def _label_coherence_from_share(share: float) -> float:
    """Coherence ramp from the share threshold up to a near-monolithic canvas."""
    if share < _LABEL_SHARE_THRESHOLD:
        return 0.0
    return round(
        min(1.0, max(0.0, (share - _LABEL_SHARE_FLOOR_FOR_COHERENCE) / _LABEL_SHARE_FLOOR_DIVISOR)),
        4,
    )


_ANCHOR_WEIGHT_PROVENANCE = 0.30
_ANCHOR_WEIGHT_RARITY = 0.25
_ANCHOR_WEIGHT_CENTRALITY = 0.20
_ANCHOR_WEIGHT_ENERGY = 0.15
_ANCHOR_WEIGHT_RECENCY = 0.10

_ANCHOR_THRESHOLD = 0.55
_ANCHOR_TOP_FRACTION = 0.20
_ANCHOR_RARITY_FLOOR = 5
_ANCHOR_CURRENT_YEAR = 2026


def _provenance_signal(t: Track) -> float:
    score = 0.0
    if t.remixer:
        score += 0.45
    if t.enrichment_confidence == "high":
        score += 0.35
    elif t.enrichment_confidence == "medium":
        score += 0.15
    if t.label:
        score += 0.20
    return min(1.0, score)


def _rarity_signal(t: Track, label_counts: dict[str, int], artist_counts: dict[str, int]) -> float:
    """Library rarity: rare label/artist scores higher, saturating at <5 other tracks."""
    label_total = label_counts.get(t.label, 0) if t.label else _ANCHOR_RARITY_FLOOR
    artist_total = artist_counts.get(t.artist, 0) if t.artist else _ANCHOR_RARITY_FLOOR
    label_rarity = max(0.0, 1.0 - max(0, label_total - 1) / _ANCHOR_RARITY_FLOOR)
    artist_rarity = max(0.0, 1.0 - max(0, artist_total - 1) / _ANCHOR_RARITY_FLOOR)
    return min(1.0, 0.5 * label_rarity + 0.5 * artist_rarity)


def _centrality_signal(t: Track, median_bpm: float, dominant_camelot: str) -> float:
    score = 0.0
    if median_bpm > 0 and abs(t.bpm - median_bpm) <= 2.0:
        score += 0.6
    if dominant_camelot and t.camelot_key == dominant_camelot:
        score += 0.4
    return min(1.0, score)


def _energy_signal(t: Track) -> float:
    if t.energy is None:
        return 0.0
    if 6 <= t.energy <= 7:
        return 0.8
    if t.energy <= 2:
        return 0.5
    return 0.0


def _recency_signal(t: Track) -> float:
    if t.year is None or t.year <= 0:
        return 0.0
    age = _ANCHOR_CURRENT_YEAR - t.year
    if age <= 3:
        return 1.0
    return 0.0


def score_anchors(
    pool: list[Track],
    tracks_by_id: dict[str, Track],
) -> dict[str, float]:
    """Compute anchor scores for every track in ``pool`` (#19).

    The score combines provenance, library rarity, pool centrality, energy
    positioning, and recency — each normalised to [0, 1] and weighted-summed.
    Library rarity uses ``tracks_by_id`` (the full collection) for label/artist
    counts. Returns ``{track_id: score}`` covering every pool track.
    """
    if not pool:
        return {}
    from collections import Counter

    label_counts: dict[str, int] = dict(Counter(t.label for t in tracks_by_id.values() if t.label))
    artist_counts: dict[str, int] = dict(Counter(t.artist for t in tracks_by_id.values() if t.artist))

    median_bpm = statistics.median(t.bpm for t in pool) if pool else 0.0
    core_keys = [t.camelot_key for t in pool]
    dominant_camelot = max(set(core_keys), key=core_keys.count) if core_keys else ""

    scores: dict[str, float] = {}
    for t in pool:
        score = (
            _provenance_signal(t) * _ANCHOR_WEIGHT_PROVENANCE
            + _rarity_signal(t, label_counts, artist_counts) * _ANCHOR_WEIGHT_RARITY
            + _centrality_signal(t, median_bpm, dominant_camelot) * _ANCHOR_WEIGHT_CENTRALITY
            + _energy_signal(t) * _ANCHOR_WEIGHT_ENERGY
            + _recency_signal(t) * _ANCHOR_WEIGHT_RECENCY
        )
        scores[t.track_id] = round(min(1.0, score), 4)
    return scores


def _select_anchor_ids(scores: dict[str, float]) -> list[str]:
    """Pick top-20% of pool by anchor score, requiring ≥ _ANCHOR_THRESHOLD absolute."""
    if not scores:
        return []
    eligible = [(tid, s) for tid, s in scores.items() if s >= _ANCHOR_THRESHOLD]
    if not eligible:
        return []
    eligible.sort(key=lambda kv: kv[1], reverse=True)
    cap = max(1, int(round(len(scores) * _ANCHOR_TOP_FRACTION)))
    return [tid for tid, _ in eligible[:cap]]


def _canvas_id(concept: MixConcept, tracks: list[Track]) -> str:
    genre = concept.mood[:8].replace(" ", "_") if concept.mood else "unknown"
    if not tracks:
        return f"{genre}_0_?"
    median_bpm = round(statistics.median(t.bpm for t in tracks), 1)
    core_keys = [t.camelot_key for t in tracks]
    dominant = max(set(core_keys), key=core_keys.count) if core_keys else "?"
    return f"{genre}_{median_bpm}_{dominant}"


def build_mix_canvas(
    concept: MixConcept,
    tracks_by_id: dict[str, Track],
) -> MixCanvas:
    tracks = [tracks_by_id[tid] for tid in concept.track_ids if tid in tracks_by_id]
    pools = partition_bpm_pools(tracks)

    core_keys = [t.camelot_key for t in pools.core]
    dominant_camelot = max(set(core_keys), key=core_keys.count) if core_keys else ""
    dominant_bpm = statistics.median(t.bpm for t in tracks) if tracks else 0.0
    bpm_values = [t.bpm for t in tracks]
    bpm_range: tuple[float, float] = (min(bpm_values), max(bpm_values)) if bpm_values else (0.0, 0.0)

    roles = _infer_roles(tracks, pools)
    contrast = _detect_contrast(tracks, dominant_camelot)
    risk_notes = _generate_risk_notes(tracks, pools, roles)

    anchor_scores = score_anchors(pools.core, tracks_by_id)
    core_anchor_ids = _select_anchor_ids(anchor_scores)

    era_window, _ = _compute_era_window(pools.core)
    dominant_label, label_share, _ = _compute_dominant_label(pools.core)

    genre = tracks[0].genre if tracks else ""

    return MixCanvas(
        canvas_id=_canvas_id(concept, tracks),
        genre=genre,
        bpm_range=bpm_range,
        dominant_bpm=dominant_bpm,
        dominant_camelot=dominant_camelot,
        core_track_ids=[t.track_id for t in pools.core],
        bridge_track_ids=[t.track_id for t in pools.bridge],
        wildcard_track_ids=[t.track_id for t in pools.wildcard],
        roles=roles,
        contrast=contrast,
        risk_notes=risk_notes,
        score=CanvasScore(),
        source_concept=concept,
        core_anchor_ids=core_anchor_ids,
        era_window=era_window,
        dominant_label=dominant_label,
        label_share=label_share,
    )


# Mirrors history._RECENCY_WINDOW and history._DECAY — keep in sync if those change.
_HIST_RECENCY = 10
_HIST_DECAY = 0.8


def _novelty_source(canvas: MixCanvas, history: ConceptHistory) -> str:
    """Return a short description of the top history contributor to novelty penalty.

    Shows the combined value alongside its track/shape components so the user can see
    when the penalty is driven by track overlap, by repeated concept shape, or both.
    """
    if not history.runs:
        return "no history"
    breakdown = similarity_breakdown_to_history(canvas, history, _HIST_RECENCY)
    if breakdown.age_of_top_match < 0 or breakdown.combined == 0.0:
        return "no overlap with history"
    entry = list(reversed(history.runs[-_HIST_RECENCY:]))[breakdown.age_of_top_match]
    return (
        f"run[{entry.created_at[:10]} genre={entry.genre}] "
        f"combined_decayed={breakdown.combined:.3f} "
        f"(track={breakdown.track_similarity:.3f}, shape={breakdown.shape_similarity:.3f})"
    )


def _emit_canvas_score_debug(
    canvas: MixCanvas,
    score: CanvasScore,
    history: ConceptHistory,
    picked_ids: frozenset[str],
) -> None:
    """Write per-canvas score diagnostics to stderr."""
    core_n = len(canvas.core_track_ids)
    overlap_count = len(picked_ids & frozenset(canvas.core_track_ids)) if canvas.core_track_ids else 0
    overlap_frac = overlap_count / core_n if core_n > 0 else 0.0
    novelty_penalty = round(1.0 - score.novelty, 3)
    risk_str = ", ".join(canvas.risk_notes) if canvas.risk_notes else "none"
    print(
        f"  core={core_n}  bridge={len(canvas.bridge_track_ids)}  wildcard={len(canvas.wildcard_track_ids)}",
        file=sys.stderr,
    )
    print(
        f"  technical_viability={score.technical_viability:.3f}  "
        f"role_coverage={score.role_coverage:.3f}  "
        f"anchor_strength={score.anchor_strength:.3f}",
        file=sys.stderr,
    )
    print(
        f"  contrast_potential={score.contrast_potential:.3f}  "
        f"distinctiveness={score.distinctiveness:.3f}  "
        f"novelty={score.novelty:.3f}  overall={score.overall:.3f}",
        file=sys.stderr,
    )
    era_str = f"{canvas.era_window[0]}-{canvas.era_window[1]}" if canvas.era_window else "—"
    label_str = f"{canvas.dominant_label} ({canvas.label_share:.2f})" if canvas.dominant_label else "—"
    print(
        f"  era_coherence={score.era_coherence:.3f} ({era_str})  "
        f"label_coherence={score.label_coherence:.3f} ({label_str})",
        file=sys.stderr,
    )
    print(
        f"  weakness_penalty={score.weakness_penalty:.3f} ({len(canvas.risk_notes)} risk note(s))  "
        f"floor_multiplier={score.floor_multiplier:.2f}",
        file=sys.stderr,
    )
    print(
        f"  overlap_penalty={overlap_frac:.3f} ({overlap_count}/{core_n} core tracks shared with picked)",
        file=sys.stderr,
    )
    print(
        f"  novelty_penalty={novelty_penalty:.3f} ({_novelty_source(canvas, history)})",
        file=sys.stderr,
    )
    print(f"  risk_notes: {risk_str}", file=sys.stderr)


_WEAKNESS_PENALTY_PER_NOTE = 0.04
_WEAKNESS_PENALTY_CAP = 0.20
_FLOOR_MULTIPLIER_BELOW_MIN_CORE = 0.5
_MIN_CORE_FOR_FULL_SCORE = 8

# Weights sum to 1.0. Tuned across v0.10 (#9) and v0.11 (#20):
#  - #9 (v0.10): distinctiveness raised from 0.10 to 0.15 at the cost of
#    technical_viability (0.25 → 0.20). Technical viability is also logarithmic
#    with saturation at 15 tracks (was linear, saturating at 20), so a 15-track
#    distinctive canvas can outrank a 30-track generic one.
#  - #20 (v0.11): era_coherence and label_coherence introduced at 0.05 each,
#    funded by halving technical_viability again (0.20 → 0.10). Canvases with a
#    tight era window or a dominant label now get a small structural bonus that
#    a uniformly-scattered pool does not — and missing year/label data is
#    treated as no signal (no penalty), so libraries with patchy metadata are
#    not punished.
_WEIGHT_TECHNICAL_VIABILITY = 0.10
_WEIGHT_ROLE_COVERAGE = 0.25
_WEIGHT_ANCHOR_STRENGTH = 0.15
_WEIGHT_CONTRAST_POTENTIAL = 0.15
_WEIGHT_DISTINCTIVENESS = 0.15
_WEIGHT_ERA_COHERENCE = 0.05
_WEIGHT_LABEL_COHERENCE = 0.05
_WEIGHT_NOVELTY = 0.10


def score_canvas(
    canvas: MixCanvas,
    history: ConceptHistory,
    picked_ids: frozenset[str],
    *,
    debug: bool = False,
) -> CanvasScore:
    core_n = len(canvas.core_track_ids)
    # Logarithmic saturation: 1.0 at 15 tracks, ~0.79 at 8, ~0.5 at 3, 0.0 at 0.
    # A 30-track pool no longer outranks a 15-track pool on this dimension.
    technical_viability = min(1.0, math.log(core_n + 1) / math.log(16)) if core_n > 0 else 0.0

    roles = canvas.roles
    roles_filled = sum(
        1
        for r in (
            roles.opener,
            roles.groove_locker,
            roles.builder,
            roles.pivot,
            roles.peak,
            roles.closer,
        )
        if r
    )
    role_coverage = roles_filled / 6.0

    # Anchor strength: presence-based, not volume-based. Rewards canvases that have
    # BOTH an opener and a closer candidate equally over canvases with only one.
    anchor_strength = (0.5 if roles.opener else 0.0) + (0.5 if roles.closer else 0.0)

    contrast_n = (
        len(canvas.contrast.vocal_moments)
        + len(canvas.contrast.texture_changes)
        + len(canvas.contrast.darker_turns)
        + len(canvas.contrast.brighter_lifts)
    )
    contrast_potential = min(1.0, contrast_n / 4.0)

    # Distinctiveness vs already-picked canvases
    if picked_ids and canvas.core_track_ids:
        shared = len(picked_ids & frozenset(canvas.core_track_ids))
        distinctiveness = 1.0 - shared / len(canvas.core_track_ids)
    else:
        distinctiveness = 1.0

    novelty = 1.0 - similarity_to_history(canvas, history)

    # Era and label coherence (#20). era_window/dominant_label/label_share are populated
    # by build_mix_canvas. Missing data → coherence 0.0 (no penalty, just no bonus).
    era_coherence = _era_coherence_from_window(canvas.era_window)
    label_coherence = _label_coherence_from_share(canvas.label_share) if canvas.dominant_label else 0.0

    weighted = (
        technical_viability * _WEIGHT_TECHNICAL_VIABILITY
        + role_coverage * _WEIGHT_ROLE_COVERAGE
        + anchor_strength * _WEIGHT_ANCHOR_STRENGTH
        + contrast_potential * _WEIGHT_CONTRAST_POTENTIAL
        + distinctiveness * _WEIGHT_DISTINCTIVENESS
        + era_coherence * _WEIGHT_ERA_COHERENCE
        + label_coherence * _WEIGHT_LABEL_COHERENCE
        + novelty * _WEIGHT_NOVELTY
    )

    # Weakness penalty: each flagged risk note shaves 0.04 off the weighted sum, capped at 0.20.
    # Pairs with the existing risk-note diagnostics so canvases with structural problems lose
    # ground to clean canvases of similar size.
    weakness_penalty = min(_WEAKNESS_PENALTY_CAP, len(canvas.risk_notes) * _WEAKNESS_PENALTY_PER_NOTE)

    # Floor multiplier: canvases with fewer than 8 core tracks are heavily deprioritised,
    # not just proportionally smaller. They survive only when every other dimension is strong.
    floor_multiplier = _FLOOR_MULTIPLIER_BELOW_MIN_CORE if core_n < _MIN_CORE_FOR_FULL_SCORE else 1.0

    overall = max(0.0, (weighted - weakness_penalty) * floor_multiplier)

    score = CanvasScore(
        technical_viability=technical_viability,
        role_coverage=role_coverage,
        anchor_strength=anchor_strength,
        contrast_potential=contrast_potential,
        distinctiveness=distinctiveness,
        novelty=novelty,
        era_coherence=era_coherence,
        label_coherence=label_coherence,
        weakness_penalty=weakness_penalty,
        floor_multiplier=floor_multiplier,
        overall=overall,
    )
    if debug:
        print(f"[DEBUG canvas:{canvas.canvas_id}]", file=sys.stderr)
        _emit_canvas_score_debug(canvas, score, history, picked_ids)
    return score


def select_canvases(
    canvases: list[MixCanvas],
    history: ConceptHistory,
    n: int = 6,
    *,
    debug: bool = False,
) -> list[MixCanvas]:
    """Score and pick up to n canvases using diversity-aware deterministic selection."""
    if not canvases:
        return []

    if debug:
        print(
            f"\n[DEBUG select_canvases] {len(canvases)} candidates → selecting up to {n}",
            file=sys.stderr,
        )
        for c in canvases:
            print(
                f"  candidate: {c.canvas_id}  core={len(c.core_track_ids)}  "
                f"bridge={len(c.bridge_track_ids)}  wildcard={len(c.wildcard_track_ids)}  "
                f"risk={c.risk_notes or 'none'}",
                file=sys.stderr,
            )

    if len(canvases) <= n:
        for canvas in canvases:
            canvas.score = score_canvas(canvas, history, frozenset())
        canvases.sort(key=lambda c: c.score.overall, reverse=True)
        if debug:
            print("\n[DEBUG final selection order (all candidates fit)]", file=sys.stderr)
            for i, c in enumerate(canvases, 1):
                print(f"\n[DEBUG pick #{i}] {c.canvas_id}  overall={c.score.overall:.3f}", file=sys.stderr)
                _emit_canvas_score_debug(c, c.score, history, frozenset())
        return canvases

    picked: list[MixCanvas] = []
    remaining = list(canvases)
    picked_core_ids: frozenset[str] = frozenset()

    while remaining and len(picked) < n:
        # Score all remaining with current overlap context
        for canvas in remaining:
            canvas.score = score_canvas(canvas, history, picked_core_ids)
        remaining.sort(key=lambda c: c.score.overall, reverse=True)
        best = remaining.pop(0)
        picked.append(best)
        pre_pick_ids = picked_core_ids
        picked_core_ids = picked_core_ids | frozenset(best.core_track_ids)
        if debug:
            print(f"\n[DEBUG pick #{len(picked)}] {best.canvas_id}  overall={best.score.overall:.3f}", file=sys.stderr)
            _emit_canvas_score_debug(best, best.score, history, pre_pick_ids)

    if debug:
        print("\n[DEBUG final selection order]", file=sys.stderr)
        for i, c in enumerate(picked, 1):
            print(f"  #{i} {c.canvas_id}", file=sys.stderr)

    return picked
