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

import datetime
import math
import random
import re
import statistics
import sys
from collections import Counter, deque
from dataclasses import dataclass

from mixlab.config import CustomGenre
from mixlab.history import ConceptHistory, similarity_breakdown_to_history, similarity_to_history
from mixlab.models import (
    BpmPools,
    CanvasRoleCandidates,
    CanvasScore,
    CanvasScoreWeights,
    ConceptAnchorCandidate,
    ConceptAnchorType,
    ContrastAssets,
    MixCanvas,
    MixConcept,
    RiskTolerance,
    Track,
    TrackMode,
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

_OPENER_MAX_ENERGY = 3
_CLOSER_MAX_ENERGY = 4
_GROOVE_ENERGY_MIN = 3
_GROOVE_ENERGY_MAX = 5
_BUILDER_ENERGY_MIN = 4
_BUILDER_ENERGY_MAX = 6
_PEAK_ENERGY_MIN = 6
_GROOVE_BPM_TOLERANCE = 2.0

# Precompiled regex for matching vocal tokens as whole words (case-insensitive)
# Matches: feat, feat., ft, ft., featuring, vocal, vocals, w/ (feat/ft with optional dot)
# Note: w/ needs special handling since / is not a word character
_VOCAL_TOKEN_PATTERN = re.compile(r"(?<![a-z])(?:feat\.?|ft\.?|featuring|vocals?|w/)(?![a-z])", re.IGNORECASE)


def _has_vocal_token(text: str) -> bool:
    return bool(_VOCAL_TOKEN_PATTERN.search(text))


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

    # Pool-relative fallback: when absolute energy thresholds yield no opener/closer
    # candidates (e.g. D&B where MIK tags everything 6-7), promote tracks at the pool's
    # minimum energy level. The least-demanding track in any pool IS a viable opener/closer
    # relative to its peers, even if high in absolute terms.
    pool_energies = sorted({t.energy for t in tracks if t.energy is not None})
    if not opener and pool_energies:
        relative_opener_max = pool_energies[0]  # strictly minimum energy in pool
        for t in tracks:
            if t.energy is not None and t.energy <= relative_opener_max and t.track_id not in opener:
                opener.append(t.track_id)
    if not closer and pool_energies:
        relative_closer_max = pool_energies[0]  # strictly minimum energy in pool
        for t in tracks:
            if t.energy is not None and t.energy <= relative_closer_max and t.track_id not in closer:
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

    if not roles.opener:
        notes.append("weak opener pool")
    if not roles.closer:
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


# Concept-anchor candidate detection (#10). Multi-signal scoring over bridge/wildcard
# tracks to surface structurally exceptional picks that Stage 2 might otherwise
# overlook. Three categories — peak, identity, structural-exception — driven by
# different signals so the tag tells Stage 2 *why* the track was flagged.
_CONCEPT_ANCHOR_THRESHOLD = 0.40
_CONCEPT_ANCHOR_PEAK_ENERGY = 6
_CONCEPT_ANCHOR_LOW_ENERGY = 3
_CONCEPT_ANCHOR_COLLECTION_RARITY_FLOOR = 20


def _concept_anchor_signals(
    track: Track,
    canvas_tracks: list[Track],
    collection: dict[str, Track],
    roles: CanvasRoleCandidates,
    dominant_camelot: str,
) -> tuple[float, ConceptAnchorType]:
    """Score a bridge/wildcard track and pick the dominant anchor category.

    Composite score in [0, 1] aggregates role-fit, within-canvas label/artist rarity,
    collection-level rarity, harmonic contrast vs the dominant key, and energy-role
    fit. Category is determined by which signal cluster is dominant — peak when the
    track sits at the top of the energy band, identity when label/artist distinctiveness
    leads, structural-exception when role-fit leads (default).
    """
    role_fit = 0.0
    if track.track_id in roles.opener:
        role_fit = max(role_fit, 0.40)
    if track.track_id in roles.closer:
        role_fit = max(role_fit, 0.40)
    if track.track_id in roles.pivot:
        role_fit = max(role_fit, 0.30)
    if track.track_id in roles.peak:
        role_fit = max(role_fit, 0.30)

    from collections import Counter

    canvas_label_counts = Counter(t.label for t in canvas_tracks if t.label)
    canvas_artist_counts = Counter(t.artist for t in canvas_tracks if t.artist)
    canvas_label_n = canvas_label_counts.get(track.label, 0)
    canvas_artist_n = canvas_artist_counts.get(track.artist, 0)
    canvas_label_rarity = max(0.0, 1.0 - max(0, canvas_label_n - 1) / 4.0) if track.label else 0.0
    canvas_artist_rarity = max(0.0, 1.0 - max(0, canvas_artist_n - 1) / 4.0) if track.artist else 0.0

    coll_label_n = sum(1 for t in collection.values() if track.label and t.label == track.label)
    coll_artist_n = sum(1 for t in collection.values() if track.artist and t.artist == track.artist)
    coll_label_rarity = max(0.0, 1.0 - max(0, coll_label_n - 1) / _CONCEPT_ANCHOR_COLLECTION_RARITY_FLOOR)
    coll_artist_rarity = max(0.0, 1.0 - max(0, coll_artist_n - 1) / _CONCEPT_ANCHOR_COLLECTION_RARITY_FLOOR)

    contrast = 0.0
    if dominant_camelot and camelot_distance(track.camelot_key, dominant_camelot) >= 3:
        contrast = 0.5

    energy_peak_signal = 0.0
    energy_low_signal = 0.0
    if track.energy is not None:
        if track.energy >= _CONCEPT_ANCHOR_PEAK_ENERGY:
            energy_peak_signal = 0.4
        elif track.energy <= _CONCEPT_ANCHOR_LOW_ENERGY:
            energy_low_signal = 0.3

    # Identity score: within-canvas rarity primary, collection rarity additive bonus.
    identity_signal = 0.7 * max(canvas_label_rarity, canvas_artist_rarity) + 0.3 * max(
        coll_label_rarity, coll_artist_rarity
    )

    composite = min(
        1.0,
        max(role_fit, identity_signal, energy_peak_signal, energy_low_signal, contrast),
    )

    # Category: pick the dominant cluster. Peak energy and identity beat plain role-fit
    # so the tag stays informative for Stage 2.
    if energy_peak_signal > 0 and energy_peak_signal >= identity_signal:
        category: ConceptAnchorType = "peak"
    elif identity_signal >= max(role_fit, contrast):
        category = "identity"
    else:
        category = "structural-exception"
    return composite, category


def _detect_concept_anchors(
    pools: BpmPools,
    tracks_by_id: dict[str, Track],
    roles: CanvasRoleCandidates,
    dominant_camelot: str,
) -> list[ConceptAnchorCandidate]:
    """Score every bridge/wildcard track and return those clearing the threshold."""
    off_core = list(pools.bridge) + list(pools.wildcard)
    if not off_core:
        return []
    canvas_tracks = list(pools.core) + off_core
    candidates: list[ConceptAnchorCandidate] = []
    for track in off_core:
        score, category = _concept_anchor_signals(track, canvas_tracks, tracks_by_id, roles, dominant_camelot)
        if score >= _CONCEPT_ANCHOR_THRESHOLD:
            candidates.append(ConceptAnchorCandidate(track_id=track.track_id, anchor_type=category))
    return candidates


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
    concept_anchor_candidates = _detect_concept_anchors(pools, tracks_by_id, roles, dominant_camelot)

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
        concept_anchor_candidates=concept_anchor_candidates,
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
    description = (
        f"run[{entry.created_at[:10]} genre={entry.genre}] "
        f"combined_decayed={breakdown.combined:.3f} "
        f"(track={breakdown.track_similarity:.3f}, shape={breakdown.shape_similarity:.3f})"
    )
    if breakdown.feedback_multiplier != 1.0:
        description += f" feedback×{breakdown.feedback_multiplier:g}"
    return description


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

# Mode-aware canvas score weights (#24). The default (`unplayed`-aligned) baseline
# below is also the v0.11 baseline used when no mode is supplied. Each mode-specific
# table shifts a small number of percentage points to align scoring with the
# mode's curation objective:
#  - unplayed: discovery — boost novelty at the cost of anchor strength
#  - played:   reassembly — boost anchor + role at the cost of distinctiveness / novelty
#  - all:      composition — boost distinctiveness at the cost of novelty
CANVAS_SCORE_WEIGHTS_BY_MODE: dict[TrackMode, CanvasScoreWeights] = {
    "unplayed": CanvasScoreWeights(
        technical_viability=0.10,
        role_coverage=0.25,
        anchor_strength=0.10,
        contrast_potential=0.15,
        distinctiveness=0.15,
        era_coherence=0.05,
        label_coherence=0.05,
        novelty=0.15,
    ),
    "played": CanvasScoreWeights(
        technical_viability=0.10,
        role_coverage=0.25,
        anchor_strength=0.25,
        contrast_potential=0.15,
        distinctiveness=0.10,
        era_coherence=0.05,
        label_coherence=0.05,
        novelty=0.05,
    ),
    "all": CanvasScoreWeights(
        technical_viability=0.10,
        role_coverage=0.25,
        anchor_strength=0.15,
        contrast_potential=0.15,
        distinctiveness=0.20,
        era_coherence=0.05,
        label_coherence=0.05,
        novelty=0.05,
    ),
}

# Default weights when no mode is supplied. Matches the v0.11 baseline (post-#20):
# anchor_strength 0.15, novelty 0.10. The unplayed-mode table is intentionally
# *not* identical — unplayed shifts 5pp from anchor to novelty because the unplayed
# pool's discovery objective benefits from a stronger novelty pull.
DEFAULT_CANVAS_WEIGHTS = CanvasScoreWeights(
    technical_viability=0.10,
    role_coverage=0.25,
    anchor_strength=0.15,
    contrast_potential=0.15,
    distinctiveness=0.15,
    era_coherence=0.05,
    label_coherence=0.05,
    novelty=0.10,
)


# Provenance of the default weights (DEFAULT_CANVAS_WEIGHTS above):
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
#  - #24 (v0.12): mode-aware weight tables now select per run via
#    CANVAS_SCORE_WEIGHTS_BY_MODE; the default above is used only when no mode
#    is supplied (legacy callers and tests).


# Risk-aware canvas score weights (#42). Layered on top of the mode table: at
# risk="medium" the mode table (or DEFAULT_CANVAS_WEIGHTS) is used unchanged —
# None here means "no override", keeping medium byte-stable with pre-#42 runs.
# risk="high" shifts weight toward contrast/novelty (the user asked to be
# surprised); risk="low" shifts weight toward role coverage and anchor safety.
CANVAS_SCORE_WEIGHTS_BY_RISK: dict[RiskTolerance, CanvasScoreWeights | None] = {
    "medium": None,
    "high": CanvasScoreWeights(
        technical_viability=0.10,
        role_coverage=0.15,
        anchor_strength=0.05,
        contrast_potential=0.25,
        distinctiveness=0.15,
        era_coherence=0.05,
        label_coherence=0.05,
        novelty=0.20,
    ),
    "low": CanvasScoreWeights(
        technical_viability=0.15,
        role_coverage=0.30,
        anchor_strength=0.20,
        contrast_potential=0.05,
        distinctiveness=0.15,
        era_coherence=0.05,
        label_coherence=0.05,
        novelty=0.05,
    ),
}


def score_canvas(
    canvas: MixCanvas,
    history: ConceptHistory,
    picked_ids: frozenset[str],
    *,
    weights: CanvasScoreWeights | None = None,
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

    w = weights if weights is not None else DEFAULT_CANVAS_WEIGHTS
    weighted = (
        technical_viability * w.technical_viability
        + role_coverage * w.role_coverage
        + anchor_strength * w.anchor_strength
        + contrast_potential * w.contrast_potential
        + distinctiveness * w.distinctiveness
        + era_coherence * w.era_coherence
        + label_coherence * w.label_coherence
        + novelty * w.novelty
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
    mode: TrackMode | None = None,
    risk: RiskTolerance = "medium",
    debug: bool = False,
) -> list[MixCanvas]:
    """Score and pick canvases using diversity-aware deterministic selection.

    ``mode`` selects the weight table from :data:`CANVAS_SCORE_WEIGHTS_BY_MODE`.
    When ``None``, falls back to :data:`DEFAULT_CANVAS_WEIGHTS` (kept stable for
    legacy callers / tests that predate the mode-aware split in #24).

    ``risk`` (#42) applies a second, higher-precedence override on top of the mode
    table: at ``risk="medium"`` (default) :data:`CANVAS_SCORE_WEIGHTS_BY_RISK` maps
    to ``None`` and the mode table is used unchanged — byte-stable with pre-#42
    behaviour. At ``"high"`` or ``"low"`` the risk table replaces the mode table
    entirely (it does not blend with it).

    The number actually selected is ``n_effective = min(n, max(3, ceil(0.75 *
    len(canvases))))`` — so a small candidate set still gets the weakest member
    dropped rather than all forwarded verbatim (3→3, 4→3, 5→4, 6→5, 8+→n). The
    greedy overlap-aware loop always runs, so novelty and distinctiveness scores
    have real consequence even when only a handful of candidates exist (#48).
    """
    if not canvases:
        return []

    weights = CANVAS_SCORE_WEIGHTS_BY_MODE[mode] if mode is not None else DEFAULT_CANVAS_WEIGHTS
    risk_weights = CANVAS_SCORE_WEIGHTS_BY_RISK[risk]
    if risk_weights is not None:
        weights = risk_weights

    n_effective = min(n, max(3, math.ceil(0.75 * len(canvases))))

    if debug:
        print(
            f"\n[DEBUG select_canvases] {len(canvases)} candidates → selecting up to "
            f"{n_effective} (cap n={n})  mode={mode or 'default'}  risk={risk}",
            file=sys.stderr,
        )
        for c in canvases:
            print(
                f"  candidate: {c.canvas_id}  core={len(c.core_track_ids)}  "
                f"bridge={len(c.bridge_track_ids)}  wildcard={len(c.wildcard_track_ids)}  "
                f"risk={c.risk_notes or 'none'}",
                file=sys.stderr,
            )

    picked: list[MixCanvas] = []
    remaining = list(canvases)
    picked_core_ids: frozenset[str] = frozenset()

    while remaining and len(picked) < n_effective:
        # Score all remaining with current overlap context
        for canvas in remaining:
            canvas.score = score_canvas(canvas, history, picked_core_ids, weights=weights)
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


# ---------------------------------------------------------------------------
# Deterministic Stage 1 — partition_pool()
# ---------------------------------------------------------------------------

MIN_SHORTLIST: int = 15
MAX_SHORTLIST: int = 25
ABSOLUTE_MIN: int = 5
MIN_POOL_COUNT: int = 3
MAX_POOL_COUNT: int = 5
MIN_CAMELOT_COMPONENT: int = 8


def _median_bpm(shortlist: list[Track]) -> float:
    return statistics.median(t.bpm for t in shortlist)


def _min_track_id(shortlist: list[Track]) -> str:
    return min(t.track_id for t in shortlist)


def _infer_shortlist_mood(tracks: list[Track]) -> tuple[str, str]:
    """Return (title, mood) for a non-empty shortlist."""
    bpm_vals = [t.bpm for t in tracks]
    bpm_lo, bpm_hi = round(min(bpm_vals)), round(max(bpm_vals))

    parsed_keys = [t.camelot_key.upper() for t in tracks if t.camelot_key and _CAMELOT_RE.match(t.camelot_key)]
    dominant_key = min(parsed_keys, key=lambda k: (-Counter(parsed_keys)[k], k)) if parsed_keys else "?"

    years = [t.year for t in tracks if t.year is not None and t.year > 0]
    era = f"{min(years)}–{max(years)}" if years else ""

    all_tags: list[str] = [tag for t in tracks for tag in t.tags]
    tag_counts = Counter(all_tags)
    top_tags = (
        ", ".join(tag_str for tag_str, _ in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2])
        if tag_counts
        else ""
    )

    parts: list[str] = [f"{bpm_lo}–{bpm_hi} BPM", dominant_key]
    if era:
        parts.append(era)
    if top_tags:
        parts.append(top_tags)

    title = " / ".join(parts)
    mood = top_tags or f"{bpm_lo}–{bpm_hi} BPM"
    return title, mood


def _find_bpm_peaks(tracks: list[Track]) -> list[list[Track]] | None:
    """Steps 1.1–1.7: histogram → smooth → peaks → merge → assign → attach outliers.

    Returns list of clusters (one per surviving peak, outliers already attached), or
    None when no peaks are found (caller should use uniform-spread fallback).
    Input must be non-empty and already sorted by (bpm, track_id).
    """
    # Step 1.1 — sort (idempotent if already sorted by caller)
    bpm_sorted = sorted(tracks, key=lambda t: (t.bpm, t.track_id))

    # Step 1.2 — histogram buckets
    min_bpm = bpm_sorted[0].bpm
    max_bpm = bpm_sorted[-1].bpm
    first_start = math.floor(min_bpm / 3) * 3
    last_start = math.floor(max_bpm / 3) * 3
    n_buckets = (last_start - first_start) // 3 + 1
    raw: list[int] = [0] * n_buckets

    for t in bpm_sorted:
        idx = int((math.floor(t.bpm / 3) * 3 - first_start) // 3)
        raw[idx] += 1

    # Step 1.3 — < 3 buckets → no peaks
    if n_buckets < 3:
        return None

    # Smooth: last_idx = last index
    last_idx = n_buckets - 1
    smoothed: list[float] = [0.0] * n_buckets
    for i in range(n_buckets):
        if i == 0:
            smoothed[0] = (raw[0] + raw[1]) / 2
        elif i == last_idx:
            smoothed[last_idx] = (raw[last_idx - 1] + raw[last_idx]) / 2
        else:
            smoothed[i] = (raw[i - 1] + raw[i] + raw[i + 1]) / 3

    # Step 1.4 — plateau-aware peak detection
    # Build runs of consecutive equal smoothed values
    peaks: list[tuple[int, float, int]] = []  # (bucket_idx, centre_bpm, raw_count)

    i = 0
    while i < n_buckets:
        j = i
        while j + 1 < n_buckets and smoothed[j + 1] == smoothed[i]:
            j += 1
        # run = [i..j]
        if i == 0 or j == last_idx:
            i = j + 1
            continue
        left_val = smoothed[i - 1]
        right_val = smoothed[j + 1]
        if smoothed[i] > left_val and smoothed[i] > right_val:
            bucket_start = first_start + i * 3
            centre = bucket_start + 1.5
            plateau_count = sum(raw[k] for k in range(i, j + 1))
            peaks.append((i, centre, plateau_count))
        i = j + 1

    if not peaks:
        return None

    # Step 1.5 — merge peaks closer than 8 BPM
    while len(peaks) >= 2:
        min_dist = None
        merge_a = merge_b = -1
        for x in range(len(peaks)):
            for y in range(x + 1, len(peaks)):
                dist = abs(peaks[x][1] - peaks[y][1])
                if dist >= 8.0:
                    continue
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    merge_a, merge_b = x, y
                elif dist == min_dist:
                    # tie: smallest combined centre sum
                    cur_sum = peaks[merge_a][1] + peaks[merge_b][1]
                    new_sum = peaks[x][1] + peaks[y][1]
                    if new_sum < cur_sum:
                        merge_a, merge_b = x, y
                    elif new_sum == cur_sum and peaks[x][0] < peaks[merge_a][0]:
                        # tie: lower-BPM peak has lower bucket index
                        merge_a, merge_b = x, y
        if min_dist is None:
            break
        pa, pb = peaks[merge_a], peaks[merge_b]
        # Keep heavier (higher raw count); tie → keep lower-BPM
        survivor = pa if pa[2] >= pb[2] else pb
        peaks = [p for idx, p in enumerate(peaks) if idx not in (merge_a, merge_b)]
        peaks.append(survivor)
        peaks.sort(key=lambda p: p[1])  # keep ascending BPM order for tie-breaks

    # Step 1.6 — assign tracks to peaks (±7 BPM window)
    peak_centres = [p[1] for p in peaks]
    clusters: list[list[Track]] = [[] for _ in peaks]
    outliers: list[Track] = []

    for t in bpm_sorted:
        candidates = [(abs(t.bpm - c), idx) for idx, c in enumerate(peak_centres) if abs(t.bpm - c) <= 7.0]
        if not candidates:
            outliers.append(t)
        else:
            candidates.sort()  # (distance, index) — lower index wins ties
            clusters[candidates[0][1]].append(t)

    # Step 1.7 — attach outliers to nearest cluster
    for t in outliers:
        nearest = min(
            range(len(peak_centres)),
            key=lambda idx: (abs(t.bpm - peak_centres[idx]), idx),
        )
        clusters[nearest].append(t)

    return clusters


def _camelot_components(tracks: list[Track]) -> list[list[Track]]:
    """Step 2: BFS Camelot sub-clustering within a BPM cluster.

    Returns list of components (≥1).  Small components (< MIN_CAMELOT_COMPONENT)
    are merged into the largest.
    """
    if not tracks:
        return []

    # Step 2.2 — sort by track_id for deterministic BFS
    sorted_t = sorted(tracks, key=lambda t: t.track_id)

    # Step 2.3 — all-unknown guard
    if all(not t.camelot_key or not _CAMELOT_RE.match(t.camelot_key) for t in sorted_t):
        return [list(sorted_t)]

    # Build adjacency using normalised keys
    id_to_track = {t.track_id: t for t in sorted_t}
    ids = [t.track_id for t in sorted_t]

    adj: dict[str, list[str]] = {tid: [] for tid in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ka = id_to_track[a].camelot_key.upper()
            kb = id_to_track[b].camelot_key.upper()
            if camelot_compatible(ka, kb):
                adj[a].append(b)
                adj[b].append(a)

    # BFS
    visited: set[str] = set()
    components: list[list[Track]] = []
    for start in ids:
        if start in visited:
            continue
        component: list[Track] = []
        q: deque[str] = deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            component.append(id_to_track[cur])
            for nb in sorted(adj[cur]):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        components.append(component)

    # Step 2.4 — merge components < MIN_CAMELOT_COMPONENT into largest
    sources = sorted(
        [c for c in components if len(c) < MIN_CAMELOT_COMPONENT],
        key=lambda c: (len(c), _min_track_id(c)),
    )
    for src in sources:
        if src not in components:
            continue
        others = [c for c in components if c is not src]
        if not others:
            break
        target = max(others, key=lambda c: (len(c), -ord(_min_track_id(c)[0]) if _min_track_id(c) else 0))
        # tie: largest by len, then lex-smallest min_track_id
        max_len = max(len(c) for c in others)
        candidates = [c for c in others if len(c) == max_len]
        target = min(candidates, key=_min_track_id)
        target.extend(src)
        components.remove(src)

    return components


def _era_split(
    shortlist: list[Track],
    *,
    per_side_min: int = 8,
) -> tuple[list[Track], list[Track]] | None:
    """Step 3: era split. Returns (era_old, era_new) or None if no split."""
    ky = [t for t in shortlist if t.year is not None and t.year > 0]
    if len(ky) / len(shortlist) < 0.60:
        return None
    # Extract (year, track) pairs with confirmed int years for typed arithmetic.
    ky_pairs: list[tuple[int, Track]] = sorted(
        ((t.year, t) for t in ky if t.year is not None),
        key=lambda p: (p[0], p[1].track_id),
    )
    k_count = len(ky_pairs)
    if k_count < 2:
        return None

    # Find largest gap
    best_gap = -1
    gap_idx = 0
    for i in range(k_count - 1):
        gap = ky_pairs[i + 1][0] - ky_pairs[i][0]
        if gap > best_gap or (gap == best_gap and i < gap_idx):
            best_gap = gap
            gap_idx = i

    if best_gap < 8:
        return None

    gap_start: int = ky_pairs[gap_idx][0]
    era_old_known = sum(1 for yr, _ in ky_pairs if yr <= gap_start)
    era_new_known = sum(1 for yr, _ in ky_pairs if yr > gap_start)

    if era_old_known < per_side_min or era_new_known < per_side_min:
        return None

    era_old = [t for t in shortlist if t.year is not None and t.year > 0 and t.year <= gap_start]
    era_new = [t for t in shortlist if t.year is not None and t.year > 0 and t.year > gap_start]
    unknowns = [t for t in shortlist if t.year is None or t.year <= 0]
    era_old = era_old + unknowns

    if len(era_old) < MIN_SHORTLIST or len(era_new) < MIN_SHORTLIST:
        return None

    return era_old, era_new


def _centrality_rank(tracks: list[Track]) -> list[Track]:
    """Rank tracks by ascending centrality, tie-broken by track_id (lexicographic).

    Centrality = ``bpm_norm + camelot_norm`` where ``bpm_norm`` is the track's absolute
    BPM deviation from the shortlist median divided by the shortlist's BPM range, and
    ``camelot_norm`` is the Camelot-wheel distance to the shortlist's dominant key
    (normalised by 7.0; 1.0 for unparseable keys or when no dominant key exists). Lower
    is more central. Used by Step 4 Attempt 3 (oversize trim) and the seeded final
    windowing spine (#48) — the ranking is identical for both callers.
    """
    parsed_keys = [t.camelot_key.upper() for t in tracks if t.camelot_key and _CAMELOT_RE.match(t.camelot_key)]
    dominant_key: str | None = min(parsed_keys, key=lambda k: (-Counter(parsed_keys)[k], k)) if parsed_keys else None
    bpm_vals = [t.bpm for t in tracks]
    bpm_range = max(bpm_vals) - min(bpm_vals)
    if bpm_range < 1e-6:
        bpm_range = 1.0
    med = _median_bpm(tracks)

    def _centrality(t: Track) -> tuple[float, str]:
        bpm_norm = abs(t.bpm - med) / bpm_range
        if dominant_key is None:
            camelot_norm = 1.0
        else:
            d = camelot_distance(t.camelot_key.upper(), dominant_key)
            camelot_norm = 1.0 if d >= 999 else d / 7.0
        return (bpm_norm + camelot_norm, t.track_id)

    return sorted(tracks, key=_centrality)


def _resize_shortlists(shortlists: list[list[Track]]) -> list[list[Track]]:
    """Step 4: sizing enforcement — 3 iterations of oversized+undersized passes,
    then pool-level MIN/MAX_POOL_COUNT adjustments.
    """
    live = list(shortlists)

    for _iter in range(3):
        changed = False

        # --- Oversized pass ---
        snap_over = sorted(
            live,
            key=lambda s: (-len(s), _min_track_id(s)),
        )
        for snap_sl in snap_over:
            if snap_sl not in live:
                continue
            if len(snap_sl) <= MAX_SHORTLIST:
                continue
            changed = True

            # Attempt 1: Camelot BFS split into exactly 2 components ≥ MIN_SHORTLIST
            comps = _camelot_components(snap_sl)
            if len(comps) == 2 and len(comps[0]) >= MIN_SHORTLIST and len(comps[1]) >= MIN_SHORTLIST:
                comps_sorted = sorted(comps, key=lambda c: (_median_bpm(c), _min_track_id(c)))
                idx = live.index(snap_sl)
                live[idx] = comps_sorted[0]
                live.insert(idx + 1, comps_sorted[1])
                continue

            # Attempt 2: era split with per_side_min=15
            era_result = _era_split(snap_sl, per_side_min=15)
            if era_result is not None:
                era_old, era_new = era_result
                idx = live.index(snap_sl)
                live[idx] = era_old
                live.insert(idx + 1, era_new)
                continue

            # Attempt 3: Camelot sector split (lower keys 1–6 vs upper keys 7–12)
            lower_sector = [t for t in snap_sl if _camelot_number(t.camelot_key.upper()) <= 6]
            upper_sector = [t for t in snap_sl if _camelot_number(t.camelot_key.upper()) > 6]
            if len(lower_sector) >= MIN_SHORTLIST and len(upper_sector) >= MIN_SHORTLIST:
                sectors_sorted = sorted(
                    [lower_sector, upper_sector],
                    key=lambda c: (_median_bpm(c), _min_track_id(c)),
                )
                idx = live.index(snap_sl)
                live[idx] = sectors_sorted[0]
                live.insert(idx + 1, sectors_sorted[1])
                continue

            # Attempt 4: centrality trim + attach remainder
            ranked = _centrality_rank(snap_sl)
            trimmed = ranked[:MAX_SHORTLIST]
            remainder = ranked[MAX_SHORTLIST:]
            idx = live.index(snap_sl)
            live[idx] = trimmed

            if remainder:
                others = [s for s in live if s is not trimmed]
                if others:
                    trimmed_med = _median_bpm(trimmed)
                    target = min(
                        others,
                        key=lambda s: (abs(_median_bpm(s) - trimmed_med), _min_track_id(s)),
                    )
                    target.extend(remainder)
                else:
                    print(
                        f"partition_pool: Attempt 4 discarded {len(remainder)} tracks (no split target).",
                        file=sys.stderr,
                    )

        # --- Undersized pass ---
        snap_under = sorted(
            live,
            key=lambda s: (len(s), _min_track_id(s)),
        )
        for snap_sl in snap_under:
            if snap_sl not in live:
                continue
            if len(snap_sl) >= MIN_SHORTLIST:
                continue
            changed = True

            others = [s for s in snap_under if s is not snap_sl and s in live]
            if not others:
                # lone shortlist — keep as-is
                continue

            # Find closest partner
            this_med = _median_bpm(snap_sl)
            closest = min(
                others,
                key=lambda s: (abs(_median_bpm(s) - this_med), _min_track_id(s)),
            )

            # Phase 1
            if len(snap_sl) + len(closest) > MAX_SHORTLIST:
                # overflow — enter Phase 2 directly with pre-Phase-1 reference BPM
                phase2_ref = this_med
                phase2_entry = "overflow"
            else:
                # merge
                if len(snap_sl) < ABSOLUTE_MIN:
                    if snap_sl in live:
                        live.remove(snap_sl)
                    continue
                merged = snap_sl + closest
                snap_idx = snap_under.index(snap_sl)
                closest_idx = snap_under.index(closest)
                lower_idx = min(snap_idx, closest_idx)
                higher_idx = max(snap_idx, closest_idx)
                # place at lower original snapshot index (by reference lookup)
                lower_sl = snap_under[lower_idx]
                higher_sl = snap_under[higher_idx]
                if lower_sl in live:
                    live[live.index(lower_sl)] = merged
                if higher_sl in live and higher_sl is not merged:
                    live.remove(higher_sl)
                if len(merged) >= MIN_SHORTLIST:
                    continue
                phase2_ref = _median_bpm(merged)
                phase2_entry = "still_small"
                snap_sl = merged  # noqa: PLW2901 — intentional reassign for Phase 2

            # Phase 2 — retry loop
            phase2_partners = sorted(
                [s for s in snap_under if s is not snap_sl and s in live],
                key=lambda s: (abs(_median_bpm(s) - phase2_ref), _min_track_id(s)),
            )
            first = True
            for partner in phase2_partners:
                if partner not in live:
                    continue
                if first and phase2_entry == "overflow":
                    first = False
                    # skip the overflowing closest partner
                    if len(snap_sl) + len(partner) > MAX_SHORTLIST:
                        continue
                first = False
                if len(snap_sl) + len(partner) > MAX_SHORTLIST:
                    continue
                # merge
                merged2 = snap_sl + partner
                if snap_sl in live:
                    snap_idx2 = live.index(snap_sl)
                else:
                    live.append(merged2)
                    if partner in live:
                        live.remove(partner)
                    snap_sl = merged2  # noqa: PLW2901
                    if len(snap_sl) >= MIN_SHORTLIST:
                        break
                    continue
                partner_idx2 = live.index(partner) if partner in live else len(live)
                if snap_idx2 <= partner_idx2:
                    live[snap_idx2] = merged2
                    if partner in live:
                        live.remove(partner)
                else:
                    live[partner_idx2] = merged2
                    live.remove(snap_sl)
                snap_sl = merged2  # noqa: PLW2901
                if len(snap_sl) >= MIN_SHORTLIST:
                    break

            # post-loop drop if below ABSOLUTE_MIN
            if snap_sl in live and len(snap_sl) < ABSOLUTE_MIN:
                live.remove(snap_sl)

        if not changed:
            break

    # Pool-level adjustments
    if len(live) > MAX_POOL_COUNT:
        while len(live) > MAX_POOL_COUNT:
            # find closest pair
            best_dist: float | None = None
            best_a = best_b = -1
            for x in range(len(live)):
                for y in range(x + 1, len(live)):
                    dist = abs(_median_bpm(live[x]) - _median_bpm(live[y]))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_a, best_b = x, y
                    elif dist == best_dist:
                        # tie-break 1: lex-min of min_track_ids
                        cur_min = min(_min_track_id(live[best_a]), _min_track_id(live[best_b]))
                        new_min = min(_min_track_id(live[x]), _min_track_id(live[y]))
                        if new_min < cur_min:
                            best_a, best_b = x, y
                        elif new_min == cur_min:
                            # tie-break 2: smallest sum of indices
                            if x + y < best_a + best_b:
                                best_a, best_b = x, y
                            elif x + y == best_a + best_b and min(x, y) < min(best_a, best_b):
                                # tie-break 3: lower-index member
                                best_a, best_b = x, y
            merged_pool = live[best_a] + live[best_b]
            lower = min(best_a, best_b)
            higher = max(best_a, best_b)
            live[lower] = merged_pool
            del live[higher]

    return live


@dataclass
class PartitionStats:
    """Diagnostics for a :func:`partition_pool_with_stats` run.

    ``overflow_counts`` has one entry per emitted shortlist (in output order): the number
    of tracks dropped by the final windowing step, or 0 when no windowing was applied
    (shortlist already within MAX_SHORTLIST). ``seed_used`` is the seed that actually
    drove the sampling — the caller-supplied value, or the date-derived default.
    """

    overflow_counts: list[int]
    seed_used: int


def _window_shortlist(shortlist: list[Track], rng: random.Random) -> tuple[list[Track], list[Track]]:
    """Window an oversized shortlist down to MAX_SHORTLIST tracks (#48).

    The MIN_SHORTLIST most-central tracks (by :func:`_centrality_rank`) form a
    deterministic spine. The remaining ``MAX_SHORTLIST - MIN_SHORTLIST`` slots are
    filled by sampling without replacement from the rest using ``rng``. Returns
    ``(window, overflow)`` where ``window`` is BPM-sorted (by ``(bpm, track_id)``) and
    ``overflow`` is the unsampled remainder. Callers only invoke this when
    ``len(shortlist) > MAX_SHORTLIST``.
    """
    ranked = _centrality_rank(shortlist)
    spine = ranked[:MIN_SHORTLIST]
    rest = ranked[MIN_SHORTLIST:]
    fill_count = MAX_SHORTLIST - MIN_SHORTLIST
    sampled = rng.sample(rest, fill_count) if len(rest) > fill_count else list(rest)
    sampled_ids = {t.track_id for t in sampled}
    overflow = [t for t in rest if t.track_id not in sampled_ids]
    window = sorted(spine + sampled, key=lambda t: (t.bpm, t.track_id))
    return window, overflow


def partition_pool_with_stats(
    tracks: list[Track],
    *,
    seed: int | None = None,
) -> tuple[list[MixConcept], PartitionStats]:
    """Partition a genre-scoped track pool into shortlists, with windowing diagnostics.

    Same behaviour as :func:`partition_pool` but also returns a :class:`PartitionStats`
    describing the seeded final windowing step (#48). Returns 3–5 MixConcepts of 15–25
    tracks each; the 15–25 contract is now enforced unconditionally — any shortlist
    exceeding MAX_SHORTLIST (including the n_groups=1 fallback's up-to-29 result and the
    pool-level merge output) is windowed via :func:`_window_shortlist`.

    Determinism: same pool + same seed → byte-identical output. When ``seed is None`` the
    seed defaults to today's date as ``YYYYMMDD`` (an int), so a given day's runs
    reproduce while different seeds explore different fills of the oversized shortlists.
    Each shortlist samples from its own derived stream ``Random(seed * 1000003 + index)``.
    """
    seed_used = seed if seed is not None else int(datetime.date.today().strftime("%Y%m%d"))

    def _finalise(final_shortlists: list[list[Track]]) -> tuple[list[MixConcept], PartitionStats]:
        concepts: list[MixConcept] = []
        overflow_counts: list[int] = []
        for idx, sl in enumerate(final_shortlists):
            if len(sl) > MAX_SHORTLIST:
                window, overflow = _window_shortlist(sl, random.Random(seed_used * 1000003 + idx))
                overflow_counts.append(len(overflow))
                emitted = window
            else:
                overflow_counts.append(0)
                emitted = sl
            title, mood = _infer_shortlist_mood(emitted)
            concepts.append(MixConcept(title=title, mood=mood, track_ids=[t.track_id for t in emitted]))
        return concepts, PartitionStats(overflow_counts=overflow_counts, seed_used=seed_used)

    if len(tracks) < ABSOLUTE_MIN:
        return [], PartitionStats(overflow_counts=[], seed_used=seed_used)

    bpm_sorted = sorted(tracks, key=lambda t: (t.bpm, t.track_id))

    if len(bpm_sorted) < MIN_SHORTLIST:
        return _finalise([bpm_sorted])

    clusters: list[list[Track]]
    if bpm_sorted[-1].bpm - bpm_sorted[0].bpm < 4:
        clusters = [bpm_sorted]
    else:
        _peaks = _find_bpm_peaks(bpm_sorted)
        if _peaks is None or len(_peaks) == 1:
            n_groups = min(MAX_POOL_COUNT, len(bpm_sorted) // MIN_SHORTLIST)
            if n_groups < 2:
                return _finalise([bpm_sorted])
            size, rem = divmod(len(bpm_sorted), n_groups)
            clusters = []
            i = 0
            for g in range(n_groups):
                end = i + size + (1 if g < rem else 0)
                clusters.append(bpm_sorted[i:end])
                i = end
        else:
            clusters = _peaks

    # Step 2: Camelot sub-clustering per BPM cluster
    shortlists: list[list[Track]] = []
    for cluster in clusters:
        shortlists.extend(_camelot_components(cluster))

    # Step 3: era split per shortlist
    expanded: list[list[Track]] = []
    for sl in shortlists:
        result = _era_split(sl)
        if result is None:
            expanded.append(sl)
        else:
            expanded.extend(result)

    # Step 4: sizing enforcement
    final = _resize_shortlists(expanded)

    # Steps 5–6: seeded windowing + mood/title + MixConcept assembly
    return _finalise(final)


def partition_pool(
    tracks: list[Track],
    *,
    seed: int | None = None,
) -> list[MixConcept]:
    """Partition a genre-scoped track pool into shortlists for Stage 2.

    Returns 3–5 MixConcepts of 15–25 tracks each. Same pool + same seed → identical
    output; a ``None`` seed defaults to today's date (see :func:`partition_pool_with_stats`,
    which this wraps and which exposes the windowing diagnostics).
    """
    concepts, _stats = partition_pool_with_stats(tracks, seed=seed)
    return concepts
