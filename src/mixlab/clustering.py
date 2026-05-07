from __future__ import annotations

import re
import statistics
import sys

from mixlab.config import CustomGenre
from mixlab.history import ConceptHistory, similarity_to_history
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
    )


# Mirrors history._RECENCY_WINDOW and history._DECAY — keep in sync if those change.
_HIST_RECENCY = 10
_HIST_DECAY = 0.8


def _novelty_source(canvas: MixCanvas, history: ConceptHistory) -> str:
    """Return a short description of the top history contributor to novelty penalty."""
    if not history.runs or not canvas.core_track_ids:
        return "no history"
    canvas_ids = frozenset(canvas.core_track_ids)
    recent = history.runs[-_HIST_RECENCY:]
    best_sim = 0.0
    best_label = "no overlap with history"
    for age, entry in enumerate(reversed(recent)):
        hist_ids = frozenset(entry.core_track_ids)
        union = canvas_ids | hist_ids
        if not union:
            continue
        jaccard = len(canvas_ids & hist_ids) / len(union)
        decayed = jaccard * (_HIST_DECAY**age)
        if decayed > best_sim:
            best_sim = decayed
            best_label = f"run[{entry.created_at[:10]} genre={entry.genre}] jaccard_decayed={decayed:.3f}"
    return best_label


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
    print(
        f"  overlap_penalty={overlap_frac:.3f} ({overlap_count}/{core_n} core tracks shared with picked)",
        file=sys.stderr,
    )
    print(
        f"  novelty_penalty={novelty_penalty:.3f} ({_novelty_source(canvas, history)})",
        file=sys.stderr,
    )
    print(f"  risk_notes: {risk_str}", file=sys.stderr)


def score_canvas(
    canvas: MixCanvas,
    history: ConceptHistory,
    picked_ids: frozenset[str],
    *,
    debug: bool = False,
) -> CanvasScore:
    core_n = len(canvas.core_track_ids)
    technical_viability = min(1.0, core_n / 20.0)

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

    anchor_pool = len(roles.opener) + len(roles.closer)
    anchor_strength = min(1.0, anchor_pool / 8.0)

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

    overall = (
        technical_viability * 0.25
        + role_coverage * 0.25
        + anchor_strength * 0.15
        + contrast_potential * 0.15
        + distinctiveness * 0.10
        + novelty * 0.10
    )

    score = CanvasScore(
        technical_viability=technical_viability,
        role_coverage=role_coverage,
        anchor_strength=anchor_strength,
        contrast_potential=contrast_potential,
        distinctiveness=distinctiveness,
        novelty=novelty,
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
