from __future__ import annotations

import re
import statistics

from mixlab.config import CustomGenre
from mixlab.models import Track

_BPM_SPREAD = 6.0

# Camelot wheel has keys 1–12, modes A (minor) and B (major).
_CAMELOT_RE = re.compile(r"^(\d{1,2})([AB])$", re.IGNORECASE)


def _camelot_number(key: str) -> int:
    m = _CAMELOT_RE.match(key)
    return int(m.group(1)) if m else 0


def _camelot_compatible(a: str, b: str) -> bool:
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
        # Find next compatible track; prefer lowest BPM among compatibles.
        compatible = [t for t in remaining if _camelot_compatible(last_key, t.camelot_key)]
        next_track = min(compatible, key=lambda t: t.bpm) if compatible else min(remaining, key=lambda t: t.bpm)
        sorted_tracks.append(next_track)
        remaining.remove(next_track)

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
