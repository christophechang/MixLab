from __future__ import annotations

from difflib import get_close_matches
from statistics import median

from mixlab.clustering import build_custom_genre_pool
from mixlab.config import CustomGenre
from mixlab.models import MixConcept, Track

_SEED_WEIGHT: float = 2.0
_UNPLAYED_WEIGHT: float = 1.5
_MAX_PLAYLIST_POOL: int = 120
_MIN_PLAYLIST_TRACKS: int = 4

# Zone-aware pool constants
_ZONE_GAP_BPM: float = 12.0  # BPM gap that splits two distinct zones
_ZONE_MIN_TRACKS: int = 3  # minimum seed tracks for a standalone zone
_ZONE_LIB_EXPANSION: float = 8.0  # BPM expansion either side of zone for library lookup
_MAX_ZONE_LIBRARY_TRACKS: int = 20  # max library tracks added per zone


def resolve_playlist(name: str, playlists: dict[str, list[str]]) -> list[str]:
    lower_name = name.lower()
    exact_matches = {key.lower(): value for key, value in playlists.items()}
    if lower_name in exact_matches:
        return exact_matches[lower_name]

    suffix_matches = [
        (key, track_ids)
        for key, track_ids in playlists.items()
        if key.lower() == lower_name or key.lower().endswith(f"/{lower_name}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0][1]
    if len(suffix_matches) > 1:
        matches = "\n".join(f"  {key}" for key, _track_ids in suffix_matches)
        raise ValueError(
            f'Ambiguous playlist name "{name}". Found in multiple folders:\n{matches}\n'
            "Pass the full path to disambiguate."
        )

    suggestion_candidates = list(
        {key.lower() for key in playlists} | {key.lower().rsplit("/", 1)[-1] for key in playlists}
    )
    suggestions = get_close_matches(lower_name, suggestion_candidates, n=5, cutoff=0.4)
    if suggestions:
        raise ValueError(f'Playlist "{name}" not found. Did you mean: {", ".join(suggestions)}?')
    raise ValueError(f'Playlist "{name}" not found.')


def build_playlist_pool(
    seed_track_ids: list[str],
    all_tracks: list[Track],
    tracks_by_id: dict[str, Track],
    unplayed_ids: set[str] | None,
    all_tracks_flag: bool = False,
    bpm_expansion: float = 15.0,
) -> list[Track]:
    seed_tracks = [tracks_by_id[track_id] for track_id in seed_track_ids if track_id in tracks_by_id]
    if len(seed_tracks) < _MIN_PLAYLIST_TRACKS:
        raise ValueError(
            "Playlist mode requires at least "
            f"{_MIN_PLAYLIST_TRACKS} valid seed tracks with BPM and Camelot key after collection parsing."
        )

    bpm_min = min(track.bpm for track in seed_tracks) - bpm_expansion
    bpm_max = max(track.bpm for track in seed_tracks) + bpm_expansion
    seed_median_bpm = median(track.bpm for track in seed_tracks)
    seed_ids = {track.track_id for track in seed_tracks}
    library_tracks = [
        track for track in all_tracks if track.track_id not in seed_ids and bpm_min <= track.bpm <= bpm_max
    ]

    def is_unplayed(track: Track) -> bool:
        if unplayed_ids is not None:
            return track.track_id in unplayed_ids
        return track.play_count == 0

    def score(track: Track) -> float:
        score_value = 1.0
        if track.track_id in seed_ids:
            score_value *= _SEED_WEIGHT
        if not all_tracks_flag and is_unplayed(track):
            score_value *= _UNPLAYED_WEIGHT
        return score_value

    ranked_library_tracks = sorted(
        library_tracks,
        key=lambda track: (-score(track), abs(track.bpm - seed_median_bpm), track.artist, track.title),
    )
    max_library_tracks = max(0, _MAX_PLAYLIST_POOL - len(seed_tracks))
    return seed_tracks + ranked_library_tracks[:max_library_tracks]


def cluster_seed_zones(
    seed_tracks: list[Track],
    gap_bpm: float = _ZONE_GAP_BPM,
    min_zone_tracks: int = _ZONE_MIN_TRACKS,
) -> list[list[Track]]:
    """Cluster seed tracks into natural BPM zones.

    Splits at gaps > gap_bpm BPM. Zones below min_zone_tracks are merged
    into their nearest adjacent zone. Returns zones in ascending BPM order.
    """
    if not seed_tracks:
        return []

    sorted_tracks = sorted(seed_tracks, key=lambda t: t.bpm)

    zones: list[list[Track]] = []
    current: list[Track] = [sorted_tracks[0]]
    for track in sorted_tracks[1:]:
        if track.bpm - current[-1].bpm > gap_bpm:
            zones.append(current)
            current = [track]
        else:
            current.append(track)
    zones.append(current)

    # Merge undersized zones into nearest neighbour
    merged = True
    while merged and len(zones) > 1:
        merged = False
        for i, zone in enumerate(zones):
            if len(zone) < min_zone_tracks:
                if i == 0:
                    neighbour = 1
                elif i == len(zones) - 1:
                    neighbour = i - 1
                else:
                    left_gap = zones[i][0].bpm - zones[i - 1][-1].bpm
                    right_gap = zones[i + 1][0].bpm - zones[i][-1].bpm
                    neighbour = i - 1 if left_gap <= right_gap else i + 1
                merged_zone = sorted(zones[i] + zones[neighbour], key=lambda t: t.bpm)
                keep = min(i, neighbour)
                zones = [z for j, z in enumerate(zones) if j not in (i, neighbour)]
                zones.insert(keep, merged_zone)
                merged = True
                break

    return zones


def build_zone_shortlists(
    seed_tracks: list[Track],
    library_tracks: list[Track],
    unplayed_ids: set[str] | None,
    all_tracks_flag: bool = False,
) -> list[MixConcept]:
    """Build one MixConcept shortlist per BPM zone of the seed playlist.

    Each shortlist contains all seed tracks in that zone plus scored library
    tracks within ± _ZONE_LIB_EXPANSION BPM. Raises ValueError if seed_tracks
    is too small to form a valid playlist.
    """
    if len(seed_tracks) < _MIN_PLAYLIST_TRACKS:
        raise ValueError(
            "Playlist mode requires at least "
            f"{_MIN_PLAYLIST_TRACKS} valid seed tracks with BPM and Camelot key after collection parsing."
        )

    zones = cluster_seed_zones(seed_tracks)
    seed_ids = {t.track_id for t in seed_tracks}
    shortlists: list[MixConcept] = []

    def _unplayed_bonus(t: Track) -> float:
        if all_tracks_flag:
            return 1.0
        if unplayed_ids is not None:
            return _UNPLAYED_WEIGHT if t.track_id in unplayed_ids else 1.0
        return _UNPLAYED_WEIGHT if t.play_count == 0 else 1.0

    for zone in zones:
        zone_min = min(t.bpm for t in zone)
        zone_max = max(t.bpm for t in zone)
        zone_mid = (zone_min + zone_max) / 2.0
        lib_min = zone_min - _ZONE_LIB_EXPANSION
        lib_max = zone_max + _ZONE_LIB_EXPANSION

        candidates = [t for t in library_tracks if t.track_id not in seed_ids and lib_min <= t.bpm <= lib_max]
        ranked = sorted(
            candidates,
            key=lambda t: (-_unplayed_bonus(t), abs(t.bpm - zone_mid), t.artist, t.title),
        )[:_MAX_ZONE_LIBRARY_TRACKS]

        all_zone = sorted(zone + ranked, key=lambda t: t.bpm)
        title = f"Zone: {zone_min:.0f}–{zone_max:.0f} BPM ({len(zone)} seed tracks)"
        mood = f"{len(zone)} seed tracks, {len(ranked)} library additions"
        shortlists.append(MixConcept(title=title, mood=mood, track_ids=[t.track_id for t in all_zone]))

    return shortlists


def filter_tracks_for_playlist_genre(
    tracks: list[Track],
    genre: str,
    genre_map: dict[str, list[str]],
    custom_genres: dict[str, CustomGenre],
) -> list[Track]:
    normalised = genre.lower().replace(" ", "_").replace("-", "_")
    if normalised in custom_genres:
        return build_custom_genre_pool(normalised, tracks, custom_genres, genre_map)
    if normalised in genre_map:
        allowed_genres = set(genre_map[normalised])
        return [track for track in tracks if track.genre in allowed_genres]

    direct_matches = [track for track in tracks if track.genre.lower() == genre.lower()]
    if direct_matches:
        return direct_matches

    raise ValueError(f"No tracks found for genre filter '{genre}'.")
