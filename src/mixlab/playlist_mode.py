from __future__ import annotations

from difflib import get_close_matches
from statistics import median

from mixlab.clustering import (
    build_custom_genre_pool,
    camelot_compatible,
)
from mixlab.config import CustomGenre
from mixlab.models import (
    AdjacencyFragment,
    EnergyShape,
    IntentBrief,
    MixConcept,
    RiskTolerance,
    SeedAnalysis,
    SetRole,
    Track,
)

_SEED_WEIGHT: float = 2.0
_UNPLAYED_WEIGHT: float = 1.5
_MAX_PLAYLIST_POOL: int = 120
_MIN_PLAYLIST_TRACKS: int = 4

# Zone-aware pool constants
_ZONE_GAP_BPM: float = 12.0  # BPM gap that splits two distinct zones
_ZONE_MIN_TRACKS: int = 3  # minimum seed tracks for a standalone zone
_ZONE_LIB_EXPANSION: float = 8.0  # BPM expansion either side of zone for library lookup
_MAX_ZONE_LIBRARY_TRACKS: int = 20  # max library tracks added per zone


def detect_adjacency_fragments(
    seed_track_ids: list[str],
    tracks_by_id: dict[str, Track],
    bpm_threshold: float = 8.0,
) -> list[AdjacencyFragment]:
    """Find consecutive pairs in original seed order that look intentionally placed.

    A pair scores 0.9 if both harmonically compatible AND BPM-close,
    0.6 if only harmonically compatible, 0.4 if only BPM-close.
    Pairs that are neither are omitted.
    """
    fragments: list[AdjacencyFragment] = []
    for i in range(len(seed_track_ids) - 1):
        a = tracks_by_id.get(seed_track_ids[i])
        b = tracks_by_id.get(seed_track_ids[i + 1])
        if a is None or b is None:
            continue
        bpm_close = abs(a.bpm - b.bpm) <= bpm_threshold
        harmonically_ok = camelot_compatible(a.camelot_key, b.camelot_key)
        if bpm_close and harmonically_ok:
            fragments.append(
                AdjacencyFragment(
                    track_ids=[seed_track_ids[i], seed_track_ids[i + 1]],
                    confidence=0.9,
                    reason="camelot_compatible + bpm_close",
                )
            )
        elif harmonically_ok:
            fragments.append(
                AdjacencyFragment(
                    track_ids=[seed_track_ids[i], seed_track_ids[i + 1]],
                    confidence=0.6,
                    reason="camelot_compatible",
                )
            )
        elif bpm_close:
            fragments.append(
                AdjacencyFragment(
                    track_ids=[seed_track_ids[i], seed_track_ids[i + 1]],
                    confidence=0.4,
                    reason="bpm_close",
                )
            )
    return fragments


def detect_energy_shape(
    seed_track_ids: list[str],
    tracks_by_id: dict[str, Track],
) -> EnergyShape:
    """Infer energy arc shape from seed tracks' energy fields.

    Returns 'unclear' if fewer than 4 tracks have energy data or coverage < 50%.
    Splits the sequence into thirds and compares averages.
    """
    energies: list[int] = []
    for tid in seed_track_ids:
        t = tracks_by_id.get(tid)
        if t is not None and t.energy is not None:
            energies.append(t.energy)

    coverage = len(energies) / max(len(seed_track_ids), 1)
    if len(energies) < 4 or coverage < 0.5:
        return "unclear"

    n = len(energies)
    third = max(n // 3, 1)
    first_avg = sum(energies[:third]) / third
    mid_avg = sum(energies[third : 2 * third]) / third
    last_count = n - 2 * third
    last_avg = sum(energies[2 * third :]) / last_count

    if mid_avg > first_avg + 1.5 and mid_avg > last_avg + 1.5:
        return "double_peak"
    if last_avg > first_avg + 1.5:
        return "single_arc"
    if abs(first_avg - mid_avg) <= 1.0 and abs(first_avg - last_avg) <= 1.0:
        return "flat"
    return "plateau"


def identify_missing_roles(
    seed_track_ids: list[str],
    tracks_by_id: dict[str, Track],
) -> list[SetRole]:
    """Identify missing set roles based on energy fields and track count.

    Only meaningful if >= 50% of seeds have energy data and seed count >= 5.
    Returns a subset of ["opener", "peak", "groove"].
    """
    tracks = [tracks_by_id[tid] for tid in seed_track_ids if tid in tracks_by_id]
    if len(tracks) < 5:
        return []

    energies = [t.energy for t in tracks if t.energy is not None]
    if not energies or len(energies) / len(tracks) < 0.5:
        return []

    missing: list[SetRole] = []

    first_track = tracks[0]
    if first_track.energy is not None and first_track.energy > 5:
        missing.append("opener")

    if not any(e >= 7 for e in energies):
        missing.append("peak")

    if not any(4 <= e <= 6 for e in energies):
        missing.append("groove")

    return missing


def compute_deterministic_intent(
    seed_track_ids: list[str],
    tracks_by_id: dict[str, Track],
) -> IntentBrief:
    """Build a deterministic IntentBrief from seed tracks without an LLM call.

    All seeds default to 'supporting' tier. The LLM Stage 0 call will override
    tiers if successful. Energy shape, adjacencies, and BPM range are always
    computed deterministically.
    """
    tracks = [tracks_by_id[tid] for tid in seed_track_ids if tid in tracks_by_id]
    if not tracks:
        return IntentBrief(
            overall_vibe="Unknown — no valid seed tracks.",
            energy_shape="unclear",
            risk_tolerance="medium",
            is_coherent_set=True,
            seed_analyses=[],
            missing_roles=[],
            strong_adjacencies=[],
            bpm_range=(0.0, 0.0),
        )

    bpms = [t.bpm for t in tracks]
    bpm_range: tuple[float, float] = (min(bpms), max(bpms))
    bpm_spread = bpm_range[1] - bpm_range[0]

    risk_tolerance: RiskTolerance
    if bpm_spread < 10:
        risk_tolerance = "low"
    elif bpm_spread < 20:
        risk_tolerance = "medium"
    else:
        risk_tolerance = "high"

    energy_shape = detect_energy_shape(seed_track_ids, tracks_by_id)
    adjacencies = detect_adjacency_fragments(seed_track_ids, tracks_by_id)
    missing_roles = identify_missing_roles(seed_track_ids, tracks_by_id)

    analyses: list[SeedAnalysis] = [
        SeedAnalysis(
            track_id=tid,
            tier="supporting",
            inferred_role="unknown",
            drop_cost=0.5,
        )
        for tid in seed_track_ids
        if tid in tracks_by_id
    ]

    zones = cluster_seed_zones(tracks, min_zone_tracks=1)
    is_coherent_set = len(zones) <= 1

    return IntentBrief(
        overall_vibe="Analysing...",
        energy_shape=energy_shape,
        risk_tolerance=risk_tolerance,
        is_coherent_set=is_coherent_set,
        seed_analyses=analyses,
        missing_roles=missing_roles,
        strong_adjacencies=[f for f in adjacencies if f.confidence >= 0.6],
        bpm_range=bpm_range,
    )


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


def _score_candidate(
    track: Track,
    zone_seeds: list[Track],
    zone_mid_bpm: float,
    missing_roles: list[str],
    is_unplayed: bool,
    bpm_expansion: float = _ZONE_LIB_EXPANSION,
) -> float:
    """Score a library track candidate for inclusion in a zone shortlist.

    Higher is better. Weights:
    - BPM proximity: 0.35 (linear decay from zone centre to bpm_expansion boundary)
    - Camelot compatibility with any zone seed: 0.35 (binary)
    - Role filling (energy-based): 0.15
    - Unplayed bonus: 0.15

    Camelot is a score factor, not a hard filter — incompatible tracks are still eligible.
    """
    bpm_dist = abs(track.bpm - zone_mid_bpm)
    bpm_score = max(0.0, 1.0 - bpm_dist / max(bpm_expansion, 1.0))

    camelot_score = 1.0 if any(camelot_compatible(track.camelot_key, s.camelot_key) for s in zone_seeds) else 0.0

    role_score = 0.0
    if track.energy is not None and missing_roles:
        if "opener" in missing_roles and track.energy <= 3:
            # Low energy alone should not dominate opener selection; reward opener candidates
            # that also fit the seed zone's tempo, key, and general genre profile.
            same_genre = any(track.genre == seed.genre for seed in zone_seeds)
            opener_bpm_close = abs(track.bpm - zone_mid_bpm) <= 2.0
            if camelot_score == 1.0 and opener_bpm_close:
                role_score = 0.8 if same_genre else 0.65
            elif camelot_score == 1.0:
                role_score = 0.45
            else:
                role_score = 0.15
        elif "peak" in missing_roles and track.energy >= 7:
            role_score = 0.8
        elif "groove" in missing_roles and 4 <= track.energy <= 6:
            role_score = 0.5
        elif "resolution" in missing_roles and track.energy <= 4:
            role_score = 0.4

    unplayed_bonus = 0.3 if is_unplayed else 0.0

    return bpm_score * 0.35 + camelot_score * 0.35 + role_score * 0.15 + unplayed_bonus * 0.15


def build_zone_shortlists(
    seed_tracks: list[Track],
    library_tracks: list[Track],
    unplayed_ids: set[str] | None,
    all_tracks_flag: bool = False,
    intent_brief: IntentBrief | None = None,
) -> list[MixConcept]:
    """Build one MixConcept shortlist per BPM zone of the seed playlist.

    Each shortlist contains all seed tracks in that zone plus scored library
    tracks within ± _ZONE_LIB_EXPANSION BPM. Library candidates are ranked by
    a composite score: BPM proximity, Camelot compatibility, role filling, and
    unplayed bias.

    Raises ValueError if seed_tracks is too small to form a valid playlist.
    """
    if len(seed_tracks) < _MIN_PLAYLIST_TRACKS:
        raise ValueError(
            "Playlist mode requires at least "
            f"{_MIN_PLAYLIST_TRACKS} valid seed tracks with BPM and Camelot key after collection parsing."
        )

    zones = cluster_seed_zones(seed_tracks)
    seed_ids = {t.track_id for t in seed_tracks}
    missing_roles: list[str] = list(intent_brief.missing_roles) if intent_brief is not None else []
    shortlists: list[MixConcept] = []

    def _is_unplayed(t: Track) -> bool:
        if all_tracks_flag:
            return False
        if unplayed_ids is not None:
            return t.track_id in unplayed_ids
        return t.play_count == 0

    for zone in zones:
        zone_min = min(t.bpm for t in zone)
        zone_max = max(t.bpm for t in zone)
        zone_mid = (zone_min + zone_max) / 2.0
        lib_min = zone_min - _ZONE_LIB_EXPANSION
        lib_max = zone_max + _ZONE_LIB_EXPANSION

        candidates = [t for t in library_tracks if t.track_id not in seed_ids and lib_min <= t.bpm <= lib_max]
        ranked = sorted(
            candidates,
            key=lambda t: -_score_candidate(t, zone, zone_mid, missing_roles, _is_unplayed(t)),
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
