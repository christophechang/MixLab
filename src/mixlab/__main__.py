from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from mixlab.cache import load_genre_cache, save_genre_cache
from mixlab.client import fetch_mix_names, fetch_played_tracks
from mixlab.clustering import (
    build_custom_genre_pool,
    build_mix_canvas,
    count_available_by_genre,
    count_outlier_genres,
    partition_bpm_pools,
    partition_outliers,
    resolve_genre_clusters,
    select_canvases,
    sort_by_camelot,
)
from mixlab.config import CUSTOM_GENRES, GENRE_MAP, IGNORED_GENRES
from mixlab.discord_client import send_report
from mixlab.history import HistoryEntry, append_run, load_history
from mixlab.llm import (
    MAX_STAGE1_POOL_CUSTOM,
    MIN_SHORTLIST_TRACKS,
    make_cascade_state,
    select_stage1_window,
    stage0_intent_brief,
    stage1_concepts,
    stage2_curate_and_report,
    validate_stage2_output,
)
from mixlab.matcher import filter_played, filter_unplayed
from mixlab.models import MixCanvas, MixConcept, Track, TrackMode
from mixlab.playlist_exporter import (
    export_merged_xml,
    generate_merged_xml_bytes,
    parse_raw_tracks,
)
from mixlab.playlist_mode import (
    build_zone_shortlists,
    filter_tracks_for_playlist_genre,
    resolve_playlist,
)
from mixlab.reader import apply_bpm_corrections, parse_collection, parse_playlists

_XML_PATH = Path("import/rekordbox.xml")
_DO_NOT_RECOMMEND_PLAYLIST = "DO NOT RECOMMEND"


def _build_tracks_by_id(tracks: list[Track]) -> dict[str, Track]:
    return {t.track_id: t for t in tracks}


def _title_case_label(label: str) -> str:
    return label.replace("_", " ").title()


def _format_report_context(
    *,
    genre: str | None,
    playlist_name: str | None,
    mode: TrackMode,
    export_dir: Path | None,
) -> str:
    base_label: str
    details: list[str] = []

    if playlist_name is not None:
        base_label = f"{playlist_name} playlist"
        if genre is not None:
            genre_detail = _title_case_label(genre)
            if genre in CUSTOM_GENRES:
                genre_detail += " custom genre"
            details.append(genre_detail)
    elif genre is not None:
        base_label = genre if genre in CUSTOM_GENRES else _title_case_label(genre)
        if genre in CUSTOM_GENRES:
            details.append("custom genre")
    else:
        base_label = "MixLab run"

    mode_label = {"all": "All Tracks", "unplayed": "unplayed tracks", "played": "played tracks"}[mode]
    details.append(mode_label)
    if export_dir is not None:
        details.append("export enabled")

    return f"Report context: {base_label} ({', '.join(details)})"


def _load_do_not_recommend_ids(xml_path: Path) -> tuple[set[str], bool]:
    """Return (denylist_ids, playlist_found) for the DO NOT RECOMMEND playlist."""
    playlists = parse_playlists(xml_path)
    denylist_ids: set[str] = set()
    target = _DO_NOT_RECOMMEND_PLAYLIST.casefold()
    playlist_found = False

    for playlist_path, track_ids in playlists.items():
        playlist_name = playlist_path.rsplit("/", 1)[-1]
        if playlist_path.casefold() == target or playlist_name.casefold() == target:
            playlist_found = True
            denylist_ids.update(track_id for track_id in track_ids if track_id)

    return denylist_ids, playlist_found


def _apply_do_not_recommend_filter(tracks: list[Track], xml_path: Path) -> tuple[list[Track], int]:
    denylist_ids, playlist_found = _load_do_not_recommend_ids(xml_path)
    if not playlist_found:
        print(
            f'WARNING: "{_DO_NOT_RECOMMEND_PLAYLIST}" playlist not found in XML — no tracks excluded. '
            "Create a Rekordbox playlist with this exact name to enable the denylist.",
            file=sys.stderr,
        )
        return tracks, 0
    filtered_tracks = [track for track in tracks if track.track_id not in denylist_ids]
    filtered_count = len(tracks) - len(filtered_tracks)
    print(
        f"DO NOT RECOMMEND filter: {len(denylist_ids)} IDs in playlist, "
        f"excluded {filtered_count} track(s) from collection.",
        file=sys.stderr,
    )
    return filtered_tracks, filtered_count


def _apply_range_filters(
    tracks: list[Track],
    *,
    min_bpm: float | None,
    max_bpm: float | None,
    min_year: int | None,
    max_year: int | None,
) -> list[Track]:
    result = tracks
    if min_bpm is not None or max_bpm is not None:
        before = len(result)
        lo: float = min_bpm if min_bpm is not None else float("-inf")
        hi: float = max_bpm if max_bpm is not None else float("inf")
        result = [t for t in result if lo <= t.bpm <= hi]
        lo_str = f"{min_bpm:g}" if min_bpm is not None else ""
        hi_str = f"{max_bpm:g}" if max_bpm is not None else ""
        print(
            f"BPM filter [{lo_str}–{hi_str}]: excluded {before - len(result)} track(s), {len(result)} remain.",
            file=sys.stderr,
        )
    if min_year is not None or max_year is not None:
        before = len(result)
        result = [
            t
            for t in result
            if t.year is not None
            and t.year != 0
            and (min_year is None or t.year >= min_year)
            and (max_year is None or t.year <= max_year)
        ]
        lo_year = str(min_year) if min_year is not None else ""
        hi_year = str(max_year) if max_year is not None else ""
        print(
            f"Year filter [{lo_year}–{hi_year}]: excluded {before - len(result)} track(s), {len(result)} remain.",
            file=sys.stderr,
        )
    return result


def _format_pipeline_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label}={count}" for label, count in counts.items()) if counts else "none"


def _print_pipeline_summary(
    *,
    collection_count: int,
    unplayed_count: int,
    used_catalog_api: bool,
    pool_label: str = "unplayed",
    genre_cluster_counts: dict[str, int],
    bpm_filtered_counts: dict[str, int],
    same_genre_outlier_count: int,
    stage1_shortlist_count: int,
    stage2_shortlist_count: int,
) -> None:
    print("Pipeline summary:")
    filter_label = pool_label if used_catalog_api else "eligible"
    print(f"- Track pool: {unplayed_count} {filter_label} / {collection_count} in scoped collection")
    print(f"- Genre scope before BPM filter: {_format_pipeline_counts(genre_cluster_counts)}")
    print(f"- Genre scope after BPM filter: {_format_pipeline_counts(bpm_filtered_counts)}")
    print(f"- Same-genre outliers considered: {same_genre_outlier_count}")
    print(f"- Stage 1 shortlists: {stage1_shortlist_count}")
    print(f"- Stage 2 shortlists sent: {stage2_shortlist_count}")
    print()


def _print_availability(
    all_tracks: list[Track],
    unplayed: list[Track],
    show_unplayed: bool = True,
    excluded_count: int = 0,
    pool_label: str = "unplayed",
) -> tuple[dict[str, tuple[int, int]], int, dict[str, tuple[int, int]]]:
    counts = count_available_by_genre(all_tracks, unplayed, GENRE_MAP)
    outlier_genres = count_outlier_genres(all_tracks, unplayed, GENRE_MAP, ignored=IGNORED_GENRES)
    outlier_count = sum(u for _, u in outlier_genres.values())

    if show_unplayed:
        print(f"\nAvailable tracks ({pool_label} / in collection):")
        for label, (total, available) in counts.items():
            bar = "█" * min(available // 10, 30)
            print(f"  {label:<20} {available:>4} / {total:<4}  {bar}")
        if outlier_genres:
            print("\n  Unmapped Rekordbox genre tags (not in GENRE_MAP):")
            for tag, (total, available) in outlier_genres.items():
                print(f"    {tag:<26} {available:>4} / {total:<4}")
    else:
        print("\nTracks in collection:")
        max_total = max((t for t, _ in counts.values()), default=1)
        for label, (total, _) in counts.items():
            bar = "█" * round((total / max_total) * 30)
            print(f"  {label:<20} {total:<4}  {bar}")
        if outlier_genres:
            print("\n  Unmapped Rekordbox genre tags (not in GENRE_MAP):")
            for tag, (total, _) in outlier_genres.items():
                print(f"    {tag:<26} {total:<4}")
    if excluded_count:
        print(f"\n  {excluded_count} track(s) excluded (DO NOT RECOMMEND)")
    print()
    return counts, outlier_count, outlier_genres


def _show_cached_genres() -> None:
    cache = load_genre_cache()
    if cache is None:
        print("No genre cache found — run mixlab first to populate it.")
        return
    saved_at = cache["saved_at"][:16].replace("T", " ") + " UTC"
    print(f"\nGenre availability (last run: {saved_at}):")
    for label, entry in cache["counts"].items():
        bar = "█" * min(entry["unplayed"] // 10, 30)
        print(f"  {label:<20} {entry['unplayed']:>4} / {entry['total']:<4}  {bar}")
    try:
        outlier_genres = cache["outlier_genres"]
    except KeyError:
        outlier_genres = {}
    if outlier_genres:
        print("\n  Unmapped Rekordbox genre tags (not in GENRE_MAP):")
        for tag, entry in outlier_genres.items():
            print(f"    {tag:<26} {entry['unplayed']:>4} / {entry['total']:<4}")
    print()


async def run_playlist_mode(
    playlist_name: str,
    genre: str | None,
    export_dir: Path | None,
    mode: TrackMode = "unplayed",
    min_bpm: float | None = None,
    max_bpm: float | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    debug: bool = False,
) -> None:
    tracks = parse_collection(_XML_PATH)
    tracks, _ = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    tracks = apply_bpm_corrections(tracks)

    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    unplayed_ids: set[str] | None = None
    mix_names: list[str] = []
    played_track_ids_set: set[str] = set()
    if mode == "all":
        print("--mode all — skipping played-track weighting in playlist mode.")
    elif catalog_url:
        try:
            played, mix_names = await asyncio.gather(
                fetch_played_tracks(api_key, catalog_url),
                fetch_mix_names(api_key, catalog_url),
            )
        except Exception as exc:
            print(f"ERROR: Could not fetch played tracks — aborting: {exc}", file=sys.stderr)
            sys.exit(1)
        if mix_names:
            print(f"Fetched {len(mix_names)} catalogue mix name(s) — injecting into Stage 2 prompt.", flush=True)
        if mode == "unplayed":
            unplayed_ids = {track.track_id for track in filter_unplayed(tracks, played)}
        else:  # mode == "played"
            _unplayed_set = {t.track_id for t in filter_unplayed(tracks, played)}
            played_track_ids_set = {t.track_id for t in tracks if t.track_id not in _unplayed_set}
            unplayed_ids = set()  # no unplayed bonus; pool is played-only

    playlists = parse_playlists(_XML_PATH)
    try:
        raw_seed_ids = resolve_playlist(playlist_name, playlists)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    tracks_by_id = _build_tracks_by_id(tracks)
    seed_ids = frozenset(track_id for track_id in raw_seed_ids if track_id in tracks_by_id)
    seed_tracks = [tracks_by_id[track_id] for track_id in raw_seed_ids if track_id in tracks_by_id]

    cascade_state = make_cascade_state()
    seed_bpm_range: tuple[float, float] = (
        (min(t.bpm for t in seed_tracks), max(t.bpm for t in seed_tracks)) if seed_tracks else (0.0, 0.0)
    )
    valid_seed_ids = [track_id for track_id in raw_seed_ids if track_id in tracks_by_id]
    intent_brief = await stage0_intent_brief(
        seed_tracks,
        valid_seed_ids,
        cascade_state,
        seed_bpm_range,
    )
    print(
        f"Intent brief: {intent_brief.overall_vibe} | "
        f"energy: {intent_brief.energy_shape} | "
        f"risk: {intent_brief.risk_tolerance} | "
        f"anchors: {len(intent_brief.anchor_ids)} | "
        f"missing roles: {', '.join(str(r) for r in intent_brief.missing_roles) or 'none'}"
    )

    library_source = tracks
    if genre is not None:
        try:
            library_source = filter_tracks_for_playlist_genre(tracks, genre, GENRE_MAP, CUSTOM_GENRES)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Playlist mode genre filter '{genre}': {len(library_source)} tracks in scope.")

    library_tracks = [t for t in library_source if t.track_id not in seed_ids]
    library_tracks = _apply_range_filters(
        library_tracks, min_bpm=min_bpm, max_bpm=max_bpm, min_year=min_year, max_year=max_year
    )
    if mode == "played" and played_track_ids_set:
        library_tracks = [t for t in library_tracks if t.track_id in played_track_ids_set]

    try:
        shortlists = build_zone_shortlists(
            seed_tracks, library_tracks, unplayed_ids, all_tracks_flag=mode != "unplayed", intent_brief=intent_brief
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if not shortlists:
        print("No zone shortlists could be built from the seed playlist.", file=sys.stderr)
        sys.exit(1)

    t_start = time.monotonic()

    all_concepts, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name=playlist_name,
        seed_ids=seed_ids,
        seed_track_ids=[track_id for track_id in raw_seed_ids if track_id in tracks_by_id],
        unplayed_ids=unplayed_ids,
        intent_brief=intent_brief,
        used_mix_names=mix_names or None,
        debug=debug,
    )
    if not all_concepts:
        print(report, file=sys.stderr)
        sys.exit(1)

    elapsed = time.monotonic() - t_start
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    report_context = _format_report_context(
        genre=genre,
        playlist_name=playlist_name,
        mode=mode,
        export_dir=export_dir,
    )
    report += f'\n\n---\n\nPlaylist completion: "{playlist_name}"'
    report += f"\n⏱ Generated in {elapsed_str}"
    print(report_context + "\n\n" + report)

    raw_tracks_xml = parse_raw_tracks(_XML_PATH)
    today = datetime.date.today().isoformat()
    folder_name = f"Mix Lab - {playlist_name} - {today}"
    merged_bytes = generate_merged_xml_bytes(all_concepts, raw_tracks_xml, folder_name, None)
    xml_attachments: list[tuple[str, bytes]] = [("rekordbox_export.xml", merged_bytes)] if merged_bytes else []

    if export_dir is not None:
        out_path = export_merged_xml(all_concepts, raw_tracks_xml, export_dir / "rekordbox_export.xml", folder_name)
        if out_path is not None:
            print(f"Exported: {out_path}")

    await send_report(
        report,
        [all_concepts[0]],
        [],
        tracks_by_id,
        counts={},
        attachments=xml_attachments,
        show_unplayed=False,
        report_context=report_context,
        filter_desc=_build_filter_desc(min_bpm=min_bpm, max_bpm=max_bpm, min_year=min_year, max_year=max_year),
    )


async def run_export_unplayed() -> None:
    tracks = parse_collection(_XML_PATH)
    tracks, denylist_excluded = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    tracks = apply_bpm_corrections(tracks)

    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    if not catalog_url:
        print("ERROR: CATALOG_API_URL not set — cannot determine unplayed tracks.", file=sys.stderr)
        sys.exit(1)

    try:
        played = await fetch_played_tracks(api_key, catalog_url)
    except Exception as exc:
        print(f"ERROR: Could not fetch played tracks — aborting: {exc}", file=sys.stderr)
        sys.exit(1)

    unplayed = filter_unplayed(tracks, played)
    print(f"Unplayed: {len(unplayed)} / {len(tracks)} tracks.")

    unplayed_ids = [t.track_id for t in unplayed]
    raw_tracks_xml = parse_raw_tracks(_XML_PATH)
    today = datetime.date.today().isoformat()
    folder_name = f"Mix Lab - All Unplayed - {today}"

    export_dir = Path("output/playlists")
    out_path = export_merged_xml([], raw_tracks_xml, export_dir / "rekordbox_export.xml", folder_name, unplayed_ids)
    if out_path is not None:
        print(f"Exported: {out_path}")

    report = f"All unplayed tracks — {len(unplayed)} tracks across full collection."
    if denylist_excluded:
        report += f" ({denylist_excluded} excluded via DO NOT RECOMMEND)"

    merged_bytes = generate_merged_xml_bytes([], raw_tracks_xml, folder_name, unplayed_ids)
    xml_attachments: list[tuple[str, bytes]] = [("rekordbox_export.xml", merged_bytes)] if merged_bytes else []

    report_context = f"Report context: All Unplayed export ({len(unplayed)} tracks)"
    await send_report(
        report,
        [],
        [],
        report_context=report_context,
        attachments=xml_attachments,
    )


async def run(
    genre: str | None,
    export_dir: Path | None,
    mode: TrackMode = "unplayed",
    min_bpm: float | None = None,
    max_bpm: float | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    debug: bool = False,
) -> None:
    # 1. Parse collection.
    tracks = parse_collection(_XML_PATH)
    tracks, denylist_excluded = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    tracks = apply_bpm_corrections(tracks)
    tracks = _apply_range_filters(tracks, min_bpm=min_bpm, max_bpm=max_bpm, min_year=min_year, max_year=max_year)

    # 2. Fetch played tracks and filter pool based on mode.
    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    used_catalog_api = False
    mix_names: list[str] = []
    played_track_ids: set[str] = set()
    pool_label = "unplayed"
    if mode == "all":
        print("--mode all — skipping played-track filter, using full collection.")
        unplayed = list(tracks)
    elif catalog_url:
        try:
            played, mix_names = await asyncio.gather(
                fetch_played_tracks(api_key, catalog_url),
                fetch_mix_names(api_key, catalog_url),
            )
        except Exception as exc:
            print(f"ERROR: Could not fetch played tracks — aborting: {exc}", file=sys.stderr)
            sys.exit(1)
        if mix_names:
            print(f"Fetched {len(mix_names)} catalogue mix name(s) — injecting into Stage 2 prompt.", flush=True)
        if mode == "played":
            pool_label = "played"
            unplayed = filter_played(tracks, played)
            # all pool tracks are played; validation should not warn about them
        else:  # mode == "unplayed"
            unplayed = filter_unplayed(tracks, played)
            _unplayed_ids = {t.track_id for t in unplayed}
            played_track_ids = {t.track_id for t in tracks if t.track_id not in _unplayed_ids}
        used_catalog_api = True
    else:
        print("No CATALOG_API_URL set — skipping played-track filter, using full collection.")
        unplayed = list(tracks)

    # 3. Always print the availability table (deterministic, no LLM cost).
    counts, outlier_count, outlier_genres = _print_availability(
        tracks, unplayed, show_unplayed=used_catalog_api, excluded_count=denylist_excluded, pool_label=pool_label
    )
    save_genre_cache(counts, outlier_count, outlier_genres)

    # 4. If no genre specified, stop here — table is the output.
    if not genre:
        print("Specify --genre <label> to generate mix concepts.")
        all_labels = sorted(GENRE_MAP.keys()) + sorted(CUSTOM_GENRES.keys())
        print("Labels: " + ", ".join(all_labels))
        return

    # 5. Cluster and scope to the requested genre.
    is_custom = genre in CUSTOM_GENRES
    t_start = time.monotonic()
    cascade_state = make_cascade_state()
    all_shortlists: list[MixConcept] = []
    custom_genre_sub_genres: list[str] | None = None
    genre_unplayed_track_ids_source: list[Track] = []
    genre_cluster_counts: dict[str, int] = {}
    bpm_filtered_counts: dict[str, int] = {}
    same_genre_outlier_count = 0

    if is_custom:
        pool = build_custom_genre_pool(genre, unplayed, CUSTOM_GENRES, GENRE_MAP)
        if not pool:
            print(f"No tracks found for custom genre '{genre}'.", file=sys.stderr)
            sys.exit(1)
        print(f"Custom genre '{genre}': {len(pool)} tracks in pool.")
        # Sort by BPM for Stage 1 window selection — ensures each window is BPM-coherent.
        # (Camelot walk not used here: it can span large BPM gaps across sub-genres.)
        bpm_sorted_pool = sorted(pool, key=lambda t: t.bpm)
        stage1_pool = select_stage1_window(bpm_sorted_pool, MAX_STAGE1_POOL_CUSTOM)
        if len(stage1_pool) < len(pool):
            print(f"  Selected {len(stage1_pool)}-track window from pool for Stage 1 (randomised per run).")
        cfg = CUSTOM_GENRES[genre]
        custom_genre_sub_genres = cfg["genres"]
        all_shortlists.extend(await stage1_concepts(stage1_pool, genre, cascade_state, custom=True))
        genre_unplayed_track_ids_source = pool
        # No outlier handling for custom genres — pool is already the full scope.
        genre_outliers: list[Track] = []
        outliers: list[Track] = []
        genre_cluster_counts = {genre: len(pool)}
        bpm_filtered_counts = {genre: len(stage1_pool)}
    else:
        clusters, outliers = partition_outliers(unplayed, GENRE_MAP)
        clusters = resolve_genre_clusters(genre, clusters, GENRE_MAP)
        genre_cluster_counts = {genre_label: len(cluster_tracks) for genre_label, cluster_tracks in clusters.items()}

        if not clusters:
            print(f"No tracks found for genre '{genre}'.", file=sys.stderr)
            sys.exit(1)

        # 6a. LLM Stage 1 — standard path.
        for genre_label, cluster_tracks in clusters.items():
            pools = partition_bpm_pools(cluster_tracks)
            bpm_filtered_counts[genre_label] = len(pools.core)
            if len(pools.core) < MIN_SHORTLIST_TRACKS:
                print(
                    f"Stage 1: skipping {genre_label} — {len(pools.core)} tracks in core BPM pool "
                    f"(minimum {MIN_SHORTLIST_TRACKS})"
                )
                continue
            sorted_tracks = sort_by_camelot(pools.core)
            all_shortlists.extend(await stage1_concepts(sorted_tracks, genre_label, cascade_state))

        # Outliers ≥ 4 within this genre scope — shortlist as Misc.
        genre_outliers = [t for t in outliers if t.genre.lower() == genre.lower()]
        same_genre_outlier_count = len(genre_outliers)
        if len(genre_outliers) >= 4:
            all_shortlists.extend(await stage1_concepts(genre_outliers, "Misc", cascade_state))

        genre_unplayed_track_ids_source = [t for cluster_tracks in clusters.values() for t in cluster_tracks]

    tracks_by_id = _build_tracks_by_id(tracks)
    all_shortlists = [s for s in all_shortlists if any(tid in tracks_by_id for tid in s.track_ids)]
    stage1_shortlist_count = len(all_shortlists)
    if not all_shortlists:
        _print_pipeline_summary(
            collection_count=len(tracks),
            unplayed_count=len(unplayed),
            used_catalog_api=used_catalog_api,
            pool_label=pool_label,
            genre_cluster_counts=genre_cluster_counts,
            bpm_filtered_counts=bpm_filtered_counts,
            same_genre_outlier_count=same_genre_outlier_count,
            stage1_shortlist_count=0,
            stage2_shortlist_count=0,
        )
        print("No shortlists generated — all tracks may have been excluded.", file=sys.stderr)
        sys.exit(1)
    if len(all_shortlists) < 3:
        print(
            f"⚠️  Stage 1 produced only {len(all_shortlists)} shortlist(s) — pool may be too thin for 3–6 concepts.",
            file=sys.stderr,
        )

    # Build Mix Canvases and select top candidates for Stage 2 (diversity-aware, deterministic).
    history = load_history(Path(".mixlab/concept-history.json"))
    all_canvases: list[MixCanvas] = [build_mix_canvas(c, tracks_by_id) for c in all_shortlists]
    selected_canvases = select_canvases(all_canvases, history, debug=debug)
    if not selected_canvases:
        print("No canvases could be built — collection may be out of sync.", file=sys.stderr)
        sys.exit(1)
    _print_pipeline_summary(
        collection_count=len(tracks),
        unplayed_count=len(unplayed),
        used_catalog_api=used_catalog_api,
        pool_label=pool_label,
        genre_cluster_counts=genre_cluster_counts,
        bpm_filtered_counts=bpm_filtered_counts,
        same_genre_outlier_count=same_genre_outlier_count,
        stage1_shortlist_count=stage1_shortlist_count,
        stage2_shortlist_count=len(selected_canvases),
    )

    # 7. LLM Stage 2 — creative curation + full report (single Anthropic call).
    print(f"Stage 2: curating {len(selected_canvases)} canvas(es)...")
    all_concepts, report = await stage2_curate_and_report(
        [c.source_concept for c in selected_canvases],
        tracks_by_id,
        custom_genre_label=genre if is_custom else None,
        custom_genre_sub_genres=custom_genre_sub_genres,
        used_mix_names=mix_names or None,
        canvases=selected_canvases,
        debug=debug,
    )
    if not all_concepts:
        print(report, file=sys.stderr)
        sys.exit(1)

    # Post-Stage-2 validation (warn-only — never aborts the run).
    validation_warnings = validate_stage2_output(
        all_concepts,
        selected_canvases,
        tracks_by_id,
        played_ids=played_track_ids,
        denylist_ids=set(),  # tracks filtered before Stage 1; can't appear in output
        allow_played=mode in ("all", "played"),
        genre=genre or "_default",
    )
    if validation_warnings:
        print("\n⚠ Validation Notes:")
        for w in validation_warnings:
            print(f"  {w}")
        report += "\n\n⚠ **Validation Notes**\n" + "\n".join(f"- {w}" for w in validation_warnings)

    # Persist run to concept history for novelty scoring in future runs.
    try:
        entry = HistoryEntry.from_run(
            selected_canvases,
            all_concepts,
            genre=genre or "_default",
            mode={"all": "all-tracks", "played": "played", "unplayed": "standard"}[mode],
        )
        append_run(history, entry, Path(".mixlab/concept-history.json"))
    except Exception as exc:
        print(f"Warning: could not write concept history: {exc}", file=sys.stderr)
    elapsed = time.monotonic() - t_start
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    report_context = _format_report_context(
        genre=genre,
        playlist_name=None,
        mode=mode,
        export_dir=export_dir,
    )
    report += f"\n⏱ Generated in {elapsed_str}"

    print(report_context + "\n\n" + report)

    # 8. Generate merged Rekordbox XML (one file — for Discord attachment and optional disk export).
    raw_tracks_xml = parse_raw_tracks(_XML_PATH)
    today = datetime.date.today().isoformat()
    folder_name = f"Mix Lab - {genre} - {today}"
    # All unplayed tracks scoped to this genre: cluster tracks + same-genre outliers.
    # Only meaningful when we fetched played tracks from the catalogue API.
    genre_unplayed_track_ids: list[str] | None = None
    if used_catalog_api:
        genre_unplayed_track_ids = [t.track_id for t in genre_unplayed_track_ids_source]
        genre_unplayed_track_ids += [t.track_id for t in genre_outliers]
    merged_bytes = generate_merged_xml_bytes(all_concepts, raw_tracks_xml, folder_name, genre_unplayed_track_ids)
    xml_attachments: list[tuple[str, bytes]] = (
        [("rekordbox_export.xml", merged_bytes)] if merged_bytes is not None else []
    )

    if export_dir is not None:
        out_path = export_merged_xml(
            all_concepts, raw_tracks_xml, export_dir / "rekordbox_export.xml", folder_name, genre_unplayed_track_ids
        )
        if out_path is not None:
            print(f"Exported: {out_path}")

    # 9. Discord delivery.
    filtered_outliers = [t for t in outliers if t.genre not in IGNORED_GENRES]
    await send_report(
        report,
        all_concepts,
        filtered_outliers,
        tracks_by_id,
        counts=counts,
        attachments=xml_attachments,
        show_unplayed=used_catalog_api,
        report_context=report_context,
        excluded_count=denylist_excluded,
        filter_desc=_build_filter_desc(min_bpm=min_bpm, max_bpm=max_bpm, min_year=min_year, max_year=max_year),
    )


def _build_filter_desc(
    *,
    min_bpm: float | None,
    max_bpm: float | None,
    min_year: int | None,
    max_year: int | None,
) -> str | None:
    parts: list[str] = []
    if min_bpm is not None or max_bpm is not None:
        if min_bpm is not None and max_bpm is not None:
            parts.append(f"BPM {min_bpm:g}–{max_bpm:g}")
        elif min_bpm is not None:
            parts.append(f"BPM ≥ {min_bpm:g}")
        else:
            parts.append(f"BPM ≤ {max_bpm:g}")
    if min_year is not None or max_year is not None:
        if min_year is not None and max_year is not None:
            parts.append(f"year {min_year}–{max_year}")
        elif min_year is not None:
            parts.append(f"year ≥ {min_year}")
        else:
            parts.append(f"year ≤ {max_year}")
    return ", ".join(parts) if parts else None


def _validate_range_args(
    *,
    min_bpm: float | None,
    max_bpm: float | None,
    min_year: int | None,
    max_year: int | None,
) -> None:
    if min_bpm is not None and max_bpm is not None and min_bpm > max_bpm:
        print(f"ERROR: --min-bpm ({min_bpm}) must not exceed --max-bpm ({max_bpm}).", file=sys.stderr)
        sys.exit(1)
    if min_year is not None and max_year is not None and min_year > max_year:
        print(f"ERROR: --min-year ({min_year}) must not exceed --max-year ({max_year}).", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="MixLab — AI-powered DJ crate assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
standard genres:
  house, drum_and_bass, breakbeat, electronica, hip_hop,
  jungle, uk_bass, progressive, disco, techno, uk_garage

custom genres (merge multiple genres into one cross-genre pool):
  170   drum_and_bass + jungle            165–175 BPM hard filter
  140   breakbeat + uk_bass + uk_garage   130–140 BPM hard filter
  4x4   house + electronica + disco +     no BPM filter (Stage 1 groups by BPM)
        progressive + techno

  Custom genres pick a random 120-track window from the full pool each run,
  so repeated runs explore different corners of the collection.

examples:
  mixlab                              show crate availability table (no LLM)
  mixlab --genre house                generate mix concepts for house
  mixlab --playlist "Monday Night"    complete a playlist concept from seed
  mixlab --playlist "Monday Night" --genre electronica  keep added tracks within electronica
  mixlab --playlist "Sets/Monday Night"  use full folder path if name is ambiguous
  mixlab --genre 4x4 --mode all        cross-genre 4x4 set from full collection
  mixlab --genre house --mode played   house set from battle-tested played tracks
  mixlab --genre house --min-bpm 122 --max-bpm 128  narrow pool by BPM range
  mixlab --genre drum_and_bass --min-year 2020       tracks from 2020 onwards only
  mixlab --genres                     show cached counts from last run (no API)
  mixlab --export-unplayed            export all unplayed tracks as Rekordbox XML + post to Discord
""",
    )
    parser.add_argument(
        "--genre",
        type=str,
        default=None,
        metavar="LABEL",
        help="Genre to target. Standard labels: house, drum_and_bass, techno, etc. "
        "Custom labels: 170, 140, 4x4 (cross-genre pools). "
        "Also accepts a Rekordbox genre tag directly, e.g. 'Deep House'. "
        "When combined with --playlist, constrains added library tracks to that genre scope.",
    )
    parser.add_argument(
        "--playlist",
        type=str,
        default=None,
        metavar="NAME",
        help="Rekordbox playlist name (or folder/name path) to use as seed for a single playlist-completion concept.",
    )
    parser.add_argument("--genres", action="store_true", help="Show genre availability from last run (no API calls)")
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="DIR",
        help="Export merged Rekordbox XML (rekordbox_export.xml) to DIR",
    )
    parser.add_argument(
        "--export-playlists",
        action="store_true",
        help="Export merged Rekordbox XML to output/playlists/rekordbox_export.xml",
    )
    parser.add_argument(
        "--mode",
        choices=["unplayed", "all", "played"],
        default="unplayed",
        help=(
            "Track selection mode: "
            "unplayed (default) — only tracks never played live; "
            "all — full collection ignoring play history; "
            "played — only battle-tested tracks from play history."
        ),
    )
    parser.add_argument(
        "--export-unplayed",
        action="store_true",
        help="Export all unplayed tracks (rekordbox minus catalog) to output/playlists/ and post to Discord. No LLM.",
    )
    parser.add_argument(
        "--min-bpm",
        type=float,
        default=None,
        metavar="BPM",
        help="Minimum BPM (inclusive). Tracks below this value are excluded after ingestion.",
    )
    parser.add_argument(
        "--max-bpm",
        type=float,
        default=None,
        metavar="BPM",
        help="Maximum BPM (inclusive). Tracks above this value are excluded after ingestion.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Minimum release year (inclusive). Tracks with no year are excluded when this is set.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Maximum release year (inclusive). Tracks with no year are excluded when this is set.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit verbose canvas scoring diagnostics to stderr. Also enabled by MIXLAB_DEBUG_SCORE=1.",
    )
    args = parser.parse_args()
    _validate_range_args(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        min_year=args.min_year,
        max_year=args.max_year,
    )
    debug = args.debug or bool(os.environ.get("MIXLAB_DEBUG_SCORE"))
    if args.genres:
        _show_cached_genres()
        return

    if args.export_unplayed:
        asyncio.run(run_export_unplayed())
        return

    export_dir: Path | None = None
    if args.export is not None:
        export_dir = Path(args.export)
    elif args.export_playlists:
        export_dir = Path("output/playlists")

    if args.playlist:
        asyncio.run(
            run_playlist_mode(
                args.playlist,
                args.genre,
                export_dir,
                args.mode,
                min_bpm=args.min_bpm,
                max_bpm=args.max_bpm,
                min_year=args.min_year,
                max_year=args.max_year,
                debug=debug,
            )
        )
        return

    asyncio.run(
        run(
            args.genre,
            export_dir,
            args.mode,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            min_year=args.min_year,
            max_year=args.max_year,
            debug=debug,
        )
    )


if __name__ == "__main__":
    main()
