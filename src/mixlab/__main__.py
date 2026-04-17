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
from mixlab.client import fetch_played_tracks
from mixlab.clustering import (
    build_custom_genre_pool,
    count_available_by_genre,
    count_outlier_genres,
    filter_by_bpm,
    partition_outliers,
    resolve_genre_clusters,
    sort_by_camelot,
)
from mixlab.config import CUSTOM_GENRES, GENRE_MAP, IGNORED_GENRES
from mixlab.discord_client import send_report
from mixlab.llm import (
    _MIN_SHORTLIST_TRACKS,
    make_cascade_state,
    select_shortlists_for_stage2,
    select_stage1_window,
    stage0_intent_brief,
    stage1_concepts,
    stage2_curate_and_report,
)
from mixlab.matcher import filter_unplayed
from mixlab.models import MixConcept, Track
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
    all_tracks: bool,
    stage2_provider: str | None,
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

    details.append("All Tracks" if all_tracks else "unplayed tracks")
    if stage2_provider is not None:
        details.append(f"stage 2: {stage2_provider}")
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
        f'DO NOT RECOMMEND filter: {len(denylist_ids)} IDs in playlist, '
        f'excluded {filtered_count} track(s) from collection.',
        file=sys.stderr,
    )
    return filtered_tracks, filtered_count


def _format_pipeline_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label}={count}" for label, count in counts.items()) if counts else "none"


def _print_pipeline_summary(
    *,
    collection_count: int,
    unplayed_count: int,
    used_catalog_api: bool,
    genre_cluster_counts: dict[str, int],
    bpm_filtered_counts: dict[str, int],
    same_genre_outlier_count: int,
    stage1_shortlist_count: int,
    stage2_shortlist_count: int,
) -> None:
    print("Pipeline summary:")
    filter_label = "unplayed" if used_catalog_api else "eligible"
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
) -> tuple[dict[str, tuple[int, int]], int, dict[str, tuple[int, int]]]:
    counts = count_available_by_genre(all_tracks, unplayed, GENRE_MAP)
    outlier_genres = count_outlier_genres(all_tracks, unplayed, GENRE_MAP, ignored=IGNORED_GENRES)
    outlier_count = sum(u for _, u in outlier_genres.values())

    if show_unplayed:
        print("\nAvailable tracks (unplayed / in collection):")
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
    stage2_provider: str | None,
    all_tracks: bool,
) -> None:
    tracks = parse_collection(_XML_PATH)
    tracks, _ = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    tracks = apply_bpm_corrections(tracks)

    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    unplayed_ids: set[str] | None = None
    if all_tracks:
        print("--all-tracks set — skipping played-track weighting in playlist mode.")
    elif catalog_url:
        try:
            played = await fetch_played_tracks(api_key, catalog_url)
        except Exception as exc:
            print(f"ERROR: Could not fetch played tracks — aborting: {exc}", file=sys.stderr)
            sys.exit(1)
        unplayed_ids = {track.track_id for track in filter_unplayed(tracks, played)}

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

    try:
        shortlists = build_zone_shortlists(seed_tracks, library_tracks, unplayed_ids, all_tracks, intent_brief)
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
        stage2_provider,
        playlist_name=playlist_name,
        seed_ids=seed_ids,
        seed_track_ids=[track_id for track_id in raw_seed_ids if track_id in tracks_by_id],
        unplayed_ids=unplayed_ids,
        intent_brief=intent_brief,
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
        all_tracks=all_tracks,
        stage2_provider=stage2_provider,
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
    )


async def run(
    genre: str | None,
    duration: int | None,
    export_dir: Path | None,
    stage2_provider: str | None = None,
    all_tracks: bool = False,
) -> None:  # noqa: ARG001 — duration reserved
    # 1. Parse collection.
    tracks = parse_collection(_XML_PATH)
    tracks, denylist_excluded = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    tracks = apply_bpm_corrections(tracks)

    # 2. Fetch played tracks and filter (skipped when --all-tracks is set).
    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    used_catalog_api = False
    if all_tracks:
        print("--all-tracks set — skipping played-track filter, using full collection.")
        unplayed = list(tracks)
    elif catalog_url:
        try:
            played = await fetch_played_tracks(api_key, catalog_url)
        except Exception as exc:
            print(f"ERROR: Could not fetch played tracks — aborting: {exc}", file=sys.stderr)
            sys.exit(1)
        unplayed = filter_unplayed(tracks, played)
        used_catalog_api = True
    else:
        print("No CATALOG_API_URL set — skipping played-track filter, using full collection.")
        unplayed = list(tracks)

    # 3. Always print the availability table (deterministic, no LLM cost).
    counts, outlier_count, outlier_genres = _print_availability(
        tracks, unplayed, show_unplayed=used_catalog_api, excluded_count=denylist_excluded
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
        from mixlab.llm import _MAX_STAGE1_POOL_CUSTOM

        pool = build_custom_genre_pool(genre, unplayed, CUSTOM_GENRES, GENRE_MAP)
        if not pool:
            print(f"No unplayed tracks found for custom genre '{genre}'.", file=sys.stderr)
            sys.exit(1)
        print(f"Custom genre '{genre}': {len(pool)} tracks in pool.")
        # Sort by BPM for Stage 1 window selection — ensures each window is BPM-coherent.
        # (Camelot walk not used here: it can span large BPM gaps across sub-genres.)
        bpm_sorted_pool = sorted(pool, key=lambda t: t.bpm)
        stage1_pool = select_stage1_window(bpm_sorted_pool, _MAX_STAGE1_POOL_CUSTOM)
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
            print(f"No unplayed tracks found for genre '{genre}'.", file=sys.stderr)
            sys.exit(1)

        # 6a. LLM Stage 1 — standard path.
        for genre_label, cluster_tracks in clusters.items():
            bpm_filtered = filter_by_bpm(cluster_tracks)
            bpm_filtered_counts[genre_label] = len(bpm_filtered)
            if len(bpm_filtered) < _MIN_SHORTLIST_TRACKS:
                print(
                    f"Stage 1: skipping {genre_label} — {len(bpm_filtered)} tracks after BPM filter "
                    f"(minimum {_MIN_SHORTLIST_TRACKS})"
                )
                continue
            sorted_tracks = sort_by_camelot(bpm_filtered)
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

    # Select shortlists for Stage 2 — random sample from top candidates (ensures variety across runs).
    all_shortlists = select_shortlists_for_stage2(all_shortlists)
    if not all_shortlists:
        print("No shortlists survived track resolution — collection may be out of sync.", file=sys.stderr)
        sys.exit(1)
    _print_pipeline_summary(
        collection_count=len(tracks),
        unplayed_count=len(unplayed),
        used_catalog_api=used_catalog_api,
        genre_cluster_counts=genre_cluster_counts,
        bpm_filtered_counts=bpm_filtered_counts,
        same_genre_outlier_count=same_genre_outlier_count,
        stage1_shortlist_count=stage1_shortlist_count,
        stage2_shortlist_count=len(all_shortlists),
    )

    # 7. LLM Stage 2 — creative curation + full report (single Anthropic call).
    print(f"Stage 2: curating {len(all_shortlists)} shortlist(s)...")
    all_concepts, report = await stage2_curate_and_report(
        all_shortlists,
        tracks_by_id,
        stage2_provider,
        custom_genre_label=genre if is_custom else None,
        custom_genre_sub_genres=custom_genre_sub_genres,
    )
    if not all_concepts:
        print(report, file=sys.stderr)
        sys.exit(1)
    elapsed = time.monotonic() - t_start
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    report_context = _format_report_context(
        genre=genre,
        playlist_name=None,
        all_tracks=all_tracks,
        stage2_provider=stage2_provider,
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
    )


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
  mixlab --genre 4x4 --all-tracks     cross-genre 4x4 set from full collection
  mixlab --genres                     show cached counts from last run (no API)
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
    parser.add_argument("--duration", type=int, default=None, help="Target set duration in minutes (reserved)")
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
        "--stage2-provider",
        type=str,
        default=None,
        metavar="PROVIDER",
        help="Stage 2 LLM provider: anthropic (default) or minimax",
    )
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Use the full collection, ignoring played-track history (overrides CATALOG_API_URL)",
    )
    args = parser.parse_args()
    if args.genres:
        _show_cached_genres()
        return

    export_dir: Path | None = None
    if args.export is not None:
        export_dir = Path(args.export)
    elif args.export_playlists:
        export_dir = Path("output/playlists")

    if args.playlist:
        asyncio.run(run_playlist_mode(args.playlist, args.genre, export_dir, args.stage2_provider, args.all_tracks))
        return

    asyncio.run(run(args.genre, args.duration, export_dir, args.stage2_provider, args.all_tracks))


if __name__ == "__main__":
    main()
