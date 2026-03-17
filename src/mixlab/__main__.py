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
    count_available_by_genre,
    count_outlier_genres,
    filter_by_bpm,
    partition_outliers,
    resolve_genre_clusters,
    sort_by_camelot,
)
from mixlab.config import GENRE_MAP, IGNORED_GENRES
from mixlab.discord_client import send_report
from mixlab.llm import stage1_concepts, stage2_curate_and_report
from mixlab.matcher import filter_unplayed
from mixlab.models import MixConcept, Track
from mixlab.playlist_exporter import export_merged_xml, generate_merged_xml_bytes, parse_raw_tracks
from mixlab.reader import apply_bpm_corrections, parse_collection

_XML_PATH = Path("import/rekordbox.xml")


def _build_tracks_by_id(tracks: list[Track]) -> dict[str, Track]:
    return {t.track_id: t for t in tracks}


def _print_availability(
    all_tracks: list[Track], unplayed: list[Track]
) -> tuple[dict[str, tuple[int, int]], int, dict[str, tuple[int, int]]]:
    counts = count_available_by_genre(all_tracks, unplayed, GENRE_MAP)
    outlier_genres = count_outlier_genres(all_tracks, unplayed, GENRE_MAP, ignored=IGNORED_GENRES)
    outlier_count = sum(u for _, u in outlier_genres.values())

    print("\nAvailable tracks (unplayed / in collection):")
    for label, (total, available) in counts.items():
        bar = "█" * min(available // 10, 30)
        print(f"  {label:<20} {available:>4} / {total:<4}  {bar}")
    if outlier_genres:
        print("\n  Unmapped Rekordbox genre tags (not in GENRE_MAP):")
        for tag, (total, available) in outlier_genres.items():
            print(f"    {tag:<26} {available:>4} / {total:<4}")
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


async def run(
    genre: str | None, duration: int | None, export_dir: Path | None, stage2_provider: str | None = None
) -> None:  # noqa: ARG001 — duration reserved
    # 1. Parse collection.
    tracks = parse_collection(_XML_PATH)
    tracks = apply_bpm_corrections(tracks)

    # 2. Fetch played tracks and filter.
    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    if not catalog_url:
        print("ERROR: CATALOG_API_URL is not set — cannot filter played tracks. Aborting.", file=sys.stderr)
        sys.exit(1)
    try:
        played = await fetch_played_tracks(api_key, catalog_url)
    except Exception as exc:
        print(f"ERROR: Could not fetch played tracks — aborting: {exc}", file=sys.stderr)
        sys.exit(1)
    unplayed = filter_unplayed(tracks, played)

    # 3. Always print the availability table (deterministic, no LLM cost).
    counts, outlier_count, outlier_genres = _print_availability(tracks, unplayed)
    save_genre_cache(counts, outlier_count, outlier_genres)

    # 4. If no genre specified, stop here — table is the output.
    if not genre:
        print("Specify --genre <label> to generate mix concepts.")
        print("Labels: " + ", ".join(sorted(GENRE_MAP.keys())))
        return

    # 5. Cluster and scope to the requested genre.
    clusters, outliers = partition_outliers(unplayed, GENRE_MAP)
    clusters = resolve_genre_clusters(genre, clusters, GENRE_MAP)

    if not clusters:
        print(f"No unplayed tracks found for genre '{genre}'.", file=sys.stderr)
        sys.exit(1)

    # 6. LLM Stage 1 — build candidate shortlists via cascade (MiniMax / Groq / etc.).
    t_start = time.monotonic()
    all_shortlists: list[MixConcept] = []
    for genre_label, cluster_tracks in clusters.items():
        sorted_tracks = sort_by_camelot(filter_by_bpm(cluster_tracks))
        all_shortlists.extend(await stage1_concepts(sorted_tracks, genre_label))

    # Outliers ≥ 4 within this genre scope — shortlist as Misc.
    genre_outliers = [t for t in outliers if t.genre.lower() == genre.lower()]
    if len(genre_outliers) >= 4:
        all_shortlists.extend(await stage1_concepts(genre_outliers, "Misc"))

    if not all_shortlists:
        print("No shortlists generated — all tracks may have been excluded.", file=sys.stderr)
        sys.exit(1)
    if len(all_shortlists) < 3:
        print(
            f"⚠️  Stage 1 produced only {len(all_shortlists)} shortlist(s) — pool may be too thin for 3–6 concepts.",
            file=sys.stderr,
        )

    # Cap shortlists sent to Stage 2 — keep the richest pools.
    tracks_by_id = _build_tracks_by_id(tracks)
    max_shortlists = 6
    all_shortlists = [s for s in all_shortlists if any(tid in tracks_by_id for tid in s.track_ids)]
    all_shortlists = sorted(all_shortlists, key=lambda s: len(s.track_ids), reverse=True)[:max_shortlists]
    if not all_shortlists:
        print("No shortlists survived track resolution — collection may be out of sync.", file=sys.stderr)
        sys.exit(1)

    # 7. LLM Stage 2 — creative curation + full report (single Anthropic call).
    all_concepts, report = await stage2_curate_and_report(all_shortlists, tracks_by_id, stage2_provider)
    if not all_concepts:
        print(report, file=sys.stderr)
        sys.exit(1)
    elapsed = time.monotonic() - t_start
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    report += f"\n\n---\n\n⏱ Generated in {elapsed_str}"

    print(report)

    # 8. Generate merged Rekordbox XML (one file — for Discord attachment and optional disk export).
    raw_tracks_xml = parse_raw_tracks(_XML_PATH)
    today = datetime.date.today().isoformat()
    folder_name = f"Mix Lab - {genre} - {today}"
    merged_bytes = generate_merged_xml_bytes(all_concepts, raw_tracks_xml, folder_name)
    xml_attachments: list[tuple[str, bytes]] = (
        [("rekordbox_export.xml", merged_bytes)] if merged_bytes is not None else []
    )

    if export_dir is not None:
        out_path = export_merged_xml(all_concepts, raw_tracks_xml, export_dir / "rekordbox_export.xml", folder_name)
        if out_path is not None:
            print(f"Exported: {out_path}")

    # 9. Discord delivery.
    filtered_outliers = [t for t in outliers if t.genre not in IGNORED_GENRES]
    await send_report(report, all_concepts, filtered_outliers, tracks_by_id, counts=counts, attachments=xml_attachments)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="MixLab — AI-powered DJ crate assistant")
    parser.add_argument("--genre", type=str, default=None, help="Genre to target (e.g. drum_and_bass, house)")
    parser.add_argument("--duration", type=int, default=None, help="Target set duration in minutes")
    parser.add_argument("--genres", action="store_true", help="Show available genres from last run (no API calls)")
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
        help="Stage 2 LLM provider: anthropic (default) or minimax",
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

    asyncio.run(run(args.genre, args.duration, export_dir, args.stage2_provider))


if __name__ == "__main__":
    main()
