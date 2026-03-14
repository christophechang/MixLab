from __future__ import annotations

import argparse
import asyncio
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
from mixlab.llm import stage1_concepts, stage2_report
from mixlab.matcher import filter_unplayed
from mixlab.models import MixConcept, PlayedTrack, Track
from mixlab.reader import apply_bpm_corrections, parse_collection

_XML_PATH = Path("import/rekordbox.xml")
_CHANGSTA_BASE = "https://api.changsta.com"


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


async def run(genre: str | None, duration: int | None) -> None:  # noqa: ARG001 — duration reserved
    # 1. Parse collection.
    tracks = parse_collection(_XML_PATH)
    tracks = apply_bpm_corrections(tracks)

    # 2. Fetch played tracks and filter.
    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    played: list[PlayedTrack] = []
    try:
        played = await fetch_played_tracks(api_key, _CHANGSTA_BASE)
    except Exception as exc:
        print(f"WARNING: Could not fetch played tracks — proceeding without exclusion: {exc}", file=sys.stderr)
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

    # 6. LLM Stage 1 — generate concepts.
    t_start = time.monotonic()
    all_concepts: list[MixConcept] = []
    for genre_label, cluster_tracks in clusters.items():
        sorted_tracks = sort_by_camelot(filter_by_bpm(cluster_tracks))
        concepts = await stage1_concepts(sorted_tracks, genre_label)
        all_concepts.extend(concepts)

    # Outliers ≥ 4 within this genre scope — pass as Misc.
    genre_outliers = [t for t in outliers if t.genre.lower() == genre.lower()]
    if len(genre_outliers) >= 4:
        misc_concepts = await stage1_concepts(genre_outliers, "Misc")
        all_concepts.extend(misc_concepts)

    if not all_concepts:
        print("No concepts generated — all tracks may have been excluded.", file=sys.stderr)
        sys.exit(1)

    # Cap concepts sent to Stage 2 — keep the richest (most tracks) to avoid truncation.
    tracks_by_id = _build_tracks_by_id(tracks)
    max_stage2_concepts = 6
    all_concepts = [c for c in all_concepts if any(tid in tracks_by_id for tid in c.track_ids)]
    all_concepts = sorted(all_concepts, key=lambda c: len(c.track_ids), reverse=True)[:max_stage2_concepts]

    # 7. LLM Stage 2 — full report.
    report = await stage2_report(all_concepts, tracks_by_id)
    elapsed = time.monotonic() - t_start
    report += f"\n\n---\n\n⏱ Generated in {elapsed:.0f}s"

    print(report)

    # 8. Discord delivery.
    filtered_outliers = [t for t in outliers if t.genre not in IGNORED_GENRES]
    await send_report(report, all_concepts, filtered_outliers, tracks_by_id, counts=counts)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="MixLab — AI-powered DJ crate assistant")
    parser.add_argument("--genre", type=str, default=None, help="Genre to target (e.g. drum_and_bass, house)")
    parser.add_argument("--duration", type=int, default=None, help="Target set duration in minutes")
    parser.add_argument("--genres", action="store_true", help="Show available genres from last run (no API calls)")
    args = parser.parse_args()
    if args.genres:
        _show_cached_genres()
        return
    asyncio.run(run(args.genre, args.duration))


if __name__ == "__main__":
    main()
