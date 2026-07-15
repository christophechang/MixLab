from __future__ import annotations

import argparse
import asyncio
import datetime
import difflib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from mixlab.cache import load_genre_cache, save_genre_cache
from mixlab.client import fetch_mix_names, fetch_played_tracks
from mixlab.clustering import (
    ABSOLUTE_MIN,
    MAX_SHORTLIST,
    build_custom_genre_pool,
    build_mix_canvas,
    count_available_by_genre,
    count_outlier_genres,
    partition_bpm_pools,
    partition_outliers,
    partition_pool,
    partition_pool_with_stats,
    resolve_genre_clusters,
    select_canvases,
    sort_by_camelot,
)
from mixlab.config import CUSTOM_GENRES, GENRE_MAP, IGNORED_GENRES
from mixlab.directions import generate_directions
from mixlab.discord_client import send_report
from mixlab.history import (
    ConceptHistory,
    HistoryEntry,
    append_run,
    apply_feedback_verdict,
    load_history,
    recent_concept_titles,
    save_history,
)
from mixlab.html_report import (
    _split_prose,  # noqa: PLC2701 — reused to build the summary.json run-notes trailer (#89)
    render_html_report,
    report_filename,
)
from mixlab.llm import (
    _match_canvas_for_concept,  # noqa: PLC2701 — reused here to stamp direction_type on cards (#84)
    _parse_user_intent,  # noqa: PLC2701 — reused here for playlist-mode risk override (#54)
    _qualifies_for_revision,  # noqa: PLC2701 — reused for the revision-gate decision (#55)
    make_cascade_state,
    resequence_suggestions,
    revise_concepts,
    stage0_intent_brief,
    stage2_curate_and_report,
    validate_stage2_output,
)
from mixlab.matcher import filter_played, filter_unplayed
from mixlab.models import MixCanvas, MixConcept, RiskTolerance, Track, TrackMode
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
from mixlab.prep import PrepItem, rank_prep_targets
from mixlab.reader import apply_bpm_corrections, parse_collection, parse_playlists
from mixlab.remote import MixLabRemote, RemoteConfig, RemoteConfigError
from mixlab.summary import build_run_summary
from mixlab.worker import WorkerConfig, run_worker


def _resolve_collection_path() -> Path:
    """Input collection path. Defaults to the usual local export; the MixLab Anywhere worker
    sets ``MIXLAB_COLLECTION_PATH`` so a remote run reads its own downloaded copy instead of
    the file a local run/automation uses. Normal CLI usage is unchanged (env var unset).
    """
    return Path(os.environ.get("MIXLAB_COLLECTION_PATH", "import/rekordbox.xml"))


_XML_PATH = _resolve_collection_path()
_HISTORY_PATH = Path(".mixlab/concept-history.json")
_DO_NOT_RECOMMEND_PLAYLIST = "DO NOT RECOMMEND"
_FEEDBACK_VERDICTS = ("played", "played_modified", "rejected", "unused")


def _build_tracks_by_id(tracks: list[Track]) -> dict[str, Track]:
    return {t.track_id: t for t in tracks}


def _write_html_report(html: str, genre: str | None, playlist_name: str | None) -> Path:
    """Write the standalone HTML report to disk and return its path (#45).

    Directory defaults to ``output/reports`` and can be overridden with
    ``MIXLAB_REPORT_DIR``. The filename embeds the genre/playlist slug and today's date.
    """
    report_dir = Path(os.environ.get("MIXLAB_REPORT_DIR", "output/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = report_dir / report_filename(genre, playlist_name, today)
    path.write_text(html, encoding="utf-8")
    print(f"HTML report: {path}")
    return path


def _title_case_label(label: str) -> str:
    return label.replace("_", " ").title()


def _annotate_direction_types(concepts: list[MixConcept], canvases: list[MixCanvas]) -> None:
    """Stamp each concept with its matched canvas's direction_type (#84)."""
    for concept in concepts:
        canvas = _match_canvas_for_concept(concept, canvases)
        if canvas is not None and canvas.direction_type:
            concept.direction_type = canvas.direction_type


def _format_report_context(
    *,
    genre: str | None,
    playlist_name: str | None,
    mode: TrackMode,
    export_dir: Path | None,
    intent: str | None = None,
    mix_length: int | None = None,
    locked: bool = False,
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
    if mix_length is not None:
        details.append(f"{mix_length}min set")
    if locked:
        details.append("locked pool")
    if export_dir is not None:
        details.append("export enabled")

    context = f"Report context: {base_label} ({', '.join(details)})"
    if intent:
        context += f'\nIntent: "{intent.strip()}"'
    return context


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


def _report_stage1_window(overflow_counts: list[int]) -> int:
    """Print a per-shortlist stderr note for each windowed (capped) Stage 1 shortlist.

    Returns the total number of overflow tracks (beyond the windows) across all
    shortlists so the caller can surface it in the pipeline summary (#48).
    """
    total = 0
    for i, overflow in enumerate(overflow_counts):
        if overflow > 0:
            print(
                f"Stage 1 window: shortlist {i} capped at {MAX_SHORTLIST} tracks "
                f"({overflow} overflow, rotated by seed)",
                file=sys.stderr,
            )
            total += overflow
    return total


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
    stage1_overflow: int = 0,
) -> None:
    print("Pipeline summary:")
    filter_label = pool_label if used_catalog_api else "eligible"
    print(f"- Track pool: {unplayed_count} {filter_label} / {collection_count} in scoped collection")
    print(f"- Genre scope before BPM filter: {_format_pipeline_counts(genre_cluster_counts)}")
    print(f"- Genre scope after BPM filter: {_format_pipeline_counts(bpm_filtered_counts)}")
    print(f"- Same-genre outliers considered: {same_genre_outlier_count}")
    print(f"- Stage 1 shortlists: {stage1_shortlist_count}")
    if stage1_overflow:
        print(f"- Stage 1 overflow (tracks beyond windows): {stage1_overflow}")
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


def _print_feedback_listing(history: ConceptHistory) -> None:
    """List the most recent run's concepts with their current feedback state (#52)."""
    if not history.runs:
        print("No concept history found — run mixlab first to generate concepts.")
        return
    latest = history.runs[-1]
    if not latest.concepts:
        print(f"Most recent run ({latest.created_at[:10]}, {latest.genre}) recorded no concepts.")
        return
    print(f"Most recent run ({latest.created_at[:10]}, {latest.genre}):")
    for i, record in enumerate(latest.concepts):
        feedback_label = record.feedback or "(none)"
        print(f"  [{i}] {record.title} — feedback: {feedback_label}")


def run_feedback(concept: str | None, verdict: str | None, notes: str, history_path: Path) -> int:
    """Handle ``mixlab --feedback`` (#52). No LLM calls, no network — pure history-file edit.

    Thin CLI wrapper around :func:`mixlab.history.apply_feedback_verdict` (#92), which
    also backs the worker's history sync so both paths use one match-and-mutate
    implementation.

    Returns the process exit code: 0 for a successful listing or feedback update,
    1 for a usage error or an unmatched concept title.
    """
    history = load_history(history_path)

    if concept is None and verdict is None:
        _print_feedback_listing(history)
        return 0
    if verdict is None:
        print("ERROR: --verdict is required when --concept is given.", file=sys.stderr)
        return 1
    if concept is None:
        print("ERROR: --concept is required when --verdict is given.", file=sys.stderr)
        return 1

    record = apply_feedback_verdict(history, concept, verdict, notes)
    if record is None:
        all_titles = [c.title for entry in history.runs for c in entry.concepts]
        suggestions = difflib.get_close_matches(concept, all_titles, n=3)
        print(f'No concept found matching "{concept}".', file=sys.stderr)
        if suggestions:
            print("Did you mean: " + ", ".join(suggestions), file=sys.stderr)
        return 1

    entry = next(e for e in history.runs if any(c is record for c in e.concepts))
    save_history(history, history_path)
    print(f'Recorded feedback "{verdict}" for concept "{record.title}" (run {entry.created_at[:10]}, {entry.genre}).')
    return 0


_PREP_LABEL_WIDTH = 42
_PREP_BUCKET_WIDTH = 14
_PREP_HINT = "Cue up the top entries in Rekordbox, re-export, and booth sheets gain clock times."


def _truncate_label(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` chars, replacing the tail with an ellipsis when it overflows."""
    if len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def _print_prep_table(items: list[PrepItem], totals: dict[str, tuple[int, int]]) -> None:
    print(f"\nCue-Prep Assistant — top {len(items)} target(s):\n")
    print(
        f"  {'#':>3}  {'Track':<{_PREP_LABEL_WIDTH}} {'Bucket':<{_PREP_BUCKET_WIDTH}} "
        f"{'BPM':>6}  {'Key':<4} {'Gap':<11} {'Score':>6}  Reason"
    )
    for i, item in enumerate(items, start=1):
        label = _truncate_label(item.label, _PREP_LABEL_WIDTH)
        bucket_label = _truncate_label(item.bucket or item.genre, _PREP_BUCKET_WIDTH)
        print(
            f"  {i:>3}  {label:<{_PREP_LABEL_WIDTH}} {bucket_label:<{_PREP_BUCKET_WIDTH}} "
            f"{item.bpm:>6.1f}  {item.camelot_key:<4} {item.gap:<11} {item.score:>6.2f}  {item.reason}"
        )

    print()
    for bucket_key, (gap_count, total_count) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
        print(f"  {bucket_key}: {gap_count} of {total_count} tracks lack cue data")

    print(f"\n{_PREP_HINT}")


def run_prep(genre: str | None, top: int) -> int:
    """Handle ``mixlab --prep`` (Cue-Prep Assistant, #72/#74). No LLM calls, no API,
    no Discord — ranks tracks with missing/partial cue-point data by expected payoff.

    Returns the process exit code: 0 on a successful listing (including the
    "nothing to prep" case), 1 when ``--genre`` names something other than a
    standard genre label or a standard label with no matching tracks.
    """
    bucket: str | None = None
    if genre is not None:
        if genre not in GENRE_MAP:
            print(
                f"ERROR: --prep accepts standard genre labels only — '{genre}' is not one. "
                f"Standard labels: {', '.join(sorted(GENRE_MAP.keys()))}",
                file=sys.stderr,
            )
            return 1
        bucket = genre

    tracks = parse_collection(_XML_PATH)
    tracks = apply_bpm_corrections(tracks)
    # Denylisted tracks are never recommended, so cueing them is wasted prep time —
    # exclude them exactly as the concept pipeline does.
    tracks, _ = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    history = load_history(_HISTORY_PATH)

    items, totals = rank_prep_targets(tracks, history, bucket=bucket, top=top)

    if bucket is not None and not totals:
        print(f"No tracks found for genre '{genre}'.", file=sys.stderr)
        return 1

    total_gap = sum(gap_count for gap_count, _ in totals.values())
    if total_gap == 0:
        print("Every track in scope already has full cue-point data — nothing to prep. Nice work!")
        return 0

    _print_prep_table(items, totals)
    return 0


async def run_playlist_mode(
    playlist_name: str,
    genre: str | None,
    export_dir: Path | None,
    mode: TrackMode = "unplayed",
    min_bpm: float | None = None,
    max_bpm: float | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    intent: str | None = None,
    debug: bool = False,
    mix_length: int | None = None,
    locked: bool = False,
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

    if intent:
        parsed_risk = _parse_user_intent(intent).get("risk")
        if parsed_risk is not None:
            print(
                f"--intent risk signal '{parsed_risk}' overrides Stage 0 risk_tolerance "
                f"'{intent_brief.risk_tolerance}'.",
                file=sys.stderr,
            )
            intent_brief.risk_tolerance = cast("RiskTolerance", parsed_risk)

    library_source = tracks
    if genre is not None:
        try:
            library_source = filter_tracks_for_playlist_genre(tracks, genre, GENRE_MAP, CUSTOM_GENRES)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Playlist mode genre filter '{genre}': {len(library_source)} tracks in scope.")

    if locked:
        print(f"--locked: library additions disabled — Stage 2 selects from {len(seed_tracks)} seed tracks only.")
        library_tracks: list[Track] = []
    else:
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

    # Load concept history so Stage 2 can deliberately diverge from prior runs (#13).
    # Playlist mode does not use canvas selection so history was previously not loaded here.
    history = load_history(_HISTORY_PATH)

    all_concepts, report = await stage2_curate_and_report(
        shortlists,
        tracks_by_id,
        playlist_name=playlist_name,
        seed_ids=seed_ids,
        seed_track_ids=[track_id for track_id in raw_seed_ids if track_id in tracks_by_id],
        unplayed_ids=unplayed_ids,
        intent_brief=intent_brief,
        used_mix_names=mix_names or None,
        concept_history=history,
        genre_intent=intent,
        debug=debug,
        mix_length=mix_length,
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
        intent=intent,
        mix_length=mix_length,
        locked=locked,
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

    # Standalone HTML report (#45) — playlist mode has no validation pass and no crate counts.
    html = render_html_report(
        all_concepts,
        tracks_by_id,
        report_text=report,
        report_context=report_context,
        validation_warnings=[],
        counts=None,
        generated_at=today,
    )
    html_path = _write_html_report(html, genre, playlist_name)
    attachments: list[tuple[str, bytes]] = [*xml_attachments, (html_path.name, html.encode("utf-8"))]

    # Structured run summary (#89) — playlist mode produces MixConcept objects via the same
    # stage2_curate_and_report pipeline genre mode uses, so build_run_summary applies unchanged.
    # `flags` intentionally differs in shape from genre mode's (playlist/genre/mode/BPM-year
    # range/locked vs. risk/directions/resequence/deep/stage1Seed) — the API/SPA contract already
    # types summary.flags as an opaque dict because it varies by mode.
    # `seed` has no playlist-mode counterpart (no Stage-1 pool partitioning happens here); reuse
    # genre mode's date-based fallback so the field stays populated without implying reproducibility.
    playlist_flags: dict[str, object] = {
        "playlist": playlist_name,
        "genre": genre,
        "mode": mode,
        "minBpm": min_bpm,
        "maxBpm": max_bpm,
        "minYear": min_year,
        "maxYear": max_year,
        "intent": intent,
        "mixLength": mix_length,
        "locked": locked,
    }
    _, run_notes_sections = _split_prose(report, len(all_concepts))
    run_notes = "\n\n".join(section for section in run_notes_sections if section.strip())
    summary = build_run_summary(
        all_concepts,
        tracks_by_id,
        flags=playlist_flags,
        seed=int(datetime.date.today().strftime("%Y%m%d")),
        validation_warnings=[],
        run_notes=run_notes,
        generated_at=today,
    )
    summary_path = html_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run summary: {summary_path}")

    await send_report(
        report,
        [all_concepts[0]],
        [],
        tracks_by_id,
        counts={},
        attachments=attachments,
        show_unplayed=False,
        report_context=report_context,
        filter_desc=_build_filter_desc(min_bpm=min_bpm, max_bpm=max_bpm, min_year=min_year, max_year=max_year),
    )


async def run_export_unplayed() -> None:
    api_key = os.environ.get("CHANGSTA_API_KEY", "")
    catalog_url = os.environ.get("CATALOG_API_URL", "")
    if not catalog_url:
        print("ERROR: CATALOG_API_URL not set — cannot determine unplayed tracks.", file=sys.stderr)
        sys.exit(1)

    tracks = parse_collection(_XML_PATH)
    tracks, denylist_excluded = _apply_do_not_recommend_filter(tracks, _XML_PATH)
    tracks = apply_bpm_corrections(tracks)

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
    intent: str | None = None,
    deep: bool = False,
    debug: bool = False,
    stage1_seed: int | None = None,
    mix_length: int | None = None,
    directions: str = "mixed",
    no_revise: bool = False,
    risk: RiskTolerance = "medium",
    resequence: bool = False,
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
    unplayed_ids_for_stage2: set[str] | None = None
    pool_label = "unplayed"
    if mode == "all":
        unplayed = list(tracks)
        if catalog_url:
            try:
                played, mix_names = await asyncio.gather(
                    fetch_played_tracks(api_key, catalog_url),
                    fetch_mix_names(api_key, catalog_url),
                )
            except Exception as exc:
                print(
                    f"--mode all: catalog fetch failed ({type(exc).__name__}: {exc}) — "
                    f"Stage 2 played/unplayed signal disabled.",
                    file=sys.stderr,
                )
            else:
                if mix_names:
                    print(
                        f"Fetched {len(mix_names)} catalogue mix name(s) — injecting into Stage 2 prompt.",
                        flush=True,
                    )
                played_ids_set = {t.track_id for t in filter_played(tracks, played)}
                unplayed_ids_for_stage2 = {t.track_id for t in tracks if t.track_id not in played_ids_set}
                used_catalog_api = True
                print(
                    f"--mode all — fetched played status: {len(played_ids_set)}/{len(tracks)} played, "
                    f"{len(unplayed_ids_for_stage2)} unplayed. Stage 2 will favour unplayed in ties.",
                    flush=True,
                )
        else:
            print(
                "--mode all — CATALOG_API_URL not set, Stage 2 played/unplayed signal disabled.",
                file=sys.stderr,
            )
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
    all_shortlists: list[MixConcept] = []
    custom_genre_sub_genres: list[str] | None = None
    genre_unplayed_track_ids_source: list[Track] = []
    genre_cluster_counts: dict[str, int] = {}
    bpm_filtered_counts: dict[str, int] = {}
    same_genre_outlier_count = 0
    total_stage1_overflow = 0

    # Resolve the effective Stage 1 seed once so every partition_pool call this run
    # shares it (and the user can reproduce the exact partition). Defaults to today's
    # date as YYYYMMDD when --stage1-seed is not supplied (#48).
    effective_seed = stage1_seed if stage1_seed is not None else int(datetime.date.today().strftime("%Y%m%d"))
    print(f"Stage 1 seed: {effective_seed} (reproduce with --stage1-seed {effective_seed})")

    if is_custom:
        pool = build_custom_genre_pool(genre, unplayed, CUSTOM_GENRES, GENRE_MAP)
        if not pool:
            print(f"No tracks found for custom genre '{genre}'.", file=sys.stderr)
            sys.exit(1)
        print(f"Custom genre '{genre}': {len(pool)} tracks in pool.")
        # Sort by BPM for Stage 1 — ensures BPM-coherent ordering.
        # (Camelot walk not used here: it can span large BPM gaps across sub-genres.)
        bpm_sorted_pool = sorted(pool, key=lambda t: t.bpm)
        bpm_sorted_pool = [t for t in bpm_sorted_pool if t.bpm > 0]  # precondition filter
        cfg = CUSTOM_GENRES[genre]
        custom_genre_sub_genres = cfg["genres"]
        shortlists, stage1_stats = partition_pool_with_stats(bpm_sorted_pool, seed=effective_seed)
        total_stage1_overflow += _report_stage1_window(stage1_stats.overflow_counts)
        stage1_pool_size = len(bpm_sorted_pool)
        if not shortlists:
            print(f"Stage 1: pool too small to partition — skipping {genre} (custom).", file=sys.stderr)
        all_shortlists.extend(shortlists)
        genre_unplayed_track_ids_source = pool
        # No outlier handling for custom genres — pool is already the full scope.
        genre_outliers: list[Track] = []
        outliers: list[Track] = []
        genre_cluster_counts = {genre: len(pool)}
        bpm_filtered_counts = {genre: stage1_pool_size}
    else:
        clusters, outliers = partition_outliers(unplayed, GENRE_MAP)
        clusters = resolve_genre_clusters(genre, clusters, GENRE_MAP)
        genre_cluster_counts = {genre_label: len(cluster_tracks) for genre_label, cluster_tracks in clusters.items()}

        if not clusters:
            print(f"No tracks found for genre '{genre}'.", file=sys.stderr)
            sys.exit(1)

        # 6a. Stage 1 — deterministic partitioning.
        for genre_label, cluster_tracks in clusters.items():
            pools = partition_bpm_pools(cluster_tracks)
            sorted_tracks = sort_by_camelot(pools.core)
            sorted_tracks = [t for t in sorted_tracks if t.bpm > 0]  # precondition filter
            bpm_filtered_counts[genre_label] = len(sorted_tracks)
            shortlists, stage1_stats = partition_pool_with_stats(sorted_tracks, seed=effective_seed)
            total_stage1_overflow += _report_stage1_window(stage1_stats.overflow_counts)
            if not shortlists:
                print(f"Stage 1: pool too small to partition — skipping {genre_label}.", file=sys.stderr)
                continue
            all_shortlists.extend(shortlists)

        # Outliers within this genre scope — shortlist as Misc.
        genre_outliers = [t for t in outliers if t.genre.lower() == genre.lower()]
        same_genre_outlier_count = len(genre_outliers)
        # Deterministic path: filter bpm<=0, use ABSOLUTE_MIN threshold.
        # Use a local variable — do NOT reassign genre_outliers (used later for XML export).
        bpm_filtered_outliers = [t for t in genre_outliers if t.bpm > 0]
        if len(bpm_filtered_outliers) >= ABSOLUTE_MIN:
            shortlists = partition_pool(bpm_filtered_outliers, seed=effective_seed)
            all_shortlists.extend(shortlists)

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
            stage1_overflow=total_stage1_overflow,
        )
        print("No shortlists generated — all tracks may have been excluded.", file=sys.stderr)
        sys.exit(1)
    if len(all_shortlists) < 3:
        print(
            f"⚠️  Stage 1 produced only {len(all_shortlists)} shortlist(s) — pool may be too thin for 3–6 concepts.",
            file=sys.stderr,
        )

    # Build Mix Canvases and select top candidates for Stage 2 (diversity-aware, deterministic).
    history = load_history(_HISTORY_PATH)
    all_canvases: list[MixCanvas] = [build_mix_canvas(c, tracks_by_id) for c in all_shortlists]

    # Name-avoid list (#75): catalogue mix names plus recent history concept titles,
    # so freshly generated names cannot repeat what earlier runs already produced.
    catalog_name_keys = {n.casefold() for n in mix_names}
    avoid_names = mix_names + [t for t in recent_concept_titles(history) if t.casefold() not in catalog_name_keys]

    # Direction source pool (#53): the full genre-scoped pool before BPM-pool partitioning.
    # For custom genres this is the merged cross-genre pool (genre_unplayed_track_ids_source
    # == pool, with no outliers); for standard genres it is the flattened genre scope plus
    # same-genre outliers. Directions are cross-strata by construction, so they draw from
    # this whole pool rather than any single BPM stratum.
    direction_pool = genre_unplayed_track_ids_source + genre_outliers

    if directions == "off":
        selected_canvases = select_canvases(all_canvases, history, mode=mode, risk=risk, debug=debug)
    elif directions == "only":
        selected_canvases = generate_directions(direction_pool, tracks_by_id, seed=effective_seed, max_directions=6)
        if not selected_canvases:
            print(
                "--directions only: no feasible directions for this pool — falling back to classic canvas selection.",
                file=sys.stderr,
            )
            selected_canvases = select_canvases(all_canvases, history, mode=mode, risk=risk, debug=debug)
    else:  # "mixed"
        direction_canvases = generate_directions(direction_pool, tracks_by_id, seed=effective_seed, max_directions=3)
        classic_n = max(3, 6 - len(direction_canvases))
        classic_selected = select_canvases(all_canvases, history, n=classic_n, mode=mode, risk=risk, debug=debug)
        selected_canvases = classic_selected + direction_canvases
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
        stage1_overflow=total_stage1_overflow,
    )

    # 7. LLM Stage 2 — creative curation + full report (single Anthropic call).
    print(f"Stage 2: curating {len(selected_canvases)} canvas(es)...")
    all_concepts, report = await stage2_curate_and_report(
        [c.source_concept for c in selected_canvases],
        tracks_by_id,
        custom_genre_label=genre if is_custom else None,
        custom_genre_sub_genres=custom_genre_sub_genres,
        used_mix_names=avoid_names or None,
        canvases=selected_canvases,
        unplayed_ids=unplayed_ids_for_stage2,
        concept_history=history,
        genre_intent=intent,
        mode=mode,
        risk=risk,
        deep=deep,
        debug=debug,
        mix_length=mix_length,
        # Lens rotation advances with every recorded run, so same-day runs get
        # different naming lenses instead of identical pressure (live finding:
        # the date-only seed gave every run of the day the same three lenses).
        # Still reproducible: derived from --stage1-seed/date plus history length.
        naming_seed=effective_seed + len(history.runs),
    )
    if not all_concepts:
        print(report, file=sys.stderr)
        sys.exit(1)

    # conceptId unification (#89 / M1): mint each concept's stable identity here, once,
    # immediately post-Stage-2 and before validation/revision. This id is shared verbatim
    # by the concept's history record (HistoryEntry.from_run) and its summary.json entry
    # so API feedback can join back onto history — see docs/architecture/mixlab-anywhere.md §5.2.
    for concept in all_concepts:
        concept.concept_id = str(uuid.uuid4())

    # Post-Stage-2 validation (warn-only — never aborts the run).
    validation_warnings = validate_stage2_output(
        all_concepts,
        selected_canvases,
        tracks_by_id,
        played_ids=played_track_ids,
        denylist_ids=set(),  # tracks filtered before Stage 1; can't appear in output
        allow_played=mode in ("all", "played"),
        genre=genre or "_default",
        risk=risk,
        used_mix_names=avoid_names or None,
    )

    # Bounded self-revision pass (#55). Runs when revision is enabled and there is
    # something to act on — validation findings, or a concept whose critique qualifies
    # it on its own (weak, or needs_attention with a suggested substitution).
    if not no_revise and (
        validation_warnings or any(_qualifies_for_revision(c, validation_warnings) for c in all_concepts)
    ):
        all_concepts, report, validation_warnings = await revise_concepts(
            all_concepts,
            report,
            validation_warnings,
            selected_canvases,
            tracks_by_id,
            played_ids=played_track_ids,
            allow_played=mode in ("all", "played"),
            genre=genre or "_default",
            risk=risk,
            unplayed_ids=unplayed_ids_for_stage2,
            used_mix_names=avoid_names or None,
        )

    # Deterministic sequencer pass (#61). Runs after revision so suggestions reflect the
    # final track lists. Suggest-only by default; --resequence applies the swaps.
    all_concepts, resequence_notes = resequence_suggestions(all_concepts, tracks_by_id, apply=resequence)
    if resequence_notes:
        print("\n🎚 Sequencer:")
        for note in resequence_notes:
            print(f"  {note}")
        report += "\n\n**Sequencer**\n" + "\n".join(f"- {note}" for note in resequence_notes)

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
        append_run(history, entry, _HISTORY_PATH)
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
        intent=intent,
        mix_length=mix_length,
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

    # 8b. Standalone HTML report (#45) — self-contained artifact written to disk and attached.
    _annotate_direction_types(all_concepts, selected_canvases)
    html = render_html_report(
        all_concepts,
        tracks_by_id,
        report_text=report,
        report_context=report_context,
        validation_warnings=validation_warnings,
        counts=counts,
        generated_at=today,
    )
    html_path = _write_html_report(html, genre, None)
    attachments: list[tuple[str, bytes]] = [*xml_attachments, (html_path.name, html.encode("utf-8"))]

    # 8c. Structured run summary (#89) — same filename stem as the HTML report, .json
    # extension. Genre mode only (run_playlist_mode has no counterpart yet).
    # Effective flags mirror the run.json "flags" shape from the architecture doc (§5.1).
    run_flags: dict[str, object] = {
        "genre": genre,
        "mode": mode,
        "risk": risk,
        "directions": directions,
        "intent": intent,
        "mixLength": mix_length,
        "resequence": resequence,
        "deep": deep,
        "stage1Seed": stage1_seed,
    }
    # run_notes v1 (documented interpretation): the report's trailing sections — the
    # same text html_report renders under "Run notes" — reconstructed via the same
    # _split_prose the HTML report uses so the two artifacts never disagree.
    _, run_notes_sections = _split_prose(report, len(all_concepts))
    run_notes = "\n\n".join(section for section in run_notes_sections if section.strip())
    summary = build_run_summary(
        all_concepts,
        tracks_by_id,
        flags=run_flags,
        seed=effective_seed,
        validation_warnings=validation_warnings,
        run_notes=run_notes,
        generated_at=today,
    )
    summary_path = html_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run summary: {summary_path}")

    # 9. Discord delivery.
    filtered_outliers = [t for t in outliers if t.genre not in IGNORED_GENRES]
    await send_report(
        report,
        all_concepts,
        filtered_outliers,
        tracks_by_id,
        counts=counts,
        attachments=attachments,
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


def _warn_intent(intent: str | None) -> None:
    """Emit stderr warnings when --intent is likely to produce ineffective results."""
    if intent is None:
        return
    intent_stripped = intent.strip()
    if not intent_stripped:
        print(
            "WARNING: --intent is empty; Stage 2 will run without creative direction.",
            file=sys.stderr,
        )
        return
    word_count = len(intent_stripped.split())
    if word_count < 5:
        word_label = "word" if word_count == 1 else "words"
        print(
            f"WARNING: --intent is very short ({word_count} {word_label}). "
            "Consider adding purpose, mood, or stylistic scope for better results.",
            file=sys.stderr,
        )
    elif word_count > 50:
        print(
            f"WARNING: --intent exceeds 50 words ({word_count}). "
            "Consider keeping it concise — Stage 2 responds better to focused direction.",
            file=sys.stderr,
        )


def _run_worker_cli(*, once: bool) -> None:
    """Handle ``mixlab --worker`` / ``--worker-once`` (#91 / M3).

    Builds the remote config from the environment (``MIXLAB_API_URL`` /
    ``MIXLAB_API_SECRET``) — a missing value prints the reason to stderr and exits 1 —
    then drives the worker loop. ``MIXLAB_WORKER_POLL_SECONDS`` (default 30) sets the
    empty-poll cadence.
    """
    try:
        remote_config = RemoteConfig.from_env()
    except RemoteConfigError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    remote = MixLabRemote(remote_config)
    worker_config = WorkerConfig.from_env()
    poll_seconds = float(os.environ.get("MIXLAB_WORKER_POLL_SECONDS", "30"))
    sys.exit(run_worker(remote, worker_config, poll_seconds=poll_seconds, once=once))


def main() -> None:
    # Line-buffer stdout even when piped (tee/log capture) so progress lines interleave
    # correctly with unbuffered stderr diagnostics instead of flushing in one late block.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
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
  traverse  all standard genres           no BPM filter; unlocks cross-genre
                                          journey concepts via ratio bridges

  Custom genre pools are partitioned deterministically by BPM/key/era (see docs/architecture/deterministic-stage1.md).

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
  mixlab --genre house --intent "warmup, melodic, outdoor afternoon"  creative direction
  mixlab --genre house --deep          opt-in critique pass per concept (2x Stage 2 cost)
  mixlab --genre house --no-revise     skip the bounded self-revision pass (#55)
  mixlab --genre house --resequence    apply the sequencer's suggested order swaps (#61)
  mixlab --genre house --risk high     promote wildcard/anchor picks, relax jump thresholds (#42)
  mixlab --playlist "Monday Night" --mix-length 60   target ~15 tracks for a 1-hour set
  mixlab --playlist "Monday Night" --mix-length 90   target ~22 tracks for a 90-minute set
  mixlab --playlist "Monday Night" --locked          trim seed playlist only, no library additions
  mixlab --genres                     show cached counts from last run (no API)
  mixlab --export-unplayed            export all unplayed tracks as Rekordbox XML + post to Discord
  mixlab --feedback                   list most recent run's concepts + feedback state (no LLM)
  mixlab --feedback --concept "Title" --verdict played --notes "great opener"  record concept feedback
  mixlab --prep                       rank uncued tracks by cue-prep payoff (offline, no LLM)
  mixlab --prep --genre house --top 10  scope to house genre, top 10 targets
  mixlab --worker                     run as a MixLab Anywhere worker (see README)
""",
    )
    parser.add_argument(
        "--genre",
        type=str,
        default=None,
        metavar="LABEL",
        help="Genre to target. Standard labels: house, drum_and_bass, techno, etc. "
        "Custom labels: 170, 140, 4x4, traverse (cross-genre pools). "
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
        "--intent",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Free-text creative direction for the Stage 2 model in genre mode. "
            "Shapes which concepts are generated, their mood, energy arc, and track selection. "
            "Most effective when it covers: (1) occasion/purpose — what is this mix for? "
            "(2) emotional destination — what should the listener feel by the end? "
            "(3) stylistic scope — any specific scene, era, or cross-genre angle. "
            "Example: 'Late-night radio showcase — UKG, UK bass, and breaks played with full conviction. "
            "The journey matters: chapters, not a playlist.' "
            "Aim for 10–50 words. Use positive framing ('dark and driven' not 'not melodic'). "
            "Also works in playlist mode: overrides the inferred DJ intent brief (from Stage 0) "
            "wherever the two conflict."
        ),
    )
    parser.add_argument(
        "--mix-length",
        type=int,
        default=None,
        metavar="MINUTES",
        dest="mix_length",
        help=(
            "Target set length in minutes (works in both --genre and --playlist mode). "
            "Scales the number of tracks Stage 2 selects — e.g. 60 targets ~15 tracks, 90 targets ~22. "
            "When the offered tracks carry real duration data, the target is derived from their "
            "actual mean length rather than the ~4 min/track estimate. "
            "Without this flag, runs default to 10–14 tracks as before."
        ),
    )
    parser.add_argument(
        "--locked",
        action="store_true",
        help=(
            "Playlist mode only. Prevents Stage 2 from adding tracks outside the seed playlist. "
            "Stage 2 can only remove or reorder — useful when you have already curated a pool "
            "and just want to trim it to fit a set length."
        ),
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help=(
            "Run a Stage 2 self-critique pass between selection and report (opt-in, "
            "doubles Stage 2 cost). Each concept gets a peer-DJ review surfaced inline "
            "as a CRITIQUE block. Genre mode only — ignored for --playlist runs."
        ),
    )
    parser.add_argument(
        "--no-revise",
        action="store_true",
        dest="no_revise",
        help=(
            "Skip the bounded self-revision pass (#55). By default MixLab attempts one minimal "
            "repair per flagged concept after validation; this disables that pass."
        ),
    )
    parser.add_argument(
        "--resequence",
        action="store_true",
        help=(
            "Apply the sequencer's suggested order improvements to exported concepts (#61). "
            "By default the sequencer only suggests swaps in the report and leaves the exported "
            "order unchanged; this flag applies them."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit verbose canvas scoring diagnostics to stderr. Also enabled by MIXLAB_DEBUG_SCORE=1.",
    )
    parser.add_argument(
        "--stage1-seed",
        type=int,
        default=None,
        dest="stage1_seed",
        help=(
            "Seed for the Stage 1 windowing of oversized pools. Same seed reproduces a prior run "
            "(the effective seed is printed at run start). Default: derived from today's date."
        ),
    )
    parser.add_argument(
        "--directions",
        choices=["mixed", "off", "only"],
        default="mixed",
        help=(
            "Concept directions (#53), genre mode only. "
            "mixed (default) — blend cross-strata creative directions (mood journeys, era "
            "dialogues, label spotlights, artist threads, energy shapes, fresh crates) with "
            "classic BPM-stratum canvases; off — classic canvases only; only — directions only "
            "(falls back to classic when none are feasible). Ignored in playlist mode."
        ),
    )
    parser.add_argument(
        "--risk",
        choices=["low", "medium", "high"],
        default="medium",
        help=(
            "Risk knob (#42), genre mode only. Shifts canvas scoring and Stage 2 framing "
            "toward safety or novelty: high promotes flagged wildcard/concept-anchor tracks "
            "as featured picks and relaxes the BPM/Camelot jump validator thresholds for "
            "transitions explicitly annotated as risky (20 BPM / 5 Camelot, vs 15/4 at "
            "medium — unannotated jumps still use the medium thresholds); low tightens "
            "scoring toward role coverage and anchor safety and lowers the jump thresholds "
            "(10 BPM / 3 Camelot); medium (default) is unchanged from prior behaviour. "
            "Composes with --directions — the risk knob affects canvas scoring and "
            "validation regardless of which direction mode is active. Ignored in playlist "
            "mode (playlist mode has its own risk-tolerance path from the Stage 0 intent "
            "brief, #54)."
        ),
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help=(
            "Record or list feedback for previously generated concepts (#52). No LLM calls, no "
            "network. Alone, lists the most recent run's concepts and their feedback state. "
            "Combine with --concept and --verdict to record a verdict."
        ),
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "With --feedback: title (case-insensitive) or concept_id prefix identifying which "
            "concept to record feedback for. Matches the most recent run containing it."
        ),
    )
    parser.add_argument(
        "--verdict",
        choices=list(_FEEDBACK_VERDICTS),
        default=None,
        help="With --feedback --concept: the feedback verdict to record.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        metavar="TEXT",
        help="With --feedback --concept --verdict: optional free-text note stored alongside the verdict.",
    )
    parser.add_argument(
        "--prep",
        action="store_true",
        help="Rank uncued tracks by cue-prep payoff — offline, no API/LLM (#74).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="With --prep: number of top targets to show (default 20).",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=(
            "Run the MixLab Anywhere worker loop (#91): poll the API for queued runs, "
            "execute each as a subprocess, and push artifacts back. Requires MIXLAB_API_URL "
            "and MIXLAB_API_SECRET. Poll cadence: MIXLAB_WORKER_POLL_SECONDS (default 30)."
        ),
    )
    parser.add_argument(
        "--worker-once",
        action="store_true",
        dest="worker_once",
        help="Run a single worker poll cycle then exit (for cron/agent/manual use). Takes precedence over --worker.",
    )
    args = parser.parse_args()

    if args.worker or args.worker_once:
        _run_worker_cli(once=args.worker_once)
        return

    _validate_range_args(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        min_year=args.min_year,
        max_year=args.max_year,
    )
    debug = args.debug or bool(os.environ.get("MIXLAB_DEBUG_SCORE"))
    if args.feedback:
        sys.exit(run_feedback(args.concept, args.verdict, args.notes, _HISTORY_PATH))

    if args.prep:
        sys.exit(run_prep(args.genre, args.top))

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

    _warn_intent(args.intent)

    if args.playlist:
        if args.deep:
            print(
                "--deep ignored in playlist mode — critique pass not yet supported on the variant-scoring path.",
                file=sys.stderr,
            )
        if args.directions != "mixed":
            print(
                "--directions ignored in playlist mode — concept directions apply to genre runs only.",
                file=sys.stderr,
            )
        if args.risk != "medium":
            print(
                "--risk ignored in playlist mode — playlist mode derives risk tolerance from the "
                "Stage 0 intent brief (#54) instead.",
                file=sys.stderr,
            )
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
                intent=args.intent,
                debug=debug,
                mix_length=args.mix_length,
                locked=args.locked,
            )
        )
        return

    if args.locked:
        print(
            "--locked is only used in playlist mode (--playlist). Ignored for genre runs.",
            file=sys.stderr,
        )

    asyncio.run(
        run(
            args.genre,
            export_dir,
            args.mode,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            min_year=args.min_year,
            max_year=args.max_year,
            intent=args.intent,
            deep=args.deep,
            debug=debug,
            stage1_seed=args.stage1_seed,
            mix_length=args.mix_length,
            directions=args.directions,
            no_revise=args.no_revise,
            risk=cast("RiskTolerance", args.risk),
            resequence=args.resequence,
        )
    )


if __name__ == "__main__":
    main()
