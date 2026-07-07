from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from lxml import etree

from mixlab.models import GridAnchor, MixPoints, Track

_MIK_ENERGY_RE: re.Pattern[str] = re.compile(r"\bEnergy\s+(\d+)", re.IGNORECASE)

_COLOUR_TO_CONFIDENCE: dict[str, Literal["high", "medium", "low"]] = {
    "#00ff00": "high",
    "0x00ff00": "high",
    "#ffa500": "medium",
    "0xffa500": "medium",
    "#ff0000": "low",
    "0xff0000": "low",
}


def _parse_mik_energy(comments: str) -> int | None:
    m = _MIK_ENERGY_RE.search(comments)
    if not m:
        return None
    return int(m.group(1))


_TAGS_RE: re.Pattern[str] = re.compile(r"/\*\s*(.+?)\s*\*/")


def _parse_tags(comments: str) -> list[str]:
    m = _TAGS_RE.search(comments)
    if not m:
        return []
    return [t.strip() for t in m.group(1).split("/") if t.strip()]


def _parse_grid(element: etree._Element) -> list[GridAnchor]:
    anchors: list[GridAnchor] = []
    for tempo in element.findall("TEMPO"):
        inizio_raw = tempo.get("Inizio", "")
        bpm_raw = tempo.get("Bpm", "")
        try:
            inizio = float(inizio_raw)
            bpm = float(bpm_raw)
        except ValueError:
            continue

        battito_raw = tempo.get("Battito", "1")
        try:
            battito = int(battito_raw)
        except ValueError:
            battito = 1

        anchors.append(
            GridAnchor(
                inizio=inizio,
                bpm=bpm,
                metro=tempo.get("Metro") or "4/4",
                battito=battito,
            )
        )

    anchors.sort(key=lambda a: a.inizio)
    return anchors


def _parse_marks(element: etree._Element) -> list[tuple[float, float | None]]:
    marks: list[tuple[float, float | None]] = []
    for mark in element.findall("POSITION_MARK"):
        start_raw = mark.get("Start", "")
        try:
            start = float(start_raw)
        except ValueError:
            continue

        end_raw = mark.get("End")
        end: float | None = None
        if end_raw is not None:
            try:
                end = float(end_raw)
            except ValueError:
                end = None

        marks.append((start, end))

    marks.sort(key=lambda m: m[0])
    return marks


def _metro_numerator(metro: str) -> int:
    numerator_str = metro.split("/", 1)[0] if "/" in metro else metro
    try:
        return int(numerator_str)
    except ValueError:
        return 4


def _bars_between(t_start: float, t_end: float, grid: list[GridAnchor]) -> float | None:
    """Bars spanned by [t_start, t_end], integrating piecewise across grid anchors.

    Positions before the first anchor use the first anchor's BPM. Beats-per-bar is
    derived from the first anchor's Metro numerator (unparseable → 4).
    """
    if not grid or t_end < t_start:
        return None

    beats_per_bar = _metro_numerator(grid[0].metro)

    boundaries: list[float] = [float("-inf")] + [a.inizio for a in grid] + [float("inf")]
    bpms: list[float] = [grid[0].bpm] + [a.bpm for a in grid]

    beats = 0.0
    for seg_start, seg_end, bpm in zip(boundaries, boundaries[1:], bpms, strict=False):
        overlap_start = max(t_start, seg_start)
        overlap_end = min(t_end, seg_end)
        if overlap_end > overlap_start:
            beats += (overlap_end - overlap_start) * bpm / 60

    return round(beats / beats_per_bar, 1)


def derive_mix_points(
    marks: list[tuple[float, float | None]],
    grid: list[GridAnchor],
    duration_secs: int | None,
) -> MixPoints | None:
    """Derive mix-in/mix-out cues from position marks per the owner's conventions.

    First mark by start time is always the mix-in point. The last mark by start
    time is treated as the mix-out point only when there are >= 2 marks and its
    start is past the midpoint of the track — a single mark is a partially-prepped
    track (mix-in known, mix-out unknown), and a lone late-track cue could just be
    a hot cue rather than an intentional mix-out marker.
    """
    if not marks:
        return None

    mix_in_secs = marks[0][0]
    intro_bars = _bars_between(0.0, mix_in_secs, grid)

    mix_out_secs: float | None = None
    outro_bars: float | None = None
    last_start = marks[-1][0]
    if len(marks) >= 2 and duration_secs is not None and last_start > 0.5 * duration_secs:
        mix_out_secs = last_start
        outro_bars = _bars_between(mix_out_secs, float(duration_secs), grid)

    loops = [(start, end) for start, end in marks if end is not None]

    return MixPoints(
        mix_in_secs=mix_in_secs,
        mix_out_secs=mix_out_secs,
        intro_bars=intro_bars,
        outro_bars=outro_bars,
        loops=loops,
        cue_count=len(marks),
    )


def parse_collection(xml_path: Path) -> list[Track]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Rekordbox XML not found: {xml_path}. Place it at import/rekordbox.xml.")

    tree = etree.parse(str(xml_path))  # noqa: S320 — local file, not user-supplied input
    root = tree.getroot()

    collection = root.find(".//COLLECTION")
    if collection is None:
        raise ValueError("No <COLLECTION> node found in Rekordbox XML.")

    tracks: list[Track] = []
    excluded = 0

    for element in collection.findall("TRACK"):
        track_id = element.get("TrackID", "")
        artist = element.get("Artist", "")
        title = element.get("Name", "")
        bpm_raw = element.get("AverageBpm", "")
        key_raw = element.get("Tonality", "")
        genre = element.get("Genre", "")
        location = element.get("Location", "")

        if location.startswith("file://localhostsoundcloud"):
            excluded += 1
            continue

        missing: list[str] = []
        if not bpm_raw:
            missing.append("BPM")
        if not key_raw:
            missing.append("Camelot key")

        if missing:
            print(
                f"WARNING: Excluding track '{artist} — {title}' (id={track_id}): missing {', '.join(missing)}",
                file=sys.stderr,
            )
            excluded += 1
            continue

        comments = element.get("Comments", "")
        play_count_raw = element.get("PlayCount", "0")
        year_raw = element.get("Year", "")
        mix_raw = element.get("Mix", "")
        colour_raw = element.get("Colour", "").lower()
        total_time_raw = element.get("TotalTime", "")
        rating_raw = element.get("Rating", "")
        duration_secs = int(total_time_raw) if total_time_raw.isdigit() else None
        grid = _parse_grid(element)
        marks = _parse_marks(element)
        tracks.append(
            Track(
                track_id=track_id,
                artist=artist,
                title=title,
                bpm=float(bpm_raw),
                camelot_key=key_raw,
                genre=genre,
                energy=_parse_mik_energy(comments),
                label=element.get("Label", ""),
                play_count=int(play_count_raw) if play_count_raw.isdigit() else 0,
                tags=_parse_tags(comments),
                year=int(year_raw) if year_raw.isdigit() else None,
                album=element.get("Album", ""),
                remixer=element.get("Remixer", ""),
                mix=[s.strip() for s in mix_raw.split(",") if s.strip()],
                enrichment_confidence=_COLOUR_TO_CONFIDENCE.get(colour_raw, ""),
                duration_secs=duration_secs,
                date_added=element.get("DateAdded", ""),
                rating=int(rating_raw) if rating_raw.isdigit() else None,
                grid=grid,
                mix_points=derive_mix_points(marks, grid, duration_secs),
            )
        )

    print(f"Parsed {len(tracks)} valid tracks, excluded {excluded} tracks.", file=sys.stderr)
    return tracks


def parse_playlists(xml_path: Path) -> dict[str, list[str]]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Rekordbox XML not found: {xml_path}. Place it at import/rekordbox.xml.")

    tree = etree.parse(str(xml_path))  # noqa: S320 — local file, not user-supplied input
    root = tree.getroot()

    playlists_root = root.find(".//PLAYLISTS")
    if playlists_root is None:
        return {}

    playlists: dict[str, list[str]] = {}

    def walk(node: etree._Element, prefix: str = "", include_name: bool = True) -> None:
        node_type = node.get("Type", "")
        node_name = node.get("Name", "")

        if node_type == "0":
            next_prefix = prefix
            if include_name and node_name:
                next_prefix = f"{prefix}{node_name}/"
            for child in node.findall("NODE"):
                walk(child, next_prefix)
            return

        if node_type == "1" and node_name:
            playlists[f"{prefix}{node_name}"] = [track.get("Key", "") for track in node.findall("TRACK")]

    for child in playlists_root.findall("NODE"):
        walk(child, include_name=False)

    return playlists


def apply_bpm_corrections(tracks: list[Track]) -> list[Track]:
    dnb_genres = {"drum & bass", "dnb"}
    result: list[Track] = []

    for track in tracks:
        if track.genre.lower() in dnb_genres and track.bpm < 100:
            corrected = track.bpm * 2
            print(
                f"⚠️ BPM corrected: {track.artist} — {track.title} {track.bpm} BPM → {corrected} BPM",
                file=sys.stderr,
            )
            result.append(track.model_copy(update={"bpm": corrected}))
        else:
            result.append(track)

    return result
