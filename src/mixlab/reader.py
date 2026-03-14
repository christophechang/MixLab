from __future__ import annotations

import sys
from pathlib import Path

from lxml import etree

from mixlab.models import Track


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

        tracks.append(
            Track(
                track_id=track_id,
                artist=artist,
                title=title,
                bpm=float(bpm_raw),
                camelot_key=key_raw,
                genre=genre,
            )
        )

    print(f"Parsed {len(tracks)} valid tracks, excluded {excluded} tracks.", file=sys.stderr)
    return tracks


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
