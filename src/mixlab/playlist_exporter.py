from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

from lxml import etree

from mixlab.models import MixConcept


def parse_raw_tracks(xml_path: Path) -> dict[str, etree._Element]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Rekordbox XML not found: {xml_path}")

    tree = etree.parse(str(xml_path))  # noqa: S320 — local file, not user-supplied input
    root = tree.getroot()

    collection = root.find(".//COLLECTION")
    if collection is None:
        raise ValueError("No <COLLECTION> node found in Rekordbox XML.")

    result: dict[str, etree._Element] = {}
    for element in collection.findall("TRACK"):
        track_id = element.get("TrackID", "")
        location = element.get("Location", "")
        if not track_id or location.startswith("file://localhostsoundcloud"):
            continue
        result[track_id] = element

    return result


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:80]


def _build_concept_xml_root(
    concept: MixConcept,
    raw_tracks: dict[str, etree._Element],
) -> etree._Element | None:
    track_elements = [copy.deepcopy(raw_tracks[tid]) for tid in concept.track_ids if tid in raw_tracks]
    if not track_elements:
        return None

    dj_playlists = etree.Element("DJ_PLAYLISTS", Version="1.0.0")
    etree.SubElement(dj_playlists, "PRODUCT", Name="rekordbox", Version="7.2.11", Company="AlphaTheta")

    collection = etree.SubElement(dj_playlists, "COLLECTION", Entries=str(len(track_elements)))
    for elem in track_elements:
        collection.append(elem)

    playlists = etree.SubElement(dj_playlists, "PLAYLISTS")
    root_node = etree.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    playlist_node = etree.SubElement(
        root_node,
        "NODE",
        Name=concept.title,
        Type="1",
        KeyType="0",
        Entries=str(len(track_elements)),
    )
    for elem in track_elements:
        etree.SubElement(playlist_node, "TRACK", Key=elem.get("TrackID", ""))

    return dj_playlists


def generate_concept_xml_bytes(
    concept: MixConcept,
    raw_tracks: dict[str, etree._Element],
) -> bytes | None:
    root = _build_concept_xml_root(concept, raw_tracks)
    if root is None:
        return None
    result = etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return bytes(result)


def generate_concept_xml_attachments(
    concepts: list[MixConcept],
    raw_tracks: dict[str, etree._Element],
) -> list[tuple[str, bytes]]:
    attachments: list[tuple[str, bytes]] = []
    for concept in concepts:
        xml_bytes = generate_concept_xml_bytes(concept, raw_tracks)
        if xml_bytes is not None:
            attachments.append((_safe_filename(concept.title) + ".xml", xml_bytes))
    return attachments


def export_concept_playlists(
    concepts: list[MixConcept],
    raw_tracks: dict[str, etree._Element],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for concept in concepts:
        root = _build_concept_xml_root(concept, raw_tracks)
        if root is None:
            print(f"WARNING: Skipping concept '{concept.title}' — no matching tracks in XML.", file=sys.stderr)
            continue

        filename = _safe_filename(concept.title) + ".xml"
        out_path = output_dir / filename
        etree.ElementTree(root).write(str(out_path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Merged XML — single Rekordbox library file with all playlists
# ---------------------------------------------------------------------------


def build_merged_xml(
    concepts: list[MixConcept],
    raw_tracks: dict[str, etree._Element],
    folder_name: str = "Generated Playlists",
) -> etree._Element | None:
    """Build one DJ_PLAYLISTS XML tree with a shared COLLECTION and all concepts as playlists."""
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for concept in concepts:
        for tid in concept.track_ids:
            if tid in raw_tracks and tid not in seen_ids:
                ordered_ids.append(tid)
                seen_ids.add(tid)

    if not ordered_ids:
        return None

    unique_tracks = {tid: copy.deepcopy(raw_tracks[tid]) for tid in ordered_ids}

    dj_playlists = etree.Element("DJ_PLAYLISTS", Version="1.0.0")
    etree.SubElement(dj_playlists, "PRODUCT", Name="rekordbox", Version="7.2.11", Company="AlphaTheta")

    collection = etree.SubElement(dj_playlists, "COLLECTION", Entries=str(len(unique_tracks)))
    for tid in ordered_ids:
        collection.append(unique_tracks[tid])

    valid_concepts = [c for c in concepts if any(tid in unique_tracks for tid in c.track_ids)]

    playlists_el = etree.SubElement(dj_playlists, "PLAYLISTS")
    root_node = etree.SubElement(playlists_el, "NODE", Type="0", Name="ROOT", Count="1")
    folder_node = etree.SubElement(root_node, "NODE", Type="0", Name=folder_name, Count=str(len(valid_concepts)))

    for concept in valid_concepts:
        concept_track_ids = [tid for tid in concept.track_ids if tid in unique_tracks]
        playlist_node = etree.SubElement(
            folder_node,
            "NODE",
            Name=concept.title,
            Type="1",
            KeyType="0",
            Entries=str(len(concept_track_ids)),
        )
        for tid in concept_track_ids:
            etree.SubElement(playlist_node, "TRACK", Key=tid)

    return dj_playlists


def generate_merged_xml_bytes(
    concepts: list[MixConcept],
    raw_tracks: dict[str, etree._Element],
    folder_name: str = "Generated Playlists",
) -> bytes | None:
    root = build_merged_xml(concepts, raw_tracks, folder_name)
    if root is None:
        return None
    return bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True))


def export_merged_xml(
    concepts: list[MixConcept],
    raw_tracks: dict[str, etree._Element],
    output_path: Path,
    folder_name: str = "Generated Playlists",
) -> Path | None:
    root = build_merged_xml(concepts, raw_tracks, folder_name)
    if root is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(str(output_path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return output_path
