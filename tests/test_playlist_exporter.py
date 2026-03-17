from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from lxml import etree

from mixlab.models import MixConcept
from mixlab.playlist_exporter import (
    _safe_filename,
    build_merged_xml,
    export_concept_playlists,
    export_merged_xml,
    generate_concept_xml_attachments,
    generate_concept_xml_bytes,
    generate_merged_xml_bytes,
    parse_raw_tracks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <PRODUCT Name="rekordbox" Version="7.2.11" Company="AlphaTheta"/>
      <COLLECTION Entries="3">
        <TRACK TrackID="101" Name="Alpha" Artist="Artist A" AverageBpm="174.00" Tonality="8A"
               Genre="Drum &amp; Bass" Location="file://localhost/Users/dj/Music/alpha.mp3">
          <TEMPO Inizio="0.100" Bpm="174.00" Metro="4/4" Battito="1"/>
        </TRACK>
        <TRACK TrackID="102" Name="Beta" Artist="Artist B" AverageBpm="126.00" Tonality="9A"
               Genre="House" Location="file://localhost/Users/dj/Music/beta.mp3"/>
        <TRACK TrackID="103" Name="Cloud" Artist="Artist C" AverageBpm="128.00" Tonality="7A"
               Genre="House" Location="file://localhostsoundcloud:tracks:99999"/>
      </COLLECTION>
    </DJ_PLAYLISTS>
""")


def _write_xml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "rekordbox.xml"
    p.write_text(content)
    return p


def _make_concept(title: str, track_ids: list[str]) -> MixConcept:
    return MixConcept(title=title, mood="test mood", track_ids=track_ids)


# ---------------------------------------------------------------------------
# parse_raw_tracks
# ---------------------------------------------------------------------------


def test_parse_raw_tracks_returns_local_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    result = parse_raw_tracks(xml_path)
    assert set(result.keys()) == {"101", "102"}


def test_parse_raw_tracks_excludes_soundcloud(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    result = parse_raw_tracks(xml_path)
    assert "103" not in result


def test_parse_raw_tracks_preserves_children(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    result = parse_raw_tracks(xml_path)
    tempo_nodes = result["101"].findall("TEMPO")
    assert len(tempo_nodes) == 1
    assert tempo_nodes[0].get("Bpm") == "174.00"


def test_parse_raw_tracks_raises_if_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_raw_tracks(tmp_path / "nonexistent.xml")


def test_parse_raw_tracks_raises_if_no_collection(tmp_path: Path) -> None:
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text('<?xml version="1.0"?><DJ_PLAYLISTS Version="1.0.0"/>')
    with pytest.raises(ValueError, match="COLLECTION"):
        parse_raw_tracks(xml_path)


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Deep Pulse", "Deep_Pulse"),
        ('Bad/Name:Here"foo', "Bad_Name_Here_foo"),
        ("  leading spaces  ", "leading_spaces"),
        ("A" * 100, "A" * 80),
    ],
)
def test_safe_filename_sanitizes(name: str, expected: str) -> None:
    assert _safe_filename(name) == expected


# ---------------------------------------------------------------------------
# export_concept_playlists
# ---------------------------------------------------------------------------


def _load_exported(path: Path) -> etree._Element:
    return etree.parse(str(path)).getroot()


def test_export_concept_playlists_creates_xml_files(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Dark Energy", ["101", "102"])]
    out_dir = tmp_path / "playlists"

    written = export_concept_playlists(concepts, raw, out_dir)

    assert len(written) == 1
    assert written[0].name == "Dark_Energy.xml"
    assert written[0].exists()


def test_export_concept_playlists_collection_contains_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Dark Energy", ["101", "102"])]
    out_dir = tmp_path / "playlists"

    written = export_concept_playlists(concepts, raw, out_dir)
    root = _load_exported(written[0])

    collection = root.find("COLLECTION")
    assert collection is not None
    assert collection.get("Entries") == "2"
    track_ids = {el.get("TrackID") for el in collection.findall("TRACK")}
    assert track_ids == {"101", "102"}


def test_export_concept_playlists_playlist_node_references_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Galactic Drift", ["101"])]
    out_dir = tmp_path / "playlists"

    written = export_concept_playlists(concepts, raw, out_dir)
    root = _load_exported(written[0])

    playlist_node = root.find(".//PLAYLISTS/NODE/NODE")
    assert playlist_node is not None
    assert playlist_node.get("Name") == "Galactic Drift"
    assert playlist_node.get("Entries") == "1"
    keys = [el.get("Key") for el in playlist_node.findall("TRACK")]
    assert keys == ["101"]


def test_export_concept_playlists_skips_unknown_track_ids(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Ghost", ["999"])]
    out_dir = tmp_path / "playlists"

    written = export_concept_playlists(concepts, raw, out_dir)
    assert written == []


def test_export_concept_playlists_multiple_concepts_independent(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [
        _make_concept("Set One", ["101"]),
        _make_concept("Set Two", ["102"]),
    ]
    out_dir = tmp_path / "playlists"

    written = export_concept_playlists(concepts, raw, out_dir)
    assert len(written) == 2

    root_one = _load_exported(written[0])
    root_two = _load_exported(written[1])

    ids_one = {el.get("TrackID") for el in root_one.findall(".//COLLECTION/TRACK")}
    ids_two = {el.get("TrackID") for el in root_two.findall(".//COLLECTION/TRACK")}
    assert ids_one == {"101"}
    assert ids_two == {"102"}


def test_export_concept_playlists_preserves_track_children(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("With Tempo", ["101"])]
    out_dir = tmp_path / "playlists"

    written = export_concept_playlists(concepts, raw, out_dir)
    root = _load_exported(written[0])

    track_el = root.find(".//COLLECTION/TRACK[@TrackID='101']")
    assert track_el is not None
    assert track_el.find("TEMPO") is not None


def test_export_concept_playlists_creates_output_dir(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("New Dir", ["101"])]
    out_dir = tmp_path / "deep" / "nested" / "playlists"

    export_concept_playlists(concepts, raw, out_dir)
    assert out_dir.exists()


# ---------------------------------------------------------------------------
# generate_concept_xml_bytes
# ---------------------------------------------------------------------------


def test_generate_concept_xml_bytes_returns_bytes(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    result = generate_concept_xml_bytes(_make_concept("Test", ["101"]), raw)
    assert isinstance(result, bytes)
    assert b"DJ_PLAYLISTS" in result


def test_generate_concept_xml_bytes_returns_none_for_unknown_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    result = generate_concept_xml_bytes(_make_concept("Ghost", ["999"]), raw)
    assert result is None


def test_generate_concept_xml_bytes_output_is_valid_xml(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    xml_bytes = generate_concept_xml_bytes(_make_concept("Check", ["101", "102"]), raw)
    assert xml_bytes is not None
    root = etree.fromstring(xml_bytes)
    assert root.tag == "DJ_PLAYLISTS"
    assert root.find(".//COLLECTION/TRACK[@TrackID='101']") is not None


# ---------------------------------------------------------------------------
# generate_concept_xml_attachments
# ---------------------------------------------------------------------------


def test_generate_concept_xml_attachments_returns_filename_and_bytes(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Dark Energy", ["101"]), _make_concept("Ghost", ["999"])]

    result = generate_concept_xml_attachments(concepts, raw)

    assert len(result) == 1
    filename, data = result[0]
    assert filename == "Dark_Energy.xml"
    assert isinstance(data, bytes)


def test_generate_concept_xml_attachments_one_per_valid_concept(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"]), _make_concept("Set B", ["102"])]

    result = generate_concept_xml_attachments(concepts, raw)

    assert len(result) == 2
    filenames = [f for f, _ in result]
    assert filenames == ["Set_A.xml", "Set_B.xml"]


# ---------------------------------------------------------------------------
# build_merged_xml
# ---------------------------------------------------------------------------


def test_build_merged_xml_shared_collection_deduplicates_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    # Track 102 appears in both concepts — should appear once in COLLECTION
    concepts = [_make_concept("Set A", ["101", "102"]), _make_concept("Set B", ["102"])]

    root = build_merged_xml(concepts, raw)

    assert root is not None
    collection = root.find("COLLECTION")
    assert collection is not None
    assert collection.get("Entries") == "2"
    track_ids = {el.get("TrackID") for el in collection.findall("TRACK")}
    assert track_ids == {"101", "102"}


def test_build_merged_xml_playlist_keys_all_exist_in_collection(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"]), _make_concept("Set B", ["102"])]

    root = build_merged_xml(concepts, raw)

    assert root is not None
    collection_ids = {el.get("TrackID") for el in root.findall(".//COLLECTION/TRACK")}
    playlist_keys = {el.get("Key") for el in root.findall(".//PLAYLISTS//TRACK")}
    assert playlist_keys <= collection_ids


def test_build_merged_xml_collection_entry_count_matches_unique_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101", "102"]), _make_concept("Set B", ["102"])]

    root = build_merged_xml(concepts, raw)

    assert root is not None
    collection = root.find("COLLECTION")
    assert collection is not None
    actual = collection.findall("TRACK")
    assert collection.get("Entries") == str(len(actual))


def test_build_merged_xml_is_well_formed_xml(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Test", ["101", "102"])]

    root = build_merged_xml(concepts, raw)

    assert root is not None
    xml_bytes = bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True))
    parsed = etree.fromstring(xml_bytes)
    assert parsed.tag == "DJ_PLAYLISTS"


def test_build_merged_xml_returns_none_when_no_tracks_match(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    root = build_merged_xml([_make_concept("Ghost", ["999"])], raw)
    assert root is None


def test_build_merged_xml_folder_node_contains_all_playlists(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"]), _make_concept("Set B", ["102"])]

    root = build_merged_xml(concepts, raw)

    assert root is not None
    folder = root.find(".//PLAYLISTS/NODE/NODE[@Name='Generated Playlists']")
    assert folder is not None
    playlist_nodes = folder.findall("NODE")
    assert len(playlist_nodes) == 2
    names = [n.get("Name") for n in playlist_nodes]
    assert names == ["Set A", "Set B"]


def test_build_merged_xml_unplayed_ids_adds_playlist(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"])]

    root = build_merged_xml(concepts, raw, unplayed_ids=["101", "102"])

    assert root is not None
    folder = root.find(".//PLAYLISTS/NODE/NODE[@Name='Generated Playlists']")
    assert folder is not None
    playlist_nodes = folder.findall("NODE")
    assert len(playlist_nodes) == 2
    names = [n.get("Name") for n in playlist_nodes]
    assert "All Unplayed Tunes" in names


def test_build_merged_xml_unplayed_ids_playlist_contains_all_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"])]

    root = build_merged_xml(concepts, raw, unplayed_ids=["101", "102"])

    assert root is not None
    unplayed_node = root.find(".//PLAYLISTS//NODE[@Name='All Unplayed Tunes']")
    assert unplayed_node is not None
    assert unplayed_node.get("Entries") == "2"
    keys = [el.get("Key") for el in unplayed_node.findall("TRACK")]
    assert set(keys) == {"101", "102"}


def test_build_merged_xml_unplayed_ids_extra_tracks_added_to_collection(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    # Concept only uses 101; unplayed includes 102 as well.
    concepts = [_make_concept("Set A", ["101"])]

    root = build_merged_xml(concepts, raw, unplayed_ids=["101", "102"])

    assert root is not None
    collection = root.find("COLLECTION")
    assert collection is not None
    track_ids = {el.get("TrackID") for el in collection.findall("TRACK")}
    assert track_ids == {"101", "102"}
    assert collection.get("Entries") == "2"


def test_build_merged_xml_unplayed_ids_ignores_unknown_track_ids(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"])]

    root = build_merged_xml(concepts, raw, unplayed_ids=["101", "999"])

    assert root is not None
    unplayed_node = root.find(".//PLAYLISTS//NODE[@Name='All Unplayed Tunes']")
    assert unplayed_node is not None
    keys = [el.get("Key") for el in unplayed_node.findall("TRACK")]
    assert "999" not in keys
    assert keys == ["101"]


def test_build_merged_xml_no_unplayed_ids_omits_playlist(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"])]

    root = build_merged_xml(concepts, raw)

    assert root is not None
    unplayed_node = root.find(".//PLAYLISTS//NODE[@Name='All Unplayed Tunes']")
    assert unplayed_node is None


# ---------------------------------------------------------------------------
# generate_merged_xml_bytes
# ---------------------------------------------------------------------------


def test_generate_merged_xml_bytes_returns_bytes(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    result = generate_merged_xml_bytes([_make_concept("Test", ["101"])], raw)
    assert isinstance(result, bytes)
    assert b"DJ_PLAYLISTS" in result


def test_generate_merged_xml_bytes_returns_none_for_no_matches(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    result = generate_merged_xml_bytes([_make_concept("Ghost", ["999"])], raw)
    assert result is None


# ---------------------------------------------------------------------------
# export_merged_xml
# ---------------------------------------------------------------------------


def test_export_merged_xml_writes_single_file(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    concepts = [_make_concept("Set A", ["101"]), _make_concept("Set B", ["102"])]
    out_path = tmp_path / "out" / "rekordbox_export.xml"

    result = export_merged_xml(concepts, raw, out_path)

    assert result == out_path
    assert out_path.exists()
    root = etree.parse(str(out_path)).getroot()
    assert root.tag == "DJ_PLAYLISTS"


def test_export_merged_xml_collection_count_matches_unique_tracks(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    # Track 102 shared — should appear once in COLLECTION
    concepts = [_make_concept("Set A", ["101", "102"]), _make_concept("Set B", ["102"])]
    out_path = tmp_path / "rekordbox_export.xml"

    export_merged_xml(concepts, raw, out_path)

    root = etree.parse(str(out_path)).getroot()
    collection = root.find("COLLECTION")
    assert collection is not None
    actual = collection.findall("TRACK")
    assert collection.get("Entries") == str(len(actual))
    assert len(actual) == 2


def test_export_merged_xml_creates_parent_dirs(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    out_path = tmp_path / "deep" / "nested" / "rekordbox_export.xml"

    export_merged_xml([_make_concept("Test", ["101"])], raw, out_path)

    assert out_path.exists()


def test_export_merged_xml_returns_none_for_no_matches(tmp_path: Path) -> None:
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    raw = parse_raw_tracks(xml_path)
    result = export_merged_xml([_make_concept("Ghost", ["999"])], raw, tmp_path / "out.xml")
    assert result is None
