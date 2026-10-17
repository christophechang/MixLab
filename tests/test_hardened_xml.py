from __future__ import annotations

import gzip
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import pytest
from lxml import etree

from mixlab import hardened_xml
from mixlab.hardened_xml import MAX_TRACK_ROWS, PROLOG_SCAN_BYTES, CollectionRefused, parse_hardened

_FIXTURES = Path(__file__).parent / "fixtures" / "run_contract"

_BOM = b"\xef\xbb\xbf"

_SMALL = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<DJ_PLAYLISTS Version="1.0.0">\n'
    '  <COLLECTION Entries="1">\n'
    '    <TRACK TrackID="1" Name="Alpha" Artist="A" AverageBpm="124" Tonality="8A" Genre="House"/>\n'
    "  </COLLECTION>\n"
    "</DJ_PLAYLISTS>\n"
)

_ENTITY_DOC = (
    '<?xml version="1.0" encoding="{encoding}"?>\n'
    '<!DOCTYPE DJ_PLAYLISTS [<!ENTITY x "EXPANDED">]>\n'
    '<DJ_PLAYLISTS Version="1.0.0">\n'
    '  <COLLECTION Entries="1">\n'
    '    <TRACK TrackID="1" Name="Alpha" Artist="A" AverageBpm="124" Tonality="8A" Genre="&x;"/>\n'
    "  </COLLECTION>\n"
    "</DJ_PLAYLISTS>\n"
)

_RecordedParser = Callable[..., etree.XMLParser]


def _write(tmp_path: Path, content: bytes, name: str = "rekordbox.xml") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _refusal(path: Path) -> str:
    with pytest.raises(CollectionRefused) as info:
        parse_hardened(path)
    return info.value.reason


def _forbid_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any parser construction or full parse fail, so a refusal is proven to come from the byte checks."""

    def no_parser(*args: object, **kwargs: object) -> etree.XMLParser:
        raise AssertionError("a parser was constructed")

    def no_parse(*args: object, **kwargs: object) -> etree._ElementTree:
        raise AssertionError("the full parse ran")

    monkeypatch.setattr(etree, "XMLParser", no_parser)
    monkeypatch.setattr(etree, "parse", no_parse)


def _first_genre(tree: etree._ElementTree) -> str | None:
    track = tree.getroot().find(".//TRACK")
    assert track is not None
    return track.get("Genre")


def _write_rows(path: Path, rows: int, cues_per_row: int) -> None:
    cue = '<POSITION_MARK Name="" Type="0" Start="1.000" Num="-1"/>'
    cues = cue * cues_per_row
    with path.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n<DJ_PLAYLISTS Version="1.0.0">\n')
        handle.write(f'<COLLECTION Entries="{rows}">\n')
        for i in range(1, rows + 1):
            genre = "SENTINEL" if i == rows else "House"
            handle.write(
                f'<TRACK TrackID="{i}" Name="T{i}" Artist="A" AverageBpm="124" Tonality="8A" Genre="{genre}">'
                f"{cues}</TRACK>\n"
            )
        handle.write("</COLLECTION>\n</DJ_PLAYLISTS>\n")


# ---------------------------------------------------------------------------
# Step 1 — the two-byte check
# ---------------------------------------------------------------------------


def test_hardened_refuses_compressed_file(tmp_path: Path) -> None:
    path = _write(tmp_path, gzip.compress(_SMALL.encode()), "rekordbox.xml.gz")

    # The risk: libxml2 inflates a gzip file passed by name.
    assert etree.parse(str(path)).getroot().tag == "DJ_PLAYLISTS"

    assert _refusal(path) == "compressed"


# ---------------------------------------------------------------------------
# Step 2 — the prolog scan
# ---------------------------------------------------------------------------


def test_hardened_refuses_doctype_in_prolog_before_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, _ENTITY_DOC.format(encoding="UTF-8").encode())

    # Without the scan, the same bytes expand the entity into the first row's Genre.
    assert _first_genre(etree.parse(str(path))) == "EXPANDED"

    _forbid_parsers(monkeypatch)
    assert _refusal(path) == "doctype"


def test_hardened_refuses_utf16_doctype_after_parse(tmp_path: Path) -> None:
    # UTF-16LE without a byte-order mark: the first byte is `<`, so the byte scan admits it, and
    # neither an element start nor `<!DOCTYPE` appears in its bytes; libxml2 detects the encoding.
    content = _ENTITY_DOC.format(encoding="UTF-16").encode("utf-16-le")
    assert content[:1] == b"<"
    assert b"<!DOCTYPE" not in content
    path = _write(tmp_path, content)

    # Without the docinfo check, the same bytes expand the entity.
    assert _first_genre(etree.parse(str(path))) == "EXPANDED"

    assert _refusal(path) == "doctype"


def test_hardened_utf16_with_byte_order_mark_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"\xff\xfe" + _SMALL.replace("UTF-8", "UTF-16").encode("utf-16-le")
    path = _write(tmp_path, content)
    assert etree.parse(str(path)).getroot().tag == "DJ_PLAYLISTS"

    _forbid_parsers(monkeypatch)
    assert _refusal(path) == "unreadable"


def test_hardened_prolog_scan_matches_api_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Admitted: a UTF-8 BOM is skipped, leading XML whitespace is skipped, and `<_` and a non-ASCII
    # lead byte are element starts, so a `<!DOCTYPE` after one (here in a comment) is not scanned.
    admitted = [
        _BOM + _SMALL.encode(),
        b" \t\r\n" + _SMALL.split("\n", 1)[1].encode(),
        b"<_root><!-- <!DOCTYPE x> --><COLLECTION/></_root>",
        "<Éroot><!-- <!DOCTYPE x> --><COLLECTION/></Éroot>".encode(),
    ]
    for i, content in enumerate(admitted):
        tree = parse_hardened(_write(tmp_path, content, f"admitted-{i}.xml"))
        assert tree.getroot() is not None

    long_padding = "<!--" + "x" * PROLOG_SCAN_BYTES + "-->"
    beyond_bound = f'<?xml version="1.0"?>\n{long_padding}\n<!DOCTYPE R>\n<R/>'.encode()

    _forbid_parsers(monkeypatch)
    # Refused before any parser: the first non-whitespace byte after the BOM must be `<`.
    unreadable = [b"", b" \r\n\t", b"x<R/>", _BOM + b"x<R/>", b"\xff\xfe<\x00R\x00/\x00>\x00", b"\x0c<R/>"]
    for i, content in enumerate(unreadable):
        assert _refusal(_write(tmp_path, content, f"unreadable-{i}.xml")) == "unreadable", content

    # `<?xml` and `<!--` are not element starts; a `<!DOCTYPE` in a prolog comment is refused.
    refused = [
        _BOM + b"<!DOCTYPE R><R/>",
        b'<?xml version="1.0"?><!-- c --><!DOCTYPE R><R/>',
        b"<!-- <!DOCTYPE R> --><R/>",
    ]
    for i, content in enumerate(refused):
        assert _refusal(_write(tmp_path, content, f"doctype-{i}.xml")) == "doctype", content

    # A 65,537-byte prolog carrying a DOCTYPE within the bound is `doctype`, not a length verdict.
    head = b'<?xml version="1.0"?>\n<!DOCTYPE R>\n<!--'
    prolog = head + b"x" * (PROLOG_SCAN_BYTES + 1 - len(head) - 3) + b"-->"
    assert len(prolog) == PROLOG_SCAN_BYTES + 1
    assert _refusal(_write(tmp_path, prolog + b"<R/>", "long-doctype.xml")) == "doctype"

    # Past the bound the scan does not see it; the docinfo check after the parse does.
    monkeypatch.undo()
    assert _refusal(_write(tmp_path, beyond_bound, "beyond-bound.xml")) == "doctype"


def test_hardened_long_prolog_without_doctype_is_admitted(tmp_path: Path) -> None:
    comment = b"<!--" + b"x" * (PROLOG_SCAN_BYTES + 1 - 7) + b"-->"
    assert len(comment) == PROLOG_SCAN_BYTES + 1

    tree = parse_hardened(_write(tmp_path, comment + _SMALL.split("\n", 1)[1].encode()))

    assert tree.getroot().tag == "DJ_PLAYLISTS"


# ---------------------------------------------------------------------------
# Step 3 — the row cap
# ---------------------------------------------------------------------------


def test_hardened_refuses_row_20001_without_building_a_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "rows.xml"
    _write_rows(path, MAX_TRACK_ROWS + 1, cues_per_row=50)

    def no_parse(*args: object, **kwargs: object) -> etree._ElementTree:
        raise AssertionError("the full parse ran")

    monkeypatch.setattr(etree, "parse", no_parse)

    tracemalloc.start()
    started = time.perf_counter()
    try:
        with pytest.raises(CollectionRefused) as info:
            parse_hardened(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - started

    assert info.value.reason == "rows"
    assert peak < 64 * 1024 * 1024, f"peak {peak} bytes in {elapsed:.1f}s"


def test_hardened_admits_20000_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.xml"
    _write_rows(path, MAX_TRACK_ROWS, cues_per_row=0)

    tree = parse_hardened(path)

    collection = tree.getroot().find("COLLECTION")
    assert collection is not None
    assert len(collection.findall("TRACK")) == MAX_TRACK_ROWS


def test_hardened_row_cap_counts_only_the_first_collection(tmp_path: Path) -> None:
    rows = "<TRACK/>" * (MAX_TRACK_ROWS + 1)
    # Rows nested deeper than the first COLLECTION's children, or under a later COLLECTION, are not counted.
    content = (
        f"<DJ_PLAYLISTS><COLLECTION><TRACK/><X>{rows}</X></COLLECTION><COLLECTION>{rows}</COLLECTION></DJ_PLAYLISTS>"
    ).encode()

    tree = parse_hardened(_write(tmp_path, content))

    assert tree.getroot().tag == "DJ_PLAYLISTS"


def test_hardened_malformed_file_raises_syntax_error(tmp_path: Path) -> None:
    path = _write(tmp_path, b"<DJ_PLAYLISTS><COLLECTION></DJ_PLAYLISTS>")

    with pytest.raises(etree.XMLSyntaxError):
        etree.parse(str(path))
    with pytest.raises(etree.XMLSyntaxError):
        parse_hardened(path)


# ---------------------------------------------------------------------------
# Step 4 — the parse
# ---------------------------------------------------------------------------


def test_hardened_parser_flags_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # lxml 6 does not expose a parser's options, so the constructor arguments are recorded.
    real: _RecordedParser = etree.XMLParser
    recorded: list[dict[str, object]] = []

    def recording_parser(**kwargs: object) -> etree.XMLParser:
        recorded.append(dict(kwargs))
        return real(**kwargs)

    monkeypatch.setattr(etree, "XMLParser", recording_parser)

    parse_hardened(_write(tmp_path, _SMALL.encode()))

    assert len(recorded) == 2
    assert isinstance(recorded[0].get("target"), hardened_xml._RowCounter)
    assert "target" not in recorded[1]
    for kwargs in recorded:
        assert kwargs["resolve_entities"] is False
        assert kwargs["no_network"] is True
        assert kwargs["huge_tree"] is False
        assert kwargs["load_dtd"] is False


def test_hardened_doctype_fixture_refused() -> None:
    assert _refusal(_FIXTURES / "s6-doctype.xml") == "doctype"
