from __future__ import annotations

from pathlib import Path
from typing import Literal

from lxml import etree

RefusalReason = Literal["compressed", "doctype", "unreadable", "rows"]

MAX_TRACK_ROWS = 20_000  # preflight.json maxTracks (R-V)
PROLOG_SCAN_BYTES = 65_536  # M2a-D19, the API's bound; a longer prolog is not refused here

_GZIP_MAGIC = b"\x1f\x8b"
_UTF8_BOM = b"\xef\xbb\xbf"
_DOCTYPE = b"<!DOCTYPE"
_XML_WHITESPACE = b" \t\r\n"
_FEED_CHUNK_BYTES = 1024 * 1024


class CollectionRefused(ValueError):  # noqa: N818 — the name is the M2b plan's interface (spec §7)
    """The Rekordbox XML is refused before (or instead of) being read."""

    def __init__(self, reason: RefusalReason) -> None:
        super().__init__(f"Rekordbox XML refused: {reason}")
        self.reason: RefusalReason = reason


class _RowCounter:
    """lxml parser target: counts TRACK rows directly under the first COLLECTION, keeping no element.

    Matches the reader's `root.find(".//COLLECTION")` then `findall("TRACK")`: the first COLLECTION
    below the root in document order, and only its direct TRACK children.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._collection_depth: int | None = None
        self._collection_closed = False
        self._rows = 0

    def start(self, tag: str | bytes, attrib: dict[str | bytes, str | bytes]) -> None:
        self._depth += 1
        if self._collection_closed:
            return
        if self._collection_depth is None:
            if tag == "COLLECTION" and self._depth > 1:
                self._collection_depth = self._depth
            return
        if tag == "TRACK" and self._depth == self._collection_depth + 1:
            self._rows += 1
            if self._rows > MAX_TRACK_ROWS:
                raise CollectionRefused("rows")

    def end(self, tag: str | bytes) -> None:
        if self._depth == self._collection_depth:
            self._collection_depth = None
            self._collection_closed = True
        self._depth -= 1

    def data(self, data: str | bytes) -> None:
        return None

    def comment(self, text: str | bytes) -> None:
        return None

    def close(self) -> None:
        return None


def _is_name_start(byte: int) -> bool:
    return ord("A") <= byte <= ord("Z") or ord("a") <= byte <= ord("z") or byte in (ord("_"), ord(":")) or byte >= 0x80


def _first_element_start(data: bytes) -> int:
    for i in range(len(data) - 1):
        if data[i] == ord("<") and _is_name_start(data[i + 1]):
            return i
    return -1


def _scan_prolog(head: bytes) -> None:
    """The API's prolog scan (PrologScan.cs) plus its first-byte rule; `head` is the file's first bytes."""
    data = head[len(_UTF8_BOM) :] if head.startswith(_UTF8_BOM) else head
    bounded = data[:PROLOG_SCAN_BYTES]

    content = bounded.lstrip(_XML_WHITESPACE)
    if not content or content[0] != ord("<"):
        raise CollectionRefused("unreadable")

    end = _first_element_start(bounded)
    prolog = bounded if end < 0 else bounded[:end]
    if _DOCTYPE in prolog:
        raise CollectionRefused("doctype")


def _count_rows(path: Path) -> None:
    parser = etree.XMLParser(
        target=_RowCounter(), resolve_entities=False, no_network=True, huge_tree=False, load_dtd=False
    )
    with path.open("rb") as handle:
        while chunk := handle.read(_FEED_CHUNK_BYTES):
            parser.feed(chunk)
    parser.close()


def parse_hardened(path: Path) -> etree._ElementTree:
    """Parse a Rekordbox XML file, refusing compressed files, DOCTYPEs and more than MAX_TRACK_ROWS rows.

    A malformed file raises lxml's own `XMLSyntaxError` from the row count, as a plain parse would.
    """
    with path.open("rb") as handle:
        head = handle.read(len(_UTF8_BOM) + PROLOG_SCAN_BYTES)

    if head.startswith(_GZIP_MAGIC):
        raise CollectionRefused("compressed")

    _scan_prolog(head)
    _count_rows(path)

    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, load_dtd=False)
    tree = etree.parse(str(path), parser=parser)
    doctype: str = tree.docinfo.doctype  # type: ignore[union-attr]  # lxml-stubs 0.5.1 omits DocInfo.doctype
    if doctype:
        raise CollectionRefused("doctype")
    return tree
