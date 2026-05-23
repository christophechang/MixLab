from __future__ import annotations

import re
import unicodedata

from mixlab.models import PlayedTrack, Track

_FEAT_RE = re.compile(
    r"\s*[\(\[]\s*(?:feat\.?|ft\.?|featuring)[^\)\]]*[\)\]]"  # (feat. ...) or (feat ...) or [feat. ...]
    r"|\s+(?:feat\.?|ft\.?|featuring)\s+\S+(?:\s+\S+)*",  # bare feat./feat at end of string
    re.IGNORECASE,
)
# Parenthesised/bracketed version suffixes that add no identifying information.
# Stripped from both catalog and Rekordbox titles before key comparison so that
# "That's Right" and "That's Right (Original Mix)" resolve to the same track.
_VERSION_RE = re.compile(
    r"\s*[\(\[]\s*(?:"
    r"original[_ ]mix|original"
    r"|extended[_ ]mix|extended"
    r"|radio[_ ]edit"
    r"|club[_ ]mix"
    r"|album[_ ]version|lp[_ ]version"
    r"|vip"
    r"|dub[_ ]mix|dub"
    r"|vocal[_ ]mix"
    r"|instrumental"
    r"|remaster(?:ed)?"
    r"|\d{4}\s*(?:remaster(?:ed)?|remix|mix)"
    r")\s*[\)\]]"
    r"|\s+original[_ ]mix\b",  # bare "Original Mix" / "Original_Mix" without brackets
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s-]")
_DASH_RE = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]")


def normalise(text: str) -> str:
    text = text.lower()
    text = _DASH_RE.sub("-", text)
    text = unicodedata.normalize("NFKD", text)
    text = _FEAT_RE.sub("", text)
    text = _VERSION_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    text = " ".join(text.split())
    return text


def _played_key(p: PlayedTrack) -> str:
    return normalise(p.artist) + " " + normalise(p.title)


def is_played(track: Track, played: list[PlayedTrack]) -> bool:
    needle = normalise(track.artist) + " " + normalise(track.title)
    return any(_played_key(p) == needle for p in played)


def filter_unplayed(tracks: list[Track], played: list[PlayedTrack]) -> list[Track]:
    played_set = {_played_key(p) for p in played}
    return [t for t in tracks if normalise(t.artist) + " " + normalise(t.title) not in played_set]


def filter_played(tracks: list[Track], played: list[PlayedTrack]) -> list[Track]:
    played_set = {_played_key(p) for p in played}
    return [t for t in tracks if normalise(t.artist) + " " + normalise(t.title) in played_set]
