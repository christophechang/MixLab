from __future__ import annotations

import re
import unicodedata

from mixlab.models import PlayedTrack, Track

_FEAT_RE = re.compile(
    r"\s*[\(\[]\s*(?:feat\.|ft\.|featuring)[^\)\]]*[\)\]]"  # (feat. ...) or [feat. ...]
    r"|\s+(?:feat\.|ft\.|featuring)\s+\S+(?:\s+\S+)*",  # bare feat. at end of string
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s-]")
_DASH_RE = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]")


def normalise(text: str) -> str:
    text = text.lower()
    text = _DASH_RE.sub("-", text)
    text = unicodedata.normalize("NFKD", text)
    text = _FEAT_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    text = " ".join(text.split())
    return text


def is_played(track: Track, played: list[PlayedTrack]) -> bool:
    needle = normalise(track.artist) + " " + normalise(track.title)
    return any(normalise(p.artist) + " " + normalise(p.title) == needle for p in played)


def filter_unplayed(tracks: list[Track], played: list[PlayedTrack]) -> list[Track]:
    return [t for t in tracks if not is_played(t, played)]
