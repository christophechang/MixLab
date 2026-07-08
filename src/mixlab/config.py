from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mixlab.models import MixConcept


class CustomGenre(TypedDict):
    genres: list[str]  # api labels from GENRE_MAP
    bpm_range: tuple[float, float] | None  # inclusive hard filter, or None to skip


CUSTOM_GENRES: dict[str, CustomGenre] = {
    "170": CustomGenre(genres=["drum_and_bass", "jungle"], bpm_range=(165.0, 175.0)),
    "140": CustomGenre(genres=["breakbeat", "uk_bass", "uk_garage"], bpm_range=(130.0, 140.0)),
    "4x4": CustomGenre(genres=["house", "electronica", "disco", "progressive", "techno"], bpm_range=None),
    # Full-collection pool for cross-genre journey concepts (#82): the genre_traverse
    # direction needs material spanning multiple tempo regimes (house → UKG →
    # jungle/DnB) linked by pitch-lock ratio bridges. No BPM filter by design.
    "traverse": CustomGenre(
        genres=[
            "house",
            "electronica",
            "disco",
            "progressive",
            "techno",
            "uk_garage",
            "uk_bass",
            "breakbeat",
            "hip_hop",
            "jungle",
            "drum_and_bass",
        ],
        bpm_range=None,
    ),
}

GENRE_MAP: dict[str, list[str]] = {
    "drum_and_bass": ["Drum & Bass", "DnB", "Liquid DnB", "Jungle/Drum'n'bass"],
    "techno": ["Techno", "Dark Techno", "Industrial Techno", "Dub Techno", "Melodic House & Techno"],
    "house": ["House", "Deep House", "Deep house", "Tech House", "Classic House", "Afro House", "Minimal / Deep Tech"],
    "uk_garage": ["UK Garage", "UKG", "2-Step", "Uk Garage", "UK Garage / Bassline"],
    "uk_bass": ["UK Bass", "Uk Bass"],
    "jungle": ["Jungle", "Ragga Jungle", "Rave"],
    "breakbeat": ["Breakbeat", "Breaks", "Nu Skool Breaks", "Hardcore"],
    "electronica": ["Electronica", "Electronic", "Downtempo", "Electronica / Downtempo", "Trip Hop"],
    "hip_hop": [
        "Hip Hop",
        "Hip-Hop",
        "hip hop",
        "Funk",
        "Hip Hop/Rap",
        "Hip hop/r&b",
        "Hip-hop & Rap",
        "Soul, Funk, Jazz",
    ],
    "progressive": ["Progressive"],
    "disco": ["Disco"],
}

IGNORED_GENRES: frozenset[str] = frozenset(
    {
        "Rock",
        "Reggae",
        "80s Classics",
        "Loop Samples",
        "sad rap",
        "R & B",
        "R&B",
        "R&B & Soul",
        "Pop",
    }
)

TRACK_COUNT_TARGETS: dict[str, tuple[int, int]] = {
    "Drum & Bass": (10, 14),
    "DnB": (10, 14),
    "Techno": (8, 12),
    "House": (8, 12),
    "Deep House": (8, 12),
    "UK Garage": (10, 13),
    "Jungle": (12, 16),
    # genre_traverse journey concepts (#82) legitimately need more tracks — 3+ per
    # chapter across up to 4 tempo-regime chapters plus ratio-bridge tracks. Keyed
    # lowercase to match the CLI's "traverse" genre label (see __main__'s genre arg),
    # unlike the Title Case keys above which mirror Rekordbox genre tags.
    "traverse": (8, 16),
    "_default": (8, 12),
}

_SHORTFALL_THRESHOLD = 4


def shortfall_warning(concept: MixConcept, genre: str) -> str | None:
    min_count, _ = TRACK_COUNT_TARGETS.get(genre, TRACK_COUNT_TARGETS["_default"])
    n = len(concept.track_ids)
    shortfall = min_count - n
    if shortfall > _SHORTFALL_THRESHOLD:
        return f"⚠️ {n} tracks found — needs {shortfall} more to fill a set. Crate dig to complete."
    return None
