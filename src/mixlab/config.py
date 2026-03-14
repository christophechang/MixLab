from __future__ import annotations

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
    "_default": (8, 12),
}
