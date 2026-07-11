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

# ---------------------------------------------------------------------------
# Mixed In Key energy scale (official, 1–10)
#
# MIK's published guidance — NOT the 1–8 scale the prompts historically claimed:
#   1–2  very chill, minimal, or atmospheric; may lack a strong beat
#   3–5  lower-to-medium intensity; lounge or smooth early-night groove
#   6–7  danceable and upbeat; 7 works for many club situations
#   8–10 high intensity; 9–10 are big club/festival moments
# Rule of thumb: mix tracks at the same level or one apart; bigger jumps only as a
# deliberate reset, contrast, or dramatic moment.
# ---------------------------------------------------------------------------

MIK_ENERGY_MIN = 1
MIK_ENERGY_MAX = 10

# (inclusive lo, inclusive hi, short label) — ordered, covering the full scale.
MIK_ENERGY_BANDS: tuple[tuple[int, int, str], ...] = (
    (1, 2, "very chill / atmospheric"),
    (3, 5, "lounge / smooth groove"),
    (6, 7, "danceable / upbeat"),
    (8, 10, "high intensity / peak"),
)


def energy_band_label(energy: int) -> str:
    """Official MIK band label for an energy level; clamps out-of-range values."""
    clamped = max(MIK_ENERGY_MIN, min(MIK_ENERGY_MAX, energy))
    for lo, hi, band_label in MIK_ENERGY_BANDS:
        if lo <= clamped <= hi:
            return band_label
    raise AssertionError("MIK_ENERGY_BANDS must cover the full 1-10 scale")


# One-paragraph prompt fragment shared by every LLM surface that shows energy:N/10
# tokens, so the model reads the numbers the way Mixed In Key defines them.
MIK_ENERGY_PROMPT_GUIDANCE = (
    "Energy values use Mixed In Key's official 1-10 scale: 1-2 very chill/atmospheric, "
    "3-5 lounge or smooth early-night groove, 6-7 danceable and upbeat, 8-10 high "
    "intensity (9-10 are big club/festival moments). Rule of thumb: consecutive tracks "
    "should sit at the same energy level or one apart (5→5 steady, 5→6 smooth lift); "
    "use a bigger jump only as a deliberate reset, contrast, or dramatic moment."
)


_SHORTFALL_THRESHOLD = 4


def shortfall_warning(concept: MixConcept, genre: str) -> str | None:
    min_count, _ = TRACK_COUNT_TARGETS.get(genre, TRACK_COUNT_TARGETS["_default"])
    n = len(concept.track_ids)
    shortfall = min_count - n
    if shortfall > _SHORTFALL_THRESHOLD:
        return f"⚠️ {n} tracks found — needs {shortfall} more to fill a set. Crate dig to complete."
    return None
