from __future__ import annotations

import pytest

from mixlab.config import (
    CUSTOM_GENRES,
    GENRE_MAP,
    MIK_ENERGY_BANDS,
    MIK_ENERGY_MAX,
    MIK_ENERGY_MIN,
    MIK_ENERGY_PROMPT_GUIDANCE,
    energy_band_label,
)


def test_energy_band_label_chill_band_returns_atmospheric() -> None:
    assert energy_band_label(1) == "very chill / atmospheric"
    assert energy_band_label(2) == "very chill / atmospheric"


def test_energy_band_label_lounge_band_returns_smooth_groove() -> None:
    assert energy_band_label(3) == "lounge / smooth groove"
    assert energy_band_label(5) == "lounge / smooth groove"


def test_energy_band_label_danceable_band_returns_upbeat() -> None:
    assert energy_band_label(6) == "danceable / upbeat"
    assert energy_band_label(7) == "danceable / upbeat"


def test_energy_band_label_high_band_returns_peak() -> None:
    assert energy_band_label(8) == "high intensity / peak"
    assert energy_band_label(10) == "high intensity / peak"


def test_energy_band_label_out_of_range_clamps_to_scale() -> None:
    assert energy_band_label(0) == energy_band_label(MIK_ENERGY_MIN)
    assert energy_band_label(11) == energy_band_label(MIK_ENERGY_MAX)


def test_mik_energy_bands_cover_full_scale_contiguously() -> None:
    expected_next = MIK_ENERGY_MIN
    for lo, hi, _ in MIK_ENERGY_BANDS:
        assert lo == expected_next
        assert hi >= lo
        expected_next = hi + 1
    assert expected_next == MIK_ENERGY_MAX + 1


@pytest.mark.parametrize("fragment", ["1-10", "6-7", "8-10", "one apart"])
def test_mik_energy_prompt_guidance_states_scale_and_rule(fragment: str) -> None:
    assert fragment in MIK_ENERGY_PROMPT_GUIDANCE


@pytest.mark.parametrize("tag", ["Hardcore", "Rave", "Hardcore, Jungle & Early House"])
def test_hardcore_tags_map_to_hardcore_pool_only(tag: str) -> None:
    owners = [label for label, tags in GENRE_MAP.items() if tag in tags]
    assert owners == ["hardcore"]


def test_hardcore_is_not_in_breakbeat_or_tempo_custom_pools() -> None:
    assert "hardcore" not in CUSTOM_GENRES["140"]["genres"]
    assert "hardcore" not in CUSTOM_GENRES["170"]["genres"]
    assert "hardcore" not in CUSTOM_GENRES["4x4"]["genres"]


def test_traverse_pool_covers_every_standard_genre() -> None:
    assert set(CUSTOM_GENRES["traverse"]["genres"]) == set(GENRE_MAP)
