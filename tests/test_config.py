from __future__ import annotations

import pytest

from mixlab.config import (
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
