from __future__ import annotations

from mixlab.booth_sheet import BoothStep, build_booth_sheet
from mixlab.models import MixConcept, MixPoints, Track, Transition


def _mix_points(
    *,
    mix_in: float = 8.0,
    mix_out: float | None = None,
    intro: float | None = None,
    outro: float | None = None,
) -> MixPoints:
    return MixPoints(mix_in_secs=mix_in, mix_out_secs=mix_out, intro_bars=intro, outro_bars=outro)


def _track(
    tid: str,
    *,
    artist: str = "Artist",
    title: str = "Title",
    bpm: float = 128.0,
    key: str = "8A",
    energy: int | None = 5,
    duration: int | None = 300,
    mix_points: MixPoints | None = None,
) -> Track:
    return Track(
        track_id=tid,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=key,
        genre="House",
        energy=energy,
        duration_secs=duration,
        mix_points=mix_points,
    )


def _concept(track_ids: list[str], transitions: list[Transition] | None = None, **kwargs: object) -> MixConcept:
    return MixConcept(
        title="Test Concept",
        mood="test mood",
        track_ids=track_ids,
        transitions=transitions or [],
        **kwargs,  # type: ignore[arg-type]
    )


def _one(concept: MixConcept, tracks: list[Track]) -> BoothStep:
    steps = build_booth_sheet(concept, {t.track_id: t for t in tracks})
    assert len(steps) == 1
    return steps[0]


# ---------------------------------------------------------------------------
# Happy path clock math / bar bands
# ---------------------------------------------------------------------------


def test_build_booth_sheet_mix_out_cue_known_opens_blend_at_formatted_clock() -> None:
    a = _track("a", title="Outbound", mix_points=_mix_points(mix_out=343.0, outro=17.0))
    b = _track("b", title="Inbound", mix_points=_mix_points(intro=8.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert "Open the blend at 5:43" in step.plan
    assert "Take a comfortable 16 bars." in step.plan
    assert step.open_at_secs == 343.0
    assert step.outro_bars == 17.0
    assert step.intro_bars == 8.0
    assert step.fallback is None  # outro 17 < 24
    assert len(step.chips) == 4
    assert all(not is_warn for _text, is_warn in step.chips)
    assert step.chips[2] == ("17 bars out", False)
    assert step.chips[3] == ("8-bar intro on incoming", False)


def test_build_booth_sheet_outro_ge_24_sets_fallback_text() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=100.0, outro=30.0))
    b = _track("b", mix_points=_mix_points(intro=8.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.fallback is not None
    assert "delay the swap to the last 16 bars" in step.fallback


def test_build_booth_sheet_outro_lt_16_plan_says_tight_window() -> None:
    a = _track("a", title="Outbound", mix_points=_mix_points(mix_out=50.0, outro=10.0))
    b = _track("b", title="Inbound", mix_points=_mix_points(intro=4.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert "Tight window — have Inbound cued and ready." in step.plan


def test_build_booth_sheet_outro_ge_32_plan_says_start_early() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=50.0, outro=40.0))
    b = _track("b", title="Inbound", mix_points=_mix_points(intro=4.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert "Start Inbound early — you have space to correct before anyone hears it." in step.plan


# ---------------------------------------------------------------------------
# Tempo relation / pitch percent
# ---------------------------------------------------------------------------


def test_build_booth_sheet_halftime_relation_pitch_pct_zero_and_chip_labelled() -> None:
    a = _track("a", bpm=174.0, mix_points=_mix_points(mix_out=200.0, outro=20.0))
    b = _track("b", bpm=87.0, mix_points=_mix_points(intro=8.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.pitch_pct == 0.0
    tempo_text, is_warn = step.chips[0]
    assert tempo_text == "174→87 halftime · 0%"
    assert is_warn is False


def test_build_booth_sheet_incompatible_tempo_pitch_pct_none_and_bpm_chip() -> None:
    a = _track("a", bpm=105.0)
    b = _track("b", bpm=120.0)
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.pitch_pct is None
    assert step.chips[0] == ("105→120 BPM", False)


def test_build_booth_sheet_key_chip_appends_energy_lift_label() -> None:
    a = _track("a", bpm=128.0, key="8A")
    b = _track("b", bpm=128.0, key="9A")
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.chips[1] == ("8A→9A energy lift", False)


# ---------------------------------------------------------------------------
# Tier rules
# ---------------------------------------------------------------------------


def test_build_booth_sheet_high_blend_score_tier_relaxed() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=100.0, outro=20.0))
    b = _track("b", mix_points=_mix_points(intro=8.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.tier == "relaxed"


def test_build_booth_sheet_mid_blend_score_tier_tight() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=100.0, outro=6.0))
    b = _track("b", mix_points=_mix_points(intro=10.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.tier == "tight"


def test_build_booth_sheet_low_blend_score_tier_hard() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=100.0, outro=2.0))
    b = _track("b", mix_points=_mix_points(intro=10.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.tier == "hard"


def test_build_booth_sheet_risky_pair_with_no_cue_data_tier_hard() -> None:
    a = _track("a", mix_points=None)
    b = _track("b", mix_points=None)
    concept = _concept(["a", "b"], [Transition(from_id="a", to_id="b", is_risky=True, risk_type="chapter_pivot")])
    step = _one(concept, [a, b])

    assert step.tier == "hard"


def test_build_booth_sheet_long_outro_no_blend_data_tier_relaxed() -> None:
    a = _track("a", mix_points=_mix_points(outro=20.0))
    b = _track("b", mix_points=None)
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.tier == "relaxed"


def test_build_booth_sheet_tempo_locked_partial_data_tier_relaxed() -> None:
    a = _track("a", bpm=128.0, mix_points=None)
    b = _track("b", bpm=128.0, mix_points=_mix_points())
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.pitch_pct == 0.0
    assert step.tier == "relaxed"


def test_build_booth_sheet_no_special_case_tier_tight() -> None:
    a = _track("a", bpm=128.0, mix_points=None)
    b = _track("b", bpm=128.0, mix_points=None)
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.tier == "tight"


# ---------------------------------------------------------------------------
# Intro-from-zero convention
# ---------------------------------------------------------------------------


def test_build_booth_sheet_intro_zero_reads_as_mixes_from_top_never_cold_drop() -> None:
    a = _track("a", mix_points=None)
    b = _track("b", title="Inbound", mix_points=_mix_points(intro=0.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert ("mixes from the top", False) in step.chips
    assert "mixes from the top" in step.plan
    assert "Inbound mixes from the top — ride the blend over its opening bars." in step.plan
    assert "cold drop" not in step.plan.lower()
    assert all("cold drop" not in text.lower() for text, _warn in step.chips)


# ---------------------------------------------------------------------------
# Scout variants
# ---------------------------------------------------------------------------


def test_build_booth_sheet_scout_both_uncued() -> None:
    a = _track("a", mix_points=None)
    b = _track("b", mix_points=None)
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.scout == "Both tracks uncued — highest-priority prep before this set."


def test_build_booth_sheet_scout_incoming_uncued_only() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=100.0, outro=20.0))
    b = _track("b", title="Inbound", mix_points=None)
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.scout == "Inbound has no cue points yet — scout its entry before the set."


def test_build_booth_sheet_scout_outgoing_uncued_only() -> None:
    a = _track("a", title="Outbound", mix_points=None)
    b = _track("b", mix_points=_mix_points(intro=8.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.scout == "Outbound has no mix-out cue — the exit point is by ear."


def test_build_booth_sheet_scout_outgoing_partially_prepped_missing_mix_out() -> None:
    a = _track("a", title="Outbound", mix_points=_mix_points(outro=20.0))  # no mix_out_secs
    b = _track("b", mix_points=_mix_points(intro=8.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.scout == "Outbound has no mix-out cue — the exit point is by ear."


def test_build_booth_sheet_scout_fully_prepped_pair_is_none() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=100.0, outro=20.0))
    b = _track("b", mix_points=_mix_points(intro=8.0))
    step = _one(_concept(["a", "b"]), [a, b])

    assert step.scout is None


# ---------------------------------------------------------------------------
# Missing track ids / resolvability
# ---------------------------------------------------------------------------


def test_build_booth_sheet_missing_track_id_skips_pair_keeps_original_index() -> None:
    a = _track("a")
    c = _track("c")
    d = _track("d")
    concept = _concept(["a", "b", "c", "d"])  # "b" absent from tracks_by_id
    steps = build_booth_sheet(concept, {"a": a, "c": c, "d": d})

    assert len(steps) == 1
    assert steps[0].index == 3
    assert steps[0].from_id == "c"
    assert steps[0].to_id == "d"


def test_build_booth_sheet_fewer_than_two_resolvable_tracks_returns_empty() -> None:
    concept = _concept(["x", "y"])
    steps = build_booth_sheet(concept, {})

    assert steps == []


# ---------------------------------------------------------------------------
# Risk annotation chip
# ---------------------------------------------------------------------------


def test_build_booth_sheet_risk_annotation_adds_warn_chip_with_risk_type() -> None:
    a = _track("a", mix_points=_mix_points(mix_out=100.0, outro=20.0))
    b = _track("b", mix_points=_mix_points(intro=8.0))
    concept = _concept(["a", "b"], [Transition(from_id="a", to_id="b", risk_type="peak_impact")])
    step = _one(concept, [a, b])

    assert ("peak_impact", True) in step.chips


def test_build_booth_sheet_no_cue_data_either_side_adds_warn_chip() -> None:
    a = _track("a", mix_points=None)
    b = _track("b", mix_points=None)
    step = _one(_concept(["a", "b"]), [a, b])

    assert ("no cue data either side", True) in step.chips


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_booth_sheet_repeated_calls_are_equal() -> None:
    a = _track("a", bpm=174.0, mix_points=_mix_points(mix_out=200.0, outro=20.0))
    b = _track("b", bpm=87.0, mix_points=_mix_points(intro=8.0))
    concept = _concept(["a", "b"], [Transition(from_id="a", to_id="b", risk_type="peak_impact")])
    tracks_by_id = {"a": a, "b": b}

    first = build_booth_sheet(concept, tracks_by_id)
    second = build_booth_sheet(concept, tracks_by_id)

    assert first == second
