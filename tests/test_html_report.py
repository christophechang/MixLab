from __future__ import annotations

from mixlab.html_report import render_html_report, report_filename
from mixlab.models import MixConcept, MixPoints, Track, Transition


def _track(
    tid: str,
    *,
    artist: str = "Artist",
    title: str = "Title",
    bpm: float = 174.0,
    key: str = "8A",
    energy: int | None = 5,
    duration: int | None = 300,
    intro: float | None = None,
    outro: float | None = None,
    mix_out: float | None = None,
) -> Track:
    mix_points = None
    if intro is not None or outro is not None or mix_out is not None:
        mix_points = MixPoints(mix_in_secs=8.0, mix_out_secs=mix_out, intro_bars=intro, outro_bars=outro)
    return Track(
        track_id=tid,
        artist=artist,
        title=title,
        bpm=bpm,
        camelot_key=key,
        genre="Drum & Bass",
        energy=energy,
        duration_secs=duration,
        mix_points=mix_points,
    )


def _concept(track_ids: list[str], **kwargs: object) -> MixConcept:
    return MixConcept(title="Dark Rollers", mood="heavy and hypnotic", track_ids=track_ids, **kwargs)  # type: ignore[arg-type]


def _tracks_by_id(tracks: list[Track]) -> dict[str, Track]:
    return {t.track_id: t for t in tracks}


# ---------------------------------------------------------------------------
# Determinism & self-containment
# ---------------------------------------------------------------------------


def test_render_html_report_deterministic_for_identical_inputs() -> None:
    tracks = [_track(str(i), energy=3 + i) for i in range(4)]
    concept = _concept([t.track_id for t in tracks], arc_type="progressive-build")
    tbi = _tracks_by_id(tracks)
    a = render_html_report(
        [concept], tbi, report_text="body", report_context="ctx", validation_warnings=[], generated_at="2026-07-07"
    )
    b = render_html_report(
        [concept], tbi, report_text="body", report_context="ctx", validation_warnings=[], generated_at="2026-07-07"
    )
    assert a == b


def test_render_html_report_contains_no_external_urls() -> None:
    tracks = [_track(str(i), energy=3 + i, intro=8, outro=16) for i in range(4)]
    concept = _concept([t.track_id for t in tracks], arc_type="wave", name_reason="Relentless.")
    html = render_html_report(
        [concept],
        _tracks_by_id(tracks),
        report_text="section one\n\n---\n\nrun notes",
        report_context="Report context: DnB (unplayed tracks)",
        validation_warnings=["watch the jump"],
        counts={"drum_and_bass": (200, 80)},
        generated_at="2026-07-07",
    )
    assert "http://" not in html
    assert "https://" not in html


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


def test_render_html_report_escapes_dynamic_strings() -> None:
    evil = _track("1", artist="A & B", title="<script>alert(1)</script>", energy=5)
    concept = _concept(["1"])
    html = render_html_report(
        [concept], _tracks_by_id([evil]), report_text="", report_context="", validation_warnings=[]
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "A &amp; B" in html


# ---------------------------------------------------------------------------
# Concept card content
# ---------------------------------------------------------------------------


def test_render_html_report_shows_card_title_arc_and_track_fields() -> None:
    t = _track("1", artist="Goldie", title="Inner City Life", bpm=174.0, key="8A", energy=7, duration=305)
    concept = _concept(["1"], arc_type="progressive-build")
    html = render_html_report([concept], _tracks_by_id([t]), report_text="", report_context="", validation_warnings=[])
    assert "Dark Rollers" in html
    assert "progressive-build" in html
    assert "Goldie — Inner City Life" in html
    assert "8A" in html
    assert "174" in html
    assert "5:05" in html  # duration m:ss
    assert ">7<" in html  # energy cell


def test_render_html_report_shows_dirtype_badge_when_direction_type_set() -> None:
    t = _track("1", energy=5)
    concept = _concept(["1"], arc_type="wave", direction_type="genre_traverse")
    html = render_html_report([concept], _tracks_by_id([t]), report_text="", report_context="", validation_warnings=[])
    assert 'class="arc dirtype"' in html
    assert ">genre_traverse<" in html


def test_render_html_report_omits_dirtype_badge_when_direction_type_empty() -> None:
    t = _track("1", energy=5)
    concept = _concept(["1"], arc_type="wave", direction_type="")
    html = render_html_report([concept], _tracks_by_id([t]), report_text="", report_context="", validation_warnings=[])
    assert 'class="arc dirtype"' not in html


def test_render_html_report_escapes_dirtype_badge() -> None:
    t = _track("1", energy=5)
    concept = _concept(["1"], direction_type="<script>alert(2)</script>")
    html = render_html_report([concept], _tracks_by_id([t]), report_text="", report_context="", validation_warnings=[])
    assert "<script>alert(2)</script>" not in html
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in html


def test_render_html_report_shows_intro_outro_bars_when_present() -> None:
    t = _track("1", intro=8, outro=16, energy=5)
    html = render_html_report(
        [_concept(["1"])], _tracks_by_id([t]), report_text="", report_context="", validation_warnings=[]
    )
    assert ">8<" in html
    assert ">16<" in html


def test_render_html_report_shows_dash_when_mix_points_absent() -> None:
    t = _track("1", intro=None, outro=None, duration=None)
    html = render_html_report(
        [_concept(["1"])], _tracks_by_id([t]), report_text="", report_context="", validation_warnings=[]
    )
    assert "—" in html


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------


def test_render_html_report_sparkline_present_with_three_energies() -> None:
    tracks = [_track(str(i), energy=3 + i) for i in range(3)]
    concept = _concept([t.track_id for t in tracks])
    html = render_html_report(
        [concept], _tracks_by_id(tracks), report_text="", report_context="", validation_warnings=[]
    )
    assert "<polyline" in html
    assert 'class="spark"' in html


def test_render_html_report_sparkline_absent_with_two_energies() -> None:
    tracks = [_track("1", energy=4), _track("2", energy=6), _track("3", energy=None)]
    concept = _concept([t.track_id for t in tracks])
    html = render_html_report(
        [concept], _tracks_by_id(tracks), report_text="", report_context="", validation_warnings=[]
    )
    assert "<polyline" not in html


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def test_render_html_report_halftime_pair_renders_mechanism_and_band() -> None:
    a = _track("1", bpm=174.0, key="8A", energy=5)
    b = _track("2", bpm=87.0, key="8A", energy=5)  # halftime of 174
    concept = _concept(["1", "2"])
    html = render_html_report(
        [concept], _tracks_by_id([a, b]), report_text="", report_context="", validation_warnings=[]
    )
    assert "halftime lock 174→87" in html
    assert "t-good" in html or "t-warn" in html or "t-bad" in html
    assert "1→2" in html


def test_render_html_report_blend_label_appears_with_mix_points_both_sides() -> None:
    a = _track("1", bpm=174.0, key="8A", energy=5, intro=8, outro=16)
    b = _track("2", bpm=174.0, key="8A", energy=5, intro=8, outro=16)
    concept = _concept(["1", "2"])
    html = render_html_report(
        [concept], _tracks_by_id([a, b]), report_text="", report_context="", validation_warnings=[]
    )
    assert 'class="blend"' in html
    assert "bars out" in html


def test_render_html_report_shows_risk_annotation() -> None:
    a = _track("1", bpm=174.0, key="8A", energy=5)
    b = _track("2", bpm=174.0, key="8A", energy=5)
    concept = _concept(["1", "2"], transitions=[Transition(from_id="1", to_id="2", risk_type="peak_impact")])
    html = render_html_report(
        [concept], _tracks_by_id([a, b]), report_text="", report_context="", validation_warnings=[]
    )
    assert "peak_impact" in html


# ---------------------------------------------------------------------------
# Prose pairing
# ---------------------------------------------------------------------------


def test_render_html_report_prose_pairs_per_card_with_trailing_notes() -> None:
    tracks = [_track(str(i)) for i in range(2)]
    concepts = [_concept(["0"]), _concept(["1"])]
    report = "concept one prose\n\n---\n\nconcept two prose\n\n---\n\nshortfall note"
    html = render_html_report(
        concepts, _tracks_by_id(tracks), report_text=report, report_context="", validation_warnings=[]
    )
    assert "concept one prose" in html
    assert "concept two prose" in html
    assert "shortfall note" in html
    assert "Run notes" in html
    assert html.count("Full report") == 2


def test_render_html_report_prose_mismatch_falls_back_to_single_section() -> None:
    tracks = [_track(str(i)) for i in range(3)]
    concepts = [_concept(["0"]), _concept(["1"]), _concept(["2"])]
    # Only two sections for three concepts -> mismatch -> full text in one trailing block.
    report = "part A\n\n---\n\npart B"
    html = render_html_report(
        concepts, _tracks_by_id(tracks), report_text=report, report_context="", validation_warnings=[]
    )
    assert "part A" in html
    assert "part B" in html
    assert "Full report" not in html  # no per-card prose
    assert "Run notes" in html


def test_render_html_report_run_notes_strip_duplicated_validation_block() -> None:
    """The textual Validation Notes block is dropped from run notes (live-run finding).

    Discord's report text embeds a "⚠ **Validation Notes**" bullet block, but the HTML
    report already renders validation_warnings as its own section — keeping the textual
    copy duplicated every warning in Run notes.
    """
    tracks = [_track("0")]
    report = "concept prose\n\n---\n\n⚠ **Validation Notes**\n- warning one\n- warning two\n\nactual shortfall note"
    html = render_html_report(
        [_concept(["0"])],
        _tracks_by_id(tracks),
        report_text=report,
        report_context="",
        validation_warnings=["warning one", "warning two"],
    )
    assert "Run notes" in html
    assert "actual shortfall note" in html
    # Warnings appear once (the dedicated section), not again inside run notes.
    assert html.count("warning one") == 1
    assert "Validation Notes" not in html


def test_render_html_report_empty_report_text_renders_no_prose() -> None:
    tracks = [_track("0")]
    html = render_html_report(
        [_concept(["0"])], _tracks_by_id(tracks), report_text="", report_context="", validation_warnings=[]
    )
    assert "Full report" not in html
    assert "Run notes" not in html


# ---------------------------------------------------------------------------
# Validation & snapshot sections
# ---------------------------------------------------------------------------


def test_render_html_report_renders_validation_warnings() -> None:
    html = render_html_report(
        [_concept(["0"])],
        _tracks_by_id([_track("0")]),
        report_text="",
        report_context="",
        validation_warnings=["BPM jump too wide", "played track slipped in"],
    )
    assert "Validation notes" in html
    assert "BPM jump too wide" in html


def test_render_html_report_omits_validation_when_empty() -> None:
    html = render_html_report(
        [_concept(["0"])], _tracks_by_id([_track("0")]), report_text="", report_context="", validation_warnings=[]
    )
    assert "Validation notes" not in html


def test_render_html_report_renders_snapshot_when_counts_given() -> None:
    html = render_html_report(
        [_concept(["0"])],
        _tracks_by_id([_track("0")]),
        report_text="",
        report_context="",
        validation_warnings=[],
        counts={"drum_and_bass": (200, 80), "house": (150, 50)},
    )
    assert "Crate snapshot" in html
    assert "80 / 200" in html


def test_render_html_report_omits_snapshot_when_counts_absent() -> None:
    html = render_html_report(
        [_concept(["0"])], _tracks_by_id([_track("0")]), report_text="", report_context="", validation_warnings=[]
    )
    assert "Crate snapshot" not in html


# ---------------------------------------------------------------------------
# Booth sheet (#69)
# ---------------------------------------------------------------------------


def test_render_html_report_cued_pair_shows_booth_sheet_relaxed_step() -> None:
    a = _track("1", title="Outbound", mix_out=100.0, outro=20.0)
    b = _track("2", title="Inbound", intro=8.0)
    concept = _concept(["1", "2"])
    html = render_html_report(
        [concept], _tracks_by_id([a, b]), report_text="", report_context="", validation_warnings=[]
    )
    assert 'class="booth"' in html
    assert "Booth sheet" in html
    assert "Open the blend at" in html
    assert '<li class="step">' in html


def test_render_html_report_tight_and_hard_steps_get_tier_classes() -> None:
    a1 = _track("1", title="A1", mix_out=100.0, outro=6.0)
    b1 = _track("2", title="B1", intro=10.0)
    tight_concept = _concept(["1", "2"])

    a2 = _track("3", title="A2", mix_out=100.0, outro=2.0)
    b2 = _track("4", title="B2", intro=10.0)
    hard_concept = _concept(["3", "4"])

    tbi = _tracks_by_id([a1, b1, a2, b2])
    html = render_html_report(
        [tight_concept, hard_concept], tbi, report_text="", report_context="", validation_warnings=[]
    )
    assert '<li class="step tight">' in html
    assert '<li class="step hard">' in html


def test_render_html_report_risk_annotation_renders_warn_chip_in_booth_sheet() -> None:
    a = _track("1", title="Outbound", mix_out=100.0, outro=20.0)
    b = _track("2", title="Inbound", intro=8.0)
    concept = _concept(["1", "2"], transitions=[Transition(from_id="1", to_id="2", risk_type="peak_impact")])
    html = render_html_report(
        [concept], _tracks_by_id([a, b]), report_text="", report_context="", validation_warnings=[]
    )
    assert '<span class="chip warn">peak_impact</span>' in html


def test_render_html_report_single_track_concept_omits_booth_sheet() -> None:
    t = _track("1")
    html = render_html_report(
        [_concept(["1"])], _tracks_by_id([t]), report_text="", report_context="", validation_warnings=[]
    )
    assert "Booth sheet" not in html
    assert 'class="booth"' not in html


def test_render_html_report_booth_sheet_escapes_track_titles() -> None:
    evil = _track("1", artist="A & B", title="<script>alert(1)</script>")
    safe = _track("2", title="Safe")
    concept = _concept(["1", "2"])
    html = render_html_report(
        [concept], _tracks_by_id([evil, safe]), report_text="", report_context="", validation_warnings=[]
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_html_report_uncued_pair_renders_step_with_scout_note() -> None:
    a = _track("1", title="Outbound")
    b = _track("2", title="Inbound")
    concept = _concept(["1", "2"])
    html = render_html_report(
        [concept], _tracks_by_id([a, b]), report_text="", report_context="", validation_warnings=[]
    )
    assert '<span class="chip warn">no cue data either side</span>' in html
    assert 'class="scout"' in html
    assert "Both tracks uncued" in html


# ---------------------------------------------------------------------------
# report_filename
# ---------------------------------------------------------------------------


def test_report_filename_genre() -> None:
    assert report_filename("house", None, "2026-07-07") == "mixlab-house-2026-07-07.html"


def test_report_filename_playlist_slugifies_spaces_and_slashes() -> None:
    assert report_filename(None, "Monday / Night", "2026-07-07") == "mixlab-monday-night-2026-07-07.html"


def test_report_filename_playlist_takes_precedence_over_genre() -> None:
    assert report_filename("house", "Monday Night", "2026-07-07") == "mixlab-monday-night-2026-07-07.html"


def test_report_filename_both_none_uses_run() -> None:
    assert report_filename(None, None, "2026-07-07") == "mixlab-run-2026-07-07.html"
