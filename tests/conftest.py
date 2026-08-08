from __future__ import annotations

from pathlib import Path

import pytest

from mixlab.models import Track


@pytest.fixture(autouse=True)
def _isolate_report_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route HTML report output to a temp dir for every test (#45).

    run()/run_playlist_mode() write an HTML report unconditionally; without this
    guard any end-to-end test would leave artifacts in the repo's output/reports/.
    """
    monkeypatch.setenv("MIXLAB_REPORT_DIR", str(tmp_path / "reports"))


# ---------------------------------------------------------------------------
# Conjunction-mining pools. Plain helpers rather than pytest fixtures: callers
# compose them (``conj_pool() + [...]``) and pass them straight into pure
# functions. Shared here because both test_mining.py (the miner in isolation)
# and test_directions.py (the miner wired into the shared candidate field) need
# a pool where the planted conjunction provably fires.
# ---------------------------------------------------------------------------


def mining_track(
    track_id: str,
    *,
    label: str = "",
    year: int | None = None,
    tags: list[str] | None = None,
    energy: int | None = None,
    remixer: str = "",
    bpm: float = 174.0,
    key: str = "8A",
    date_added: str = "",
) -> Track:
    return Track(
        track_id=track_id,
        artist=f"A{track_id}",
        title=f"T{track_id}",
        bpm=bpm,
        camelot_key=key,
        genre="Drum & Bass",
        label=label,
        year=year,
        tags=tags or [],
        energy=energy,
        remixer=remixer,
        date_added=date_added,
    )


def conj_pool() -> list[Track]:
    """Hospital Records x Liquid inside an 80-track pool.

    NOTE: adjusted from the brief's original 20/20/20/20 split. That split gave
    label totals of 40 and tag totals of 40 with a 20-track overlap, i.e. lift =
    20*80/(40*40) = 1.0 — exactly chance, so scan_pairs' MIN_LIFT=1.3 gate (Task
    5) drops it and the planted conjunction never fires. Shrinking the
    Hospital-only / Liquid-only populations to 5 each keeps the 20-track overlap
    (support) the same but shrinks both marginal totals to 25, lifting lift to
    20*80/(25*25) = 2.56 while Jaccard-vs-parent stays 20/25 = 0.8 (under the 0.9
    subsumption threshold). year is dropped from the base groups entirely (kept
    only on the widen-test's added trailing group) so no era predicate arises to
    compete with or subsume the planted pair. bpm/camelot_key are left at the
    ``mining_track`` defaults everywhere so bpm_regime/key_hood cover the whole
    pool and get excluded by the 70% coverage cap, matching the original
    fixture's intent.

    Named builders over this pool: label_spotlight and fresh_crate propose;
    the other five have no material (no mood poles, no years, no artist with
    2-3 tracks, no energy, one tempo regime). date_added is set throughout so
    fresh_crate has its material.
    """
    pool: list[Track] = []
    for i in range(20):
        pool.append(
            mining_track(f"hl{i}", label="Hospital Records", tags=["Liquid"], date_added=f"2022-0{1 + i % 6}-10")
        )
    for i in range(5):
        pool.append(mining_track(f"ho{i}", label="Hospital Records", date_added=f"2021-0{1 + i % 6}-10"))
    for i in range(5):
        pool.append(mining_track(f"lo{i}", tags=["Liquid"], date_added=f"2021-0{1 + i % 6}-10"))
    for i in range(20):
        pool.append(mining_track(f"p{i}", label="Shogun Audio", tags=["Deep"], date_added=f"2023-0{1 + i % 6}-10"))
    for i in range(30):
        pool.append(mining_track(f"n{i}", date_added=f"2020-0{1 + i % 6}-10"))
    return pool


def mech_conj_pool() -> list[Track]:
    """tag "liquid" x bpm_regime "130": exercises mine_pool's single-namable
    branch (title has no mechanical value or "×", mood pairs a display noun
    with the namable kind, brief cites the mechanical value only in prose).

    Groups (N=50 total):
      tb (20): tags=["Liquid"], bpm=130.0 -- the planted overlap (support).
      to (5):  tags=["Liquid"], bpm=100.0 -- tag-only, off the 130 peak.
      bo (5):  no tags,          bpm=130.0 -- bpm-only, in the 130 peak.
      f  (20): no tags, bpm = 140, 150, ..., 330 (one track per value, spaced
        10 BPM apart so no other bin's +-3 BPM smoothing window in
        directions._tempo_regimes can accumulate 5+ mass and form a
        competing peak).

    tag "liquid" total = tb+to = 25; bpm_regime "130" total = tb+bo = 25.
    (`to`'s 5 tracks at 100 BPM are tight enough to clear _tempo_regimes'
    internal 5-track peak floor and form their own regime, but
    extract_predicates' outer per-kind gate is >=8, so that 5-track "100"
    predicate never actually gets emitted -- no competing pair.)

    support (tb) = 20; lift = 20*50 / (25*25) = 1.6 >= MIN_LIFT (1.3).
    Jaccard(members=20, tag_total=25) = Jaccard(members=20, bpm_total=25)
    = 0.8, under the 0.9 subsumption threshold, so the pair survives intact.
    key_hood never appears: every track (including filler) keeps the
    ``mining_track`` default camelot_key "8A", so key_hood "8A" would cover
    100% of the pool and gets dropped by the 70% coverage cap, same trick as
    ``conj_pool``.
    """
    pool: list[Track] = []
    for i in range(20):
        pool.append(mining_track(f"tb{i}", tags=["Liquid"], bpm=130.0, date_added=f"2022-0{1 + i % 6}-10"))
    for i in range(5):
        pool.append(mining_track(f"to{i}", tags=["Liquid"], bpm=100.0, date_added=f"2021-0{1 + i % 6}-10"))
    for i in range(5):
        pool.append(mining_track(f"bo{i}", bpm=130.0, date_added=f"2021-0{1 + i % 6}-10"))
    for i in range(20):
        pool.append(mining_track(f"f{i}", bpm=140.0 + 10 * i, date_added=f"2020-0{1 + i % 6}-10"))
    return pool
