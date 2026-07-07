"""Deterministic per-transition "booth sheet" (#68).

Turns a :class:`~mixlab.models.MixConcept` into a track-by-track execution plan a
DJ can read off the booth screen mid-set: where to open the blend, how tight the
window is, what to watch for, and what to scout ahead of time. Pure and
side-effect free — no LLM calls, no I/O, no HTML — built entirely from
:mod:`mixlab.transitions` primitives (``tempo_relation``, ``blend_headroom``,
``score_transition``/``relation_label``) so the mechanics stay byte-identical to
what the HTML report already shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mixlab import transitions
from mixlab.models import MixConcept, Track

Tier = Literal["relaxed", "tight", "hard"]

# Ratio each tempo_relation maps to, mirroring transitions._RATIOS. Duplicated here
# (rather than imported) because transitions.py only exposes the relation *name*,
# not the ratio — this module needs the ratio to project the incoming BPM into the
# outgoing track's time-base for the pitch-percent readout.
_RATIO_BY_RELATION: dict[str, float] = {
    "straight": 1.0,
    "halftime": 0.5,
    "double": 2.0,
    "three_four": 0.75,
    "four_three": 4.0 / 3.0,
}

# Short display names for the tempo chip. Distinct from transitions.relation_label's
# "halftime lock" / "double-time lock" / "3:4 shuffle" / "4:3 push" mechanism labels
# (those are prose fragments meant to be joined with commas); the booth chip needs a
# single short word/token to sit before "· <pct>%".
_RELATION_CHIP_NAME: dict[str, str] = {
    "halftime": "halftime",
    "double": "double",
    "three_four": "3:4",
    "four_three": "4:3",
}


@dataclass(frozen=True)
class BoothStep:
    index: int  # 1-based pair number (pair i -> i+1), original concept position
    from_id: str
    to_id: str
    from_label: str  # "Artist — Title"
    to_label: str
    open_at_secs: float | None  # outgoing track's mix_points.mix_out_secs when known
    outro_bars: float | None  # outgoing side
    intro_bars: float | None  # incoming side
    pitch_pct: float | None  # signed, rounded to 1 decimal; None when incompatible
    tier: Tier
    chips: tuple[tuple[str, bool], ...]  # (text, is_warn) in display order
    plan: str
    fallback: str | None
    scout: str | None


def _fmt_bpm(bpm: float) -> str:
    """Trim a trailing ``.0`` while preserving real decimals (124.0 -> "124")."""
    return f"{bpm:g}"


def _fmt_pitch_pct(pct: float) -> str:
    rounded = round(pct, 1)
    if rounded == 0:
        return "0%"
    sign = "−" if rounded < 0 else "+"
    return f"{sign}{abs(rounded):.1f}%"


def _tempo_chip(a: Track, b: Track) -> tuple[str, float | None]:
    """Build the tempo chip text and the derived pitch percentage.

    Returns ``(chip_text, pitch_pct)``; ``pitch_pct`` is ``None`` when the tempo
    relation is incompatible.
    """
    relation, _stretch = transitions.tempo_relation(a.bpm, b.bpm)
    base = f"{_fmt_bpm(a.bpm)}→{_fmt_bpm(b.bpm)}"
    if relation == "incompatible":
        return f"{base} BPM", None

    ratio = _RATIO_BY_RELATION[relation]
    effective = b.bpm / ratio
    pitch_pct = round((effective - a.bpm) / a.bpm * 100, 1)

    name = _RELATION_CHIP_NAME.get(relation, "")
    suffix = f" {name}" if name else ""
    return f"{base}{suffix} · {_fmt_pitch_pct(pitch_pct)}", pitch_pct


def _key_chip(a: Track, b: Track, edge: transitions.TransitionEdge) -> str:
    text = f"{a.camelot_key}→{b.camelot_key}"
    if edge.is_energy_lift:
        text += " energy lift"
    return text


def _bars_chips(outro_bars: float | None, intro_bars: float | None) -> list[tuple[str, bool]]:
    chips: list[tuple[str, bool]] = []
    if outro_bars is not None:
        chips.append((f"{outro_bars:.0f} bars out", False))
    if intro_bars is not None:
        if intro_bars > 0:
            chips.append((f"{intro_bars:.0f}-bar intro on incoming", False))
        else:
            chips.append(("mixes from the top", False))
    return chips


def _warn_chips(
    *,
    outgoing_missing: bool,
    incoming_missing: bool,
    annotation_risk_type: str,
) -> list[tuple[str, bool]]:
    chips: list[tuple[str, bool]] = []
    if outgoing_missing and incoming_missing:
        chips.append(("no cue data either side", True))
    elif incoming_missing:
        chips.append(("no intro data on incoming", True))
    elif outgoing_missing:
        chips.append(("no outro data on outgoing", True))
    if annotation_risk_type:
        chips.append((annotation_risk_type, True))
    return chips


def _fmt_clock(secs: float) -> str:
    total = int(round(secs))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}:{seconds:02d}"


def _tier(
    a: Track,
    b: Track,
    *,
    pitch_pct: float | None,
    is_risky_annotated: bool,
    outgoing_missing: bool,
    incoming_missing: bool,
) -> Tier:
    headroom, _label = transitions.blend_headroom(a, b)
    if headroom is not None:
        if headroom >= 0.7:
            return "relaxed"
        if headroom >= 0.3:
            return "tight"
        return "hard"

    if is_risky_annotated and outgoing_missing and incoming_missing:
        return "hard"

    outro_bars = a.mix_points.outro_bars if a.mix_points is not None else None
    long_outro = outro_bars is not None and outro_bars >= 16
    tempo_locked = (
        pitch_pct is not None and abs(pitch_pct) < 0.5 and (a.mix_points is not None or b.mix_points is not None)
    )
    if long_outro or tempo_locked:
        return "relaxed"

    return "tight"


def _plan(
    a: Track,
    b: Track,
    *,
    open_at_secs: float | None,
    outro_bars: float | None,
    intro_bars: float | None,
    pitch_pct: float | None,
    tempo_chip_text: str,
    outgoing_missing: bool,
    incoming_missing: bool,
    risk_type: str,
) -> str:
    to_title = b.title
    from_title = a.title

    if open_at_secs is not None and outro_bars is not None:
        clock = _fmt_clock(open_at_secs)
        plan = f"Open the blend at {clock} — {from_title}'s mix-out cue, {outro_bars:.0f} bars from the end."
        if outro_bars >= 32:
            plan += f" Start {to_title} early — you have space to correct before anyone hears it."
        elif outro_bars >= 16:
            plan += " Take a comfortable 16 bars."
        else:
            plan += f" Tight window — have {to_title} cued and ready."
    elif outgoing_missing and incoming_missing and risk_type:
        plan = (
            f"Flagged {risk_type} moment, flying on ears only — commit hard on the swap; hesitation reads as a mistake."
        )
    elif pitch_pct is not None and abs(pitch_pct) < 0.5 and (a.mix_points is not None or b.mix_points is not None):
        # Only point at the mix-in cue if the incoming track actually has one.
        entry = "from its mix-in cue" if b.mix_points is not None else "on the phrase"
        plan = f"Tempos already locked — start {to_title} {entry} and let the key move do the work."
    else:
        plan = f"Blend on ears: {tempo_chip_text} move, no full cue picture — plan the swap around {to_title}'s first phrase."

    if intro_bars is not None and intro_bars == 0:
        plan += f" {to_title} mixes from the top — ride the blend over its opening bars."

    return plan


def _fallback(outro_bars: float | None) -> str | None:
    if outro_bars is not None and outro_bars >= 24:
        return (
            "If it isn't landing, delay the swap to the last 16 bars and cut on the phrase — "
            "the outro is long enough either way."
        )
    return None


def _scout(a: Track, b: Track) -> str | None:
    if a.mix_points is None and b.mix_points is None:
        return "Both tracks uncued — highest-priority prep before this set."
    if b.mix_points is None:
        return f"{b.title} has no cue points yet — scout its entry before the set."
    if a.mix_points is None:
        return f"{a.title} has no mix-out cue — the exit point is by ear."
    if a.mix_points.mix_out_secs is None:
        return f"{a.title} has no mix-out cue — the exit point is by ear."
    return None


def build_booth_sheet(concept: MixConcept, tracks_by_id: dict[str, Track]) -> list[BoothStep]:
    """Build one :class:`BoothStep` per resolvable consecutive track pair.

    Pairs where either track id is missing from ``tracks_by_id`` are skipped
    entirely (not bridged across the gap) — ``index`` always reflects the pair's
    original 1-based position among the concept's consecutive pairs, so skipped
    pairs leave a gap in the sequence rather than causing a renumbering.
    """
    annotations = {(tr.from_id, tr.to_id): tr for tr in concept.transitions}
    steps: list[BoothStep] = []

    track_ids = concept.track_ids
    for i in range(len(track_ids) - 1):
        a_id, b_id = track_ids[i], track_ids[i + 1]
        a = tracks_by_id.get(a_id)
        b = tracks_by_id.get(b_id)
        if a is None or b is None:
            continue

        annotation = annotations.get((a.track_id, b.track_id))
        risk_type = annotation.risk_type if annotation is not None else ""
        is_risky_annotated = annotation is not None and annotation.is_risky

        outgoing_missing = a.mix_points is None or a.mix_points.outro_bars is None
        incoming_missing = b.mix_points is None or b.mix_points.intro_bars is None

        open_at_secs = a.mix_points.mix_out_secs if a.mix_points is not None else None
        outro_bars = a.mix_points.outro_bars if a.mix_points is not None else None
        intro_bars = b.mix_points.intro_bars if b.mix_points is not None else None

        edge = transitions.score_transition(a, b)
        tempo_chip_text, pitch_pct = _tempo_chip(a, b)

        chips: list[tuple[str, bool]] = [
            (tempo_chip_text, False),
            (_key_chip(a, b, edge), False),
            *_bars_chips(outro_bars, intro_bars),
            *_warn_chips(
                outgoing_missing=outgoing_missing,
                incoming_missing=incoming_missing,
                annotation_risk_type=risk_type,
            ),
        ]

        tier = _tier(
            a,
            b,
            pitch_pct=pitch_pct,
            is_risky_annotated=is_risky_annotated,
            outgoing_missing=outgoing_missing,
            incoming_missing=incoming_missing,
        )
        plan = _plan(
            a,
            b,
            open_at_secs=open_at_secs,
            outro_bars=outro_bars,
            intro_bars=intro_bars,
            pitch_pct=pitch_pct,
            tempo_chip_text=tempo_chip_text,
            outgoing_missing=outgoing_missing,
            incoming_missing=incoming_missing,
            risk_type=risk_type,
        )

        steps.append(
            BoothStep(
                index=i + 1,
                from_id=a.track_id,
                to_id=b.track_id,
                from_label=f"{a.artist} — {a.title}",
                to_label=f"{b.artist} — {b.title}",
                open_at_secs=open_at_secs,
                outro_bars=outro_bars,
                intro_bars=intro_bars,
                pitch_pct=pitch_pct,
                tier=tier,
                chips=tuple(chips),
                plan=plan,
                fallback=_fallback(outro_bars),
                scout=_scout(a, b),
            )
        )

    return steps
