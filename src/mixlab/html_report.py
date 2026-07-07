"""Deterministic, self-contained HTML report renderer (#45).

Renders one standalone HTML file per run: concept cards carrying everything the
text report already surfaces (title, arc, mood, track list, prose) plus computed
transition intelligence from :mod:`mixlab.transitions` (mechanism labels, blend
headroom, mixability score bands) and an inline energy sparkline.

Everything here is pure and stdlib-only. There are ZERO external requests: the CSS
and JS are inlined, the font stack is ``system-ui``, and no images/CDNs/fonts are
referenced. Output is byte-stable for identical inputs — the only time-varying
input is ``generated_at``, which the caller supplies so the renderer stays pure.
"""

from __future__ import annotations

import re
from html import escape

from mixlab import transitions
from mixlab.booth_sheet import BoothStep, build_booth_sheet
from mixlab.models import MixConcept, Track

_PROSE_SEPARATOR = "\n\n---\n\n"

# Sparkline geometry — fixed so output is deterministic and layout-stable.
_SPARK_W = 280
_SPARK_H = 48
_SPARK_PAD = 4
_MIN_SPARK_POINTS = 3


def _esc(value: str) -> str:
    """HTML-escape every dynamic string that reaches the document."""
    return escape(value, quote=True)


def _slugify(value: str) -> str:
    """Lowercase, collapse every run of non-alphanumeric characters to a single '-'."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return slug.strip("-")


def report_filename(genre: str | None, playlist_name: str | None, date_str: str) -> str:
    """Build the on-disk filename, e.g. ``mixlab-house-2026-07-07.html``.

    Playlist runs take precedence over genre; when both are absent the stem is
    ``mixlab-run-<date>``.
    """
    if playlist_name:
        stem = _slugify(playlist_name) or "run"
    elif genre:
        stem = _slugify(genre) or "run"
    else:
        stem = "run"
    return f"mixlab-{stem}-{date_str}.html"


def _fmt_duration(secs: int | None) -> str:
    if secs is None or secs <= 0:
        return "—"
    minutes, seconds = divmod(secs, 60)
    return f"{minutes}:{seconds:02d}"


def _fmt_bars(bars: float | None) -> str:
    if bars is None:
        return "—"
    return f"{bars:.0f}"


def _fmt_energy(energy: int | None) -> str:
    return "—" if energy is None else str(energy)


def _score_band(score: float) -> str:
    if score >= 0.7:
        return "t-good"
    if score >= 0.4:
        return "t-warn"
    return "t-bad"


def _resolved_tracks(concept: MixConcept, tracks_by_id: dict[str, Track]) -> list[Track]:
    return [t for tid in concept.track_ids if (t := tracks_by_id.get(tid)) is not None]


def _sparkline(tracks: list[Track]) -> str:
    """Inline SVG polyline of per-track energies. Empty string when <3 energies."""
    energies = [t.energy for t in tracks if t.energy is not None]
    if len(energies) < _MIN_SPARK_POINTS:
        return ""
    lo, hi = min(energies), max(energies)
    span = hi - lo
    inner_h = _SPARK_H - 2 * _SPARK_PAD
    inner_w = _SPARK_W - 2 * _SPARK_PAD
    points: list[str] = []
    for i, energy in enumerate(energies):
        x = _SPARK_PAD + (inner_w * i / (len(energies) - 1))
        frac = 0.5 if span == 0 else (energy - lo) / span
        y = _SPARK_PAD + (1.0 - frac) * inner_h
        points.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(points)
    return (
        f'<svg class="spark" viewBox="0 0 {_SPARK_W} {_SPARK_H}" width="{_SPARK_W}" height="{_SPARK_H}" '
        f'role="img" aria-label="Energy arc across the set">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{poly}"/></svg>'
    )


def _split_prose(report_text: str, n_concepts: int) -> tuple[list[str | None], list[str]]:
    """Pair leading report sections with concepts; collect any remainder as run notes.

    Returns ``(per_concept, trailing)``. When the report has at least one section per
    concept, section ``i`` pairs with concept ``i`` and the rest become trailing run
    notes. When the counts do not line up, the whole report becomes a single trailing
    section (never dropped) and no per-concept prose is emitted. An empty report yields
    no prose at all.
    """
    if not report_text.strip():
        return [None] * n_concepts, []
    # The validation-notes block is rendered as its own section from the structured
    # warnings list; strip its textual copy so run notes don't duplicate it.
    report_text = re.sub(r"\n*⚠ \*\*Validation Notes\*\*\n(?:- [^\n]*\n?)*", "\n", report_text)
    parts = report_text.split(_PROSE_SEPARATOR)
    if n_concepts > 0 and len(parts) >= n_concepts:
        per_concept: list[str | None] = list(parts[:n_concepts])
        trailing = parts[n_concepts:]
        return per_concept, trailing
    return [None] * n_concepts, [report_text]


def _render_track_rows(tracks: list[Track]) -> str:
    rows: list[str] = []
    for idx, t in enumerate(tracks, start=1):
        label = f"{t.artist} — {t.title}"
        intro = t.mix_points.intro_bars if t.mix_points else None
        outro = t.mix_points.outro_bars if t.mix_points else None
        rows.append(
            "<tr>"
            f'<td class="num">{idx}</td>'
            f'<td class="track" data-copy="{_esc(label)}" title="Click to copy">{_esc(label)}</td>'
            f'<td class="key">{_esc(t.camelot_key)}</td>'
            f'<td class="bpm">{t.bpm:g}</td>'
            f'<td class="dur">{_esc(_fmt_duration(t.duration_secs))}</td>'
            f'<td class="en">{_esc(_fmt_energy(t.energy))}</td>'
            f'<td class="bars">{_esc(_fmt_bars(intro))}</td>'
            f'<td class="bars">{_esc(_fmt_bars(outro))}</td>'
            "</tr>"
        )
    return (
        '<div class="tablewrap"><table class="tracks">'
        "<thead><tr>"
        "<th>#</th><th>Artist — Title</th><th>Key</th><th>BPM</th>"
        "<th>Dur</th><th>En</th><th>Intro</th><th>Outro</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _render_transitions(concept: MixConcept, tracks: list[Track]) -> str:
    if len(tracks) < 2:
        return ""
    annotations = {(tr.from_id, tr.to_id): tr for tr in concept.transitions}
    rows: list[str] = []
    for i in range(len(tracks) - 1):
        a, b = tracks[i], tracks[i + 1]
        edge = transitions.score_transition(a, b)
        mechanism = transitions.relation_label(edge, a, b)
        band = _score_band(edge.score)
        blend = f'<span class="blend">{_esc(edge.blend_label)}</span>' if edge.blend_label else ""
        annotation = annotations.get((a.track_id, b.track_id))
        risk = ""
        if annotation is not None and annotation.risk_type:
            risk = f'<span class="risk">{_esc(annotation.risk_type)}</span>'
        rows.append(
            f'<li class="{band}">'
            f'<span class="tpair">{i + 1}→{i + 2}</span>'
            f'<span class="tmech">{_esc(mechanism)}</span>'
            f"{blend}"
            f'<span class="tscore">{edge.score:.2f}</span>'
            f"{risk}"
            "</li>"
        )
    return '<div class="transitions"><h4>Transitions</h4><ol class="tlist">' + "".join(rows) + "</ol></div>"


def _render_booth_step(step: BoothStep) -> str:
    cls = "step"
    if step.tier == "tight":
        cls = "step tight"
    elif step.tier == "hard":
        cls = "step hard"
    chips = "".join(
        f'<span class="chip warn">{_esc(text)}</span>' if is_warn else f'<span class="chip">{_esc(text)}</span>'
        for text, is_warn in step.chips
    )
    fallback = f'<p class="fallback">{_esc(step.fallback)}</p>' if step.fallback is not None else ""
    scout = f'<p class="scout">{_esc(step.scout)}</p>' if step.scout is not None else ""
    return (
        f'<li class="{cls}">'
        '<div class="head">'
        f'<span class="pair">{step.index}→{step.index + 1}</span>'
        f'<span class="titles">{_esc(step.from_label)} <span class="into">into</span> {_esc(step.to_label)}</span>'
        "</div>"
        f'<div class="chips">{chips}</div>'
        f'<p class="plan">{_esc(step.plan)}</p>'
        f"{fallback}{scout}"
        "</li>"
    )


def _render_booth_sheet(concept: MixConcept, tracks_by_id: dict[str, Track]) -> str:
    steps = build_booth_sheet(concept, tracks_by_id)
    if not steps:
        return ""
    items = "".join(_render_booth_step(step) for step in steps)
    return (
        '<details class="prose-details booth-details" open><summary>Booth sheet</summary>'
        f'<ol class="booth">{items}</ol></details>'
    )


def _render_concept(
    concept: MixConcept,
    tracks_by_id: dict[str, Track],
    prose: str | None,
) -> str:
    tracks = _resolved_tracks(concept, tracks_by_id)
    parts: list[str] = ['<details class="card" open>']
    badge = f'<span class="arc">{_esc(concept.arc_type)}</span>' if concept.arc_type else ""
    parts.append(f'<summary class="card-head"><span class="title">{_esc(concept.title)}</span>{badge}</summary>')
    parts.append('<div class="card-body">')
    if concept.mood:
        parts.append(f'<p class="mood">{_esc(concept.mood)}</p>')
    if concept.name_reason:
        parts.append(f'<p class="reason"><em>{_esc(concept.name_reason)}</em></p>')
    spark = _sparkline(tracks)
    if spark:
        parts.append(f'<div class="sparkwrap">{spark}</div>')
    parts.append(_render_track_rows(tracks))
    parts.append(_render_transitions(concept, tracks))
    parts.append(_render_booth_sheet(concept, tracks_by_id))
    if prose is not None and prose.strip():
        parts.append(
            '<details class="prose-details"><summary>Full report</summary>'
            f'<pre class="prose">{_esc(prose)}</pre></details>'
        )
    parts.append("</div></details>")
    return "".join(parts)


def _render_snapshot(counts: dict[str, tuple[int, int]]) -> str:
    rows: list[str] = []
    for label, (total, available) in counts.items():
        pct = (available / total) if total else 0.0
        rows.append(
            "<tr>"
            f'<td class="snap-label">{_esc(label)}</td>'
            f'<td class="snap-count">{available} / {total}</td>'
            f'<td class="snap-bar"><span class="bar"><span class="fill" style="width:{pct * 100:.0f}%"></span></span></td>'
            "</tr>"
        )
    return (
        '<section class="snapshot"><h2>Crate snapshot</h2>'
        '<table class="snap"><tbody>' + "".join(rows) + "</tbody></table></section>"
    )


def _render_validation(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{_esc(w)}</li>" for w in warnings)
    return f'<section class="validation"><h2>⚠ Validation notes</h2><ul>{items}</ul></section>'


def _render_run_notes(trailing: list[str]) -> str:
    filtered = [p for p in trailing if p.strip()]
    if not filtered:
        return ""
    blocks = "".join(f'<pre class="prose">{_esc(p)}</pre>' for p in filtered)
    return (
        '<section class="run-notes"><details class="prose-details"><summary>Run notes</summary>'
        f"{blocks}</details></section>"
    )


_STYLE = """
:root {
  --bg: #f7f8fa; --fg: #1c2024; --muted: #5c636e; --card: #ffffff; --border: #e2e5ea;
  --accent: #5b8def; --good: #1a7f47; --warn: #9a6a00; --bad: #c0392b;
  --bar-track: #e6e9ef; --code-bg: #f2f4f7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c; --fg: #e6e8ec; --muted: #9aa2ad; --card: #1c2027; --border: #2b313a;
    --accent: #7aa4f5; --good: #4fbf82; --warn: #d9a441; --bad: #ec7063;
    --bar-track: #2b313a; --code-bg: #1a1e24;
  }
}
:root[data-theme="light"] {
  --bg: #f7f8fa; --fg: #1c2024; --muted: #5c636e; --card: #ffffff; --border: #e2e5ea;
  --accent: #5b8def; --good: #1a7f47; --warn: #9a6a00; --bad: #c0392b;
  --bar-track: #e6e9ef; --code-bg: #f2f4f7;
}
:root[data-theme="dark"] {
  --bg: #14171c; --fg: #e6e8ec; --muted: #9aa2ad; --card: #1c2027; --border: #2b313a;
  --accent: #7aa4f5; --good: #4fbf82; --warn: #d9a441; --bad: #ec7063;
  --bar-track: #2b313a; --code-bg: #1a1e24;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.45; font-size: 15px;
}
main { max-width: 880px; margin: 0 auto; padding: 24px 16px 64px; }
header.top h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: -0.01em; }
header.top .ctx { color: var(--muted); white-space: pre-wrap; margin: 2px 0; font-size: 14px; }
header.top .gen { color: var(--muted); font-size: 13px; margin-top: 6px; }
h2 { font-size: 17px; margin: 28px 0 10px; }
h4 { font-size: 14px; margin: 14px 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
section.snapshot table.snap { width: 100%; border-collapse: collapse; }
table.snap td { padding: 4px 8px; border-bottom: 1px solid var(--border); font-size: 14px; }
.snap-count { font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; }
.snap-bar { width: 45%; }
.bar { display: block; height: 8px; background: var(--bar-track); border-radius: 4px; overflow: hidden; }
.fill { display: block; height: 100%; background: var(--accent); }
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  margin: 14px 0; padding: 4px 16px 4px;
}
.card-head { cursor: pointer; padding: 12px 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.card-head .title { font-size: 18px; font-weight: 650; }
.arc {
  font-size: 12px; color: var(--accent); border: 1px solid var(--accent);
  border-radius: 999px; padding: 1px 9px; font-weight: 600;
}
.card-body { padding-bottom: 12px; }
.mood { margin: 2px 0 4px; }
.reason { margin: 2px 0 10px; color: var(--muted); }
.sparkwrap { color: var(--accent); margin: 8px 0 14px; }
.spark { display: block; max-width: 100%; }
.tablewrap { overflow-x: auto; }
table.tracks { width: 100%; border-collapse: collapse; font-size: 14px; }
table.tracks th {
  text-align: left; font-weight: 600; color: var(--muted); font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.03em; padding: 6px 8px; border-bottom: 1px solid var(--border);
}
table.tracks td { padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
td.num, td.bpm, td.dur, td.en, td.bars { font-variant-numeric: tabular-nums; white-space: nowrap; }
td.num, td.en, td.bars { color: var(--muted); }
td.track { cursor: pointer; }
td.track:hover { color: var(--accent); }
td.track.copied { color: var(--good); }
td.track.copied::after { content: " ✓ copied"; font-size: 12px; color: var(--good); }
.transitions .tlist { list-style: none; margin: 0; padding: 0; }
.tlist li {
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  padding: 5px 8px; border-left: 3px solid var(--border); margin: 4px 0; font-size: 14px;
}
.tlist li.t-good { border-left-color: var(--good); }
.tlist li.t-warn { border-left-color: var(--warn); }
.tlist li.t-bad { border-left-color: var(--bad); }
.tpair { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 44px; }
.tmech { font-weight: 550; }
.blend { color: var(--muted); font-size: 13px; }
.tscore { font-variant-numeric: tabular-nums; color: var(--muted); margin-left: auto; }
.risk {
  font-size: 12px; color: var(--warn); border: 1px solid var(--warn);
  border-radius: 999px; padding: 0 8px;
}
.prose-details { margin-top: 12px; }
.prose-details summary { cursor: pointer; color: var(--accent); font-size: 14px; }
pre.prose {
  white-space: pre-wrap; word-wrap: break-word; background: var(--code-bg);
  border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 13px;
  overflow-x: auto; margin: 10px 0 0;
}
section.validation ul { margin: 0; padding-left: 20px; }
section.validation li { color: var(--warn); margin: 4px 0; }
footer { margin-top: 40px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--border); padding-top: 14px; }
.booth { list-style: none; margin: 8px 0 0; padding: 0; }
.booth li.step { border: 1px solid var(--border); border-left: 4px solid var(--good); border-radius: 8px; padding: 10px 12px; margin: 10px 0; }
.booth li.step.tight { border-left-color: var(--warn); }
.booth li.step.hard { border-left-color: var(--bad); }
.step .head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.step .pair { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px; min-width: 40px; }
.step .titles { font-weight: 600; }
.step .titles .into { color: var(--muted); font-weight: 400; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; margin: 2px 0 8px; }
.chip { background: var(--code-bg); border: 1px solid var(--border); border-radius: 999px; font-size: 12px; padding: 1px 9px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.chip.warn { color: var(--warn); border-color: var(--warn); background: transparent; }
.step .plan { margin: 0; font-size: 14px; }
.step .fallback { margin: 6px 0 0; font-size: 13px; color: var(--muted); }
.step .fallback::before { content: "↩ "; }
.step .scout { margin: 6px 0 0; font-size: 13px; color: var(--warn); }
.step .scout::before { content: "⚠ "; }
@media print { .booth li.step { break-inside: avoid; } }
""".strip()

_SCRIPT = """
document.addEventListener('click', function (ev) {
  var el = ev.target.closest('[data-copy]');
  if (!el || !navigator.clipboard) return;
  navigator.clipboard.writeText(el.getAttribute('data-copy')).then(function () {
    el.classList.add('copied');
    setTimeout(function () { el.classList.remove('copied'); }, 1200);
  });
});
""".strip()


def render_html_report(
    concepts: list[MixConcept],
    tracks_by_id: dict[str, Track],
    *,
    report_text: str,
    report_context: str,
    validation_warnings: list[str],
    counts: dict[str, tuple[int, int]] | None = None,
    generated_at: str = "",
) -> str:
    """Render a single self-contained HTML document for a MixLab run.

    Deterministic for identical inputs: the only time-varying value is
    ``generated_at``, which the caller supplies. No network, no external assets.
    """
    per_concept, trailing = _split_prose(report_text, len(concepts))

    ctx_lines = "".join(f'<p class="ctx">{_esc(line)}</p>' for line in report_context.split("\n") if line)
    gen_line = f'<p class="gen">{_esc(generated_at)}</p>' if generated_at else ""

    body: list[str] = [
        "<main>",
        f'<header class="top"><h1>MixLab</h1>{ctx_lines}{gen_line}</header>',
    ]
    if counts:
        body.append(_render_snapshot(counts))
    for concept, prose in zip(concepts, per_concept, strict=True):
        body.append(_render_concept(concept, tracks_by_id, prose))
    body.append(_render_run_notes(trailing))
    body.append(_render_validation(validation_warnings))
    footer_gen = f" · {_esc(generated_at)}" if generated_at else ""
    body.append(f"<footer>Generated by MixLab{footer_gen}</footer>")
    body.append("</main>")

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>MixLab report</title>"
        f"<style>{_STYLE}</style></head><body>" + "".join(body) + f"<script>{_SCRIPT}</script></body></html>"
    )
