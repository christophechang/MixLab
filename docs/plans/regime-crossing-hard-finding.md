# Plan: make unbridged regime crossings a hard finding for every concept

## Problem

Run `r_20261007_d4c0` (Breakbeat, unplayed, risk high). The mix "Crate Archaeology"
goes 123 → 129 → **156** → **141**. Neither of the last two moves can be beatmatched:

| Move | Straight stretch | Nearest ratio | Verdict (`tempo_relation`) |
|---|---|---|---|
| 129 → 156 | 20.9% | 4:3 → 172, 9.3% off | incompatible |
| 156 → 141 | 9.6% | 3:4 → 117, way off | incompatible |

Compare "Ratio Crossing" 168 → 126: exactly 3:4, stretch 0%. That is a real ratio
bridge and must stay allowed.

Why Crate Archaeology shipped as it did (`src/mixlab/llm.py`, ~L1510–1558):

1. At **any** risk level, a transition annotated `is_risky=True` with a non-empty
   `risk_type` (`justified_risk`) suppresses the `BPM jump` hard finding. At `high`,
   the threshold is also relaxed to 20 BPM for `is_risky` moves.
2. The fallback check for non-traverse concepts emits
   `regime crossing without a ratio bridge … plan a cut or a reorder`, which is
   **warn-only by design**: its wording avoids every `_HARD_FINDING_MARKERS`
   substring, so it never triggers the Stage 2 self-revision pass.
3. Only `genre_traverse` concepts get the hard `unbridged regime crossing` variant.
4. Revision also needs **two** hard findings (`_qualifies_for_revision`), so even a
   single hard finding would not have been repaired.

Result: a crossing no DJ can ride is reported, but never repaired.

The breakbeat pool has no BPM filter (`GENRE_MAP["breakbeat"]` covers Breaks and
Hardcore, roughly 120–175), so this applies to every breakbeat run, not just this one.

## Definition

An **unbridged regime crossing** is an adjacent pair where both BPMs are > 0,
`abs(a.bpm - b.bpm) > 12.0`, and `tempo_relation(a.bpm, b.bpm)` returns
`"incompatible"`. This is the existing traverse definition, unchanged.

Consequence, stated on purpose: since `tempo_relation` allows only ±6% straight,
any non-ratio move over 12 BPM (below ~200 BPM) is now a hard finding **at every
risk level, whatever its annotation**. The `--risk` BPM thresholds and the
`justified_risk` suppression still govern the `BPM jump` warning, but they can no
longer let a >12 BPM unbridged move through unflagged. The thresholds are not
removed. They still matter for the 10–12 BPM band at `low` and for Camelot.

Known accepted gap: an incompatible move of 6%–12 BPM (for example 129 → 141) is
not a crossing. The 12 BPM floor is kept deliberately (it was tuned for traverse in
#82). It is pinned by a test, and the revision prompt states the rule the validator
actually enforces.

## Change

Scope: MixLab only. No API or contract change: warning strings are free text that
only `html_report` and `summary.json` pass through, and no code parses them except
`_HARD_FINDING_MARKERS`. Playlist mode never runs `validate_stage2_output`, so it is
unaffected.

### 1. One hard check for all concepts (`validate_stage2_output`, llm.py ~L1510)

- Run the existing traverse crossing loop for **every** concept, not just
  `genre_traverse`, and add the `bpm > 0` guard on both sides.
- Wording: `{label} unbridged regime crossing {a:g}→{b:g} between {A} and {B}`,
  then a suffix: `— traverse hops must be ratio bridges` for traverse (unchanged)
  and `— reorder or swap so the tempo move is a ratio bridge` for everything else.
  Both contain the existing marker, so `_HARD_FINDING_MARKERS` is unchanged.
- **Not suppressed by `justified_risk`**, the same as traverse today.
- Delete the warn-only `elif … regime crossing without a ratio bridge` branch.
- **No dedup with `BPM jump`.** An unannotated crossing over the risk threshold
  draws both findings, exactly as traverse concepts do today. This keeps the
  `BPM jump` semantics and the risk-threshold tests as they are, and an annotated
  crossing now draws one hard finding instead of a soft one.

### 2. A single crossing qualifies for revision (`_qualifies_for_revision`, ~L2961)

Add a rule like the existing `direction key tracks missing` one: a single
`unbridged regime crossing` finding qualifies on its own. Update the docstring.

### 3. Revision accept counting (`revise_concepts`, ~L3131)

Two changes to how n_before / n_after are counted. Emission stays no-dedup, so the
validator output and the tests about it are untouched.

- **Count one defect once.** Add a helper `_revision_finding_count(findings) -> int`
  that counts hard findings but skips a `BPM jump` finding when there is an
  `unbridged regime crossing` finding for the same pair. Match on the shared
  `between {A} — {T} and {B} — {T}` substring. Use it for both n_before and n_after.
  Without this, a revision that only re-annotates a crossing as `is_risky` would
  suppress the `BPM jump`, go from 2 to 1, and be accepted as "Revised" with the
  unplayable hop still there.
- **Crossings must not increase.** Also reject the revision (keep the original) if
  its count of `unbridged regime crossing` findings is higher than the original's.
  Print `Revision: {title} — revision added a regime crossing — keeping original`
  to stderr.

### 4. Prompts

- `_STAGE2_REVISION_SYSTEM`: add "tempo-regime crossings with no ratio bridge" to
  the opening list of findings, and add one rule: adjacent tracks more than 12 BPM
  apart must form a halftime, double, 3:4 or 4:3 ratio within ±6%. Fix it by
  reordering, swapping or dropping tracks; re-annotating the transition as risky
  does not fix it.
- Stage 2 curation prompt (~L1688, "Allow bold moves — larger key jumps, tempo
  pivots …"): qualify tempo pivots with "a tempo pivot of more than 12 BPM must be
  a ratio bridge (halftime/double/3:4/4:3); an `is_risky` annotation does not
  excuse an unbridged one". One clause, so generation is not told the opposite of
  what validation enforces.

### 5. Docs

- README "Risk knob" (~L292): note that the relaxed 20 BPM threshold never covers an
  unbridged crossing, which is hard at every risk level.
- README validation bullet (~L578): add unbridged regime crossings (> 12 BPM, no
  ratio) to the strong-tier list.
- README "Bounded self-revision" (~L583): the trigger is two or more hard findings,
  **or** a single unbridged regime crossing, **or** a single missing pinned key
  track. Add blend risk, unbridged crossing and direction key tracks to the
  hard-finding list. Mention the rule that crossings must not increase.
- `--risk` help text in `src/mixlab/__main__.py` (~L1776): add a half-clause saying
  unbridged crossings stay hard at every level.
- Update the comment block at llm.py ~L1510–1514 so no stale "warn-only variant"
  text remains.
- CHANGELOG `## Unreleased`: one bold-lead prose bullet in the existing style.

### Out of scope / known gaps (state in PR)

- BPM-jump thresholds, `PITCH_WINDOW` and the 12 BPM floor are unchanged.
- Intended side effect on traverse: the new `bpm > 0` guard means a 0-BPM (unknown
  tempo) track no longer draws a traverse crossing. A >15 BPM gap still draws
  `BPM jump`.
- No pool-level BPM filter for breakbeat.
- mixlab-web copy `src/trigger/describeCombo.ts` (high risk: "room up to 20 BPM")
  overstates the room now. That is a separate follow-up in that repo.
- `--resequence` reorders after the last validation and could introduce a crossing
  that nothing re-checks. This already exists today. Note it, don't fix it.
- A concept with no matched canvas is reported but skips revision (existing
  behaviour). A pinned key track stranded across a tempo gap can't be fixed without
  dropping it, so the revision nets zero and is rejected. That is acceptable.

## Decision (operator, 2026-10-07)

**No `cut_only` exemption.** The model can label any transition `cut_only`, so an
exemption would bring back the exact loophole that step 1 closes. A cut from 129 to
156 also still breaks the set's tempo coherence.

## Tests (`tests/test_llm.py`)

Existing tests to change (block ~L5100–5260):

- `test_validate_stage2_output_non_traverse_incompatible_crossing_silent` (77→174
  non-traverse): rename to `test_validate_stage2_output_non_traverse_incompatible_crossing_hard`
  and assert that `unbridged regime crossing` is present.
- `test_validate_stage2_output_non_traverse_justified_crossing_warns_softly`:
  rename to `..._justified_crossing_is_hard`. Assert `unbridged regime crossing`
  is present, the soft wording is absent and `BPM jump` is absent (still suppressed
  by justified_risk).
- `test_validate_stage2_output_non_traverse_unjustified_crossing_no_duplicate`:
  rename to `..._unjustified_crossing_draws_jump_and_crossing`. Assert both
  `BPM jump` and `unbridged regime crossing` are present, and the soft wording is
  absent.
- `test_validate_stage2_output_non_traverse_subthreshold_incompatible_crossing_warns`
  (a >12 BPM pair under the risk threshold): assert it is now
  `unbridged regime crossing` and that `BPM jump` is absent.
- `test_non_traverse_crossing_warning_contains_no_hard_markers`: delete it, because
  its premise is reversed. It is replaced by the marker test below.
- BPM-jump tests (~L3643–3857, L4863, L4881), test_main.py and the
  `revise_concepts` tests (all at 174 BPM) need no change. This was verified by
  applying steps 1–2 to a scratch copy: only the three tests listed above failed.

New tests:

- `test_validate_stage2_output_non_traverse_crossing_wording_carries_hard_marker`:
  the non-traverse crossing string contains `unbridged regime crossing` and the
  reorder/swap suffix.
- `test_validate_stage2_output_three_four_bridge_not_a_crossing`: 168→126 gives no
  crossing and no `BPM jump`.
- `test_validate_stage2_output_twelve_bpm_incompatible_move_not_a_crossing`:
  129→141 gives no crossing (pins the accepted 12 BPM floor).
- `test_validate_stage2_output_zero_bpm_pair_not_a_crossing`: 0→150 gives no
  crossing. It still draws `BPM jump`, so don't assert that absent.
- `test_qualifies_for_revision_single_regime_crossing_true`: the only hard finding
  is one crossing, so the concept qualifies.
- `test_revise_concepts_rejects_revision_that_adds_crossings`: build the fixture so
  only the new rule rejects. Annotate every crossing pair `is_risky=True,
  risk_type="chapter_pivot"` (this suppresses `BPM jump`). The original has
  1 crossing + 2 Camelot jumps (count 3); the mocked revision has 2 crossings and
  no Camelot jumps (count 2). Assert the original is kept, that 2 < 3 (so the
  total did drop), and that stderr contains "added a regime crossing".
- `test_revise_concepts_rejects_reannotation_only_revision`: the original has an
  unannotated crossing (crossing + BPM jump) plus one Camelot jump. The mocked
  revision keeps the order and only annotates the crossing pair `is_risky`. With
  the dedup count it goes 2 → 2, so the original is kept and there is no
  **Revised** annotation.
- `test_revision_finding_count_dedups_bpm_jump_for_crossing_pair`: a BPM jump
  plus a crossing on the same pair count 1; a BPM jump on a different pair
  counts separately.
- Traverse tests stay green unchanged.

## Acceptance

- `ruff format`, `ruff check`, `mypy`, `pytest` all green.
- On a real breakbeat run, no `regime crossing without a ratio bridge` lines
  appear. Each `unbridged regime crossing` either went through revision (the
  report shows a **Revised** annotation) or remains in ⚠ Validation Notes /
  `summary.json` `conceptWarnings`. 3:4 bridges like Ratio Crossing still appear.

## Risk

- More revision calls per run: one bounded Stage 2 call per concept with an
  unbridged hop. The curation-prompt clause should keep that rate low.
- If the pool has no bridging track, revision keeps the original. That is the same
  as today, so it can't make a run worse.
