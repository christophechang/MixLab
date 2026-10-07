# Implementation: regime-crossing hard finding

Source plan: `regime-crossing-hard-finding.md`. Tier: medium.

Commands (from the repo root, using the main checkout's venv):

```bash
V=/Users/christophechang/Development/MixLab/.venv/bin/python
$V -m ruff format . && $V -m ruff check . && $V -m mypy . && $V -m pytest -q
```

## Unit 1: validator (`src/mixlab/llm.py` `validate_stage2_output`, tests)

- Fold the crossing check so it runs for every concept. The traverse block runs
  unconditionally, with a guard `a.bpm > 0 and b.bpm > 0`. The suffix depends on
  `is_traverse_concept`.
- Delete the soft `elif … regime crossing without a ratio bridge` branch and its
  comment. Rewrite the comment block at ~L1510.
- Tests: change and rename the three failing tests per the plan, rename
  `..._unjustified_crossing_no_duplicate` to `..._unjustified_crossing_draws_jump_and_crossing`
  (assert `BPM jump` AND `unbridged regime crossing` both present, soft wording
  absent, docstring rewritten), delete
  `test_non_traverse_crossing_warning_contains_no_hard_markers`, and add the
  wording, 3:4, 12 BPM and zero-BPM tests.
- Verify: `pytest tests/test_llm.py -q -k "crossing or bpm_jump"`, then the full
  suite.

## Unit 2: revision (`_qualifies_for_revision`, `revise_concepts`, tests)

- Add the single-crossing qualification rule and update the docstring.
- Add `_revision_finding_count(findings: list[str]) -> int`, which dedups a
  `BPM jump` against an `unbridged regime crossing` on the same pair. Pinned parse:
  key = `re.search(r"BPM jump [\d.]+ between (.*)$", w).group(1)`; covered iff some
  crossing finding contains `f" between {key} — "` (trailing ` — ` is the delimiter,
  avoiding prefix collisions like title "W" vs "Wonder"). Use it for n_before, n_after,
  and the pre-request stderr "N hard finding(s)" line.
- Add the "crossings must not increase" rejection and its stderr line. Check it
  before the count comparison.
- Tests: `test_qualifies_for_revision_single_regime_crossing_true`,
  `test_revision_finding_count_dedups_bpm_jump_for_crossing_pair`,
  `test_revise_concepts_rejects_revision_that_adds_crossings`,
  `test_revise_concepts_rejects_reannotation_only_revision`. Model these on the
  existing `revise_concepts` tests and their `_revision_lib` fixture, with BPMs
  overridden per track. Traps: put Camelot jumps on pairs other than the crossing
  (an `is_risky` annotation also suppresses Camelot on its own pair); blend risk
  stays silent only because fixtures lack cue data. Add a prefix-collision case to
  the count-helper test. Stderr text exactly:
  `revision added a regime crossing — keeping original`.
- Verify: full suite.

## Unit 3: prompts (`llm.py` curation prompt ~L1688, `_STAGE2_REVISION_SYSTEM`)

- Add the one-clause tempo pivot qualifier and the revision rule, as the plan
  words them.
- Verify: full suite. Prompt snapshot or substring tests may need updating; if so,
  call that out.

## Unit 4: docs

- Content per source plan §5 bullets (L583 must state the single-crossing trigger, the
  crossings-must-not-increase rule, and add blend risk / unbridged crossing / direction
  key tracks to the hard list). README L292, L578, L583; `--risk` help in `__main__.py` ~L1776; one CHANGELOG
  `## Unreleased` bullet in bold-lead prose style.
- Verify: ruff, mypy, full suite.

Commit per unit with conventional commits (`fix:` for units 1–3, `docs:` for unit 4).

## Closing

- PR body carries the source plan's "Out of scope / known gaps" list and notes the
  live breakbeat-run acceptance check is for the operator after deploy.
