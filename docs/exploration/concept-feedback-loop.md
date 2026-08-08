# Concept Feedback Loop — Discovery

Discovery doc for issue #12. Not a spec, not a commitment. Sketches a lightweight mechanism for recording which generated concepts were actually used in real sets, and how that signal could feed back into canvas / Stage 2 scoring without becoming brittle.

> **Status:** discovery only. Do not implement before #17 lands and the existing novelty + canvas-scoring system has been validated in real use across 20+ runs. The richer history schema (#6) is already in place; the foundation exists.

---

## 1. Why this matters

The current novelty system in [`history.py`](../../src/mixlab/history.py) penalises a candidate canvas by its similarity to **every recent run** in the history file. The recency window is 10 entries; decay is `0.8^age`. This is symmetric — a concept that was generated, looked at, and immediately rejected counts the same as a concept that was played at three real club nights.

That is wrong on two axes:

- **False negatives:** a concept that has been played repeatedly should be deprioritised *more* than a concept that was generated once and ignored. Today both contribute identically to the similarity penalty.
- **False positives:** a concept that was generated and rejected should not penalise future generation at all — the DJ said no. Today rejected concepts continue to penalise novel candidates for ten runs.

A feedback loop closes that gap. The signal needed is binary at minimum (used / not-used) and three-valued at most (used / used-with-modifications / rejected).

## 2. Non-goals

The discovery deliberately rules out:

- A web UI, a server-side store, or a database.
- An API the DJ has to call from the booth — feedback capture has to survive the moment after a set, not during one.
- ML-based preference modelling. Any scoring change must remain explainable in a one-line formula.
- Discord reactions as the sole capture channel — too easy to lose, and the DJ may use MixLab without Discord (`DISCORD_BOT_TOKEN` is optional).

## 3. Data model

### 3.1 Where the signal lives

Add a `feedback` field to `HistoryEntry` in [`history.py`](../../src/mixlab/history.py). The field defaults to `None` (no signal yet) and is filled in retroactively — the entry is written at run time when only the concept is known; the verdict comes hours or days later.

```python
ConceptFeedback = Literal["played", "played_modified", "rejected", "unused"]

@dataclass
class HistoryEntry:
    # … existing fields …
    feedback: ConceptFeedback | None = None
    feedback_notes: str = ""           # free-text reason; never parsed structurally
    feedback_recorded_at: str = ""     # ISO timestamp; "" when no feedback yet
```

The four-value enum is the maximum useful resolution. `played` and `rejected` are the load-bearing cases; `played_modified` is signal that the concept's *shape* was right but specific picks were wrong (interesting for #7-style shape novelty); `unused` is the explicit-noise state — the DJ deliberately looked and passed, distinct from no-signal-yet (`None`).

### 3.2 Forward compatibility

`load_history` already filters incoming JSON to known fields (see `history.py:179` and the v0.10 forward-compat comment). Old history files without `feedback` load with the default `None`. The schema upgrade is invisible.

### 3.3 The `feedback` is *per run*, not *per concept*

A single run typically produces 3–6 concepts. The DJ usually picks one. Recording feedback per concept (rather than per run) needs concept IDs the DJ can reference — currently the only identifier is `concept_title`, which is free-text and may collide.

**Recommendation:** add a stable `concept_id` (uuid4) to `MixConcept` at curation time, store it in `HistoryEntry.concept_track_ids`'s sibling field, and let feedback target a single concept_id within the run. Multiple concepts from the same run can carry separate feedback verdicts.

```python
@dataclass
class HistoryEntry:
    # … existing …
    concept_ids: list[str] = field(default_factory=list)   # uuids, parallel to concept_track_ids
    feedback_by_concept: dict[str, ConceptFeedback] = field(default_factory=dict)
    feedback_notes_by_concept: dict[str, str] = field(default_factory=dict)
```

This is the recommended shape. The simpler per-run flat field above is the v0.1 fallback if the per-concept structure proves too heavy.

## 4. How the DJ enters feedback

Three plausible channels, ordered by friction:

### 4.1 CLI subcommand (recommended primary)

```bash
./mixlab feedback played            # most recent concept of most recent run
./mixlab feedback played -r abc123 -c def456 --notes "swapped track 7 for a Burial tune"
./mixlab feedback list              # last 10 entries with feedback state
./mixlab feedback rejected -c def456 --notes "opener was too hot"
```

Properties:

- Survives "after the set" capture — DJ runs it the next morning from terminal.
- Versionable — `.mixlab/concept-history.json` carries the verdict in git.
- Frictionless when shorthand is provided: `./mixlab feedback played` defaults to the last-generated concept.
- Discoverable via `--help`.

This is the primary mechanism. Everything else is a backup.

### 4.2 Discord reaction capture (secondary)

When `DISCORD_BOT_TOKEN` is set, MixLab posts the report to Discord. Adding a reaction listener that maps emoji → verdict is low-effort:

| Emoji | Verdict |
| ----- | ------- |
| ✅ | `played` |
| 🔁 | `played_modified` |
| ❌ | `rejected` |
| 🙈 | `unused` |

Bot polls or subscribes to message reactions on its own posts. When a reaction fires, look up the run by message ID (stored in history as `discord_message_id`), update the corresponding `HistoryEntry`. The reaction is recorded once; subsequent reactions overwrite (most recent wins).

This is the **lowest-friction** capture method for the actual session-night moment — but it depends on Discord being configured and the message still being reachable. Treat as augment, not replacement.

### 4.3 Manual JSON edit (escape hatch)

The history file is human-readable JSON. A power user editing it directly is a supported workflow — no extra mechanism needed. The CLI subcommand is the friendly UI; direct edit is always available for batch recovery, replays of historical data, or correcting a mistaken `played` mark.

## 5. How feedback influences scoring

Two integration points. Keep both conservative.

### 5.1 Weighted similarity_to_history

Today `similarity_to_history` returns `combined ∈ [0, 1]`. Feedback amplifies or attenuates this per-entry:

```python
FEEDBACK_MULTIPLIER = {
    "played":          1.4,  # played concepts penalise novelty more strongly
    "played_modified": 1.2,  # shape was right; slight extra weight
    "rejected":        0.0,  # DJ said no — do not penalise the next run
    "unused":          0.6,  # DJ looked and passed — partial signal
    None:              1.0,  # no signal — current behaviour
}

for age, entry in enumerate(reversed(recent)):
    base = base_similarity(canvas_shape, entry)
    multiplier = FEEDBACK_MULTIPLIER[entry.feedback]
    decayed = base * (_DECAY ** age) * multiplier
```

`rejected → 0.0` means the system genuinely forgets rejected runs for novelty purposes. That is the desired behaviour — a rejected concept should not gatekeep future generation.

The multipliers are tunable. They are deliberately blunt: a simple table the user can read, not a learned model.

### 5.2 Anchor preference boost

Tracks that appear in many `played` concepts are by definition battle-tested in *this DJ's* sets. Boost their anchor score in [`clustering.py`](../../src/mixlab/clustering.py) `score_anchors`:

```python
played_appearances = sum(
    1 for entry in history.runs[-30:]
    if entry.feedback == "played" and track_id in entry.concept_track_ids
)
played_boost = min(0.3, played_appearances * 0.05)
score += played_boost
```

This naturally up-weights tracks the DJ has *chosen* to play, distinct from tracks the catalog API already marks as played. Played-mode (`--mode played`) already restricts the pool; this is finer-grained — within the played pool, played-as-anchor tracks rise.

Cap the boost so a single hit track doesn't dominate.

## 6. Feedback decay and tradition

Feedback gets stale. A track played at every gig in 2024 may be unfashionable to that DJ by 2026.

Two decay options:

- **Time decay.** Multiply `played_appearances` weight by `0.95 ^ months_since`. Conservative — keeps recent feedback dominant without wiping old data.
- **Window decay.** Only consider feedback from the last 30 runs (matches `_MAX_HISTORY = 50` headroom with a 30-run sliding view for boost purposes).

Recommend **window decay** for v1 — simpler to reason about, and matches the existing `_RECENCY_WINDOW = 10` mental model.

## 7. Risks

| Risk | Mitigation |
| ---- | ---------- |
| DJ never enters feedback → system stays at v0 forever | Default-off boost; absence of feedback is the no-op state. No degradation when no data. |
| Over-fitting to past preferences (DJ's taste drifts) | Window decay; cap individual boost at 0.3; multipliers stay blunt and explainable. |
| Feedback recorded on the wrong concept | `feedback list` shows recent entries; CLI supports overwrite. Manual JSON edit is the recovery path. |
| Discord reactions misfire from non-DJ users | Restrict reaction-listening to messages the bot itself posted; ignore reactions from anyone other than the bot's configured user. |
| Concept IDs are unstable across runs | uuid4 per concept at curation time; never derived from content. Stable forever. |
| `played` vs `unused` ambiguous when the DJ used a *track* from the concept but not the concept itself | Document explicitly: `played` means the concept's framing and order were taken seriously, not that any individual track happened to appear. Add `feedback_notes` for nuance. |

## 8. Open questions

1. **Per-track feedback?** A `played` verdict could carry a list of `kept_track_ids` so the next run knows which tracks the DJ retained. Out of scope for v1 — adds capture friction. Worth revisiting if `played_modified` proves the most common verdict.
2. **Feedback influence on canvas weights?** Mode-aware weights (#24) could shift further when feedback exists — e.g. boost anchor_strength by an additional 5pp in played mode if the DJ historically prefers anchor-heavy concepts. Skip in v1; too easy to overfit.
3. **Integration with critique (#22)?** A `--deep` critique that disagrees with the DJ's verdict could itself be recorded — does the model's self-critique correlate with later rejection? Worthwhile diagnostic but not load-bearing for scoring.
4. **Export feedback?** A `feedback summary` subcommand surfacing aggregate stats (acceptance rate by genre / mode / arc_type) would be cheap and visible signal that the loop is working.

## 9. Suggested implementation order

If/when this discovery is converted into an implementation ticket:

1. Schema bump: add `concept_id` to `MixConcept`, `concept_ids` + `feedback_by_concept` to `HistoryEntry`. Forward-compat load path already in place.
2. CLI: `mixlab feedback played | played_modified | rejected | unused [-r RUN_ID] [-c CONCEPT_ID] [--notes TEXT]`.
3. CLI: `mixlab feedback list` (read-only diagnostic).
4. Scoring: `FEEDBACK_MULTIPLIER` applied inside `similarity_breakdown_to_history`.
5. Scoring: `played_appearances` boost in `score_anchors`.
6. Diagnostics: `--debug` emits the per-entry feedback verdict and multiplier in the novelty source line.
7. Tests for each scoring change. Aim for the same test coverage density as the existing novelty + anchor code.
8. Discord reaction listener as a *follow-up* — gate behind a separate sub-issue; the CLI mechanism stands alone.

Estimated scope: 1–2 days for steps 1–7. Discord reaction listener adds 1 more day plus integration testing against the Discord API.

## 10. Decision signal

Revisit this discovery doc when **all** of the following are true:

- Phase 3 has shipped (#22, #23, #24 done; #17 reviewed against real use).
- The user has at least 20 runs in `.mixlab/concept-history.json`.
- The user has at least once said out loud "MixLab keeps suggesting the same shape" or "MixLab didn't learn that I played that one".

Either statement is the signal that the novelty system is no longer doing enough on its own and external feedback is the missing input. Without that signal, the loop is speculative and not worth the build.
