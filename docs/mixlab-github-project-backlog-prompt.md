# MixLab GitHub Project and Backlog Prompt

## Purpose

Use this prompt with Claude Code to review the current MixLab Mix Canvas architecture, protect what is already working, then create a GitHub Project and a detailed backlog of implementation tickets for the next phase.

This is **not** a code implementation prompt.

Claude should inspect the repo, think hard about the current state, create a well-structured GitHub Project, and populate it with tickets that contain enough detail to implement safely later.

---

# MixLab next phase prompt: create GitHub Project and backlog for safe Mix Canvas improvements

## Context

I have a working Python CLI called MixLab that generates DJ mix concepts from a Rekordbox XML collection.

It is already useful and I do not want to destabilise it.

MixLab currently:

- parses a Rekordbox XML collection
- filters played and unplayed tracks
- filters by requested genre
- uses Stage 1 free/cheap LLMs for technical concept generation
- wraps Stage 1 concepts into deterministic Mix Canvases
- passes selected canvases into Stage 2
- uses Anthropic Sonnet as the main creative DJ brain
- generates DJ mix concepts with track order, roles, energy path, sections, transitions, and report prose
- exports Rekordbox XML playlists
- delivers the report to Discord
- writes concept history to `.mixlab/concept-history.json`

The recent Mix Canvas upgrade was successful. It improved the Stage 1 to Stage 2 handoff without adding required premium LLM calls.

## Important constraint

This is a real working system, not a prototype.

Do not propose risky rewrites.

Do not increase normal premium LLM usage.

Do not replace the Stage 2 Sonnet creative role.

Do not change playlist completion mode unless there is a strong reason and the change is clearly separated.

Do not implement feature code as part of this task.

This task is to:

1. inspect the codebase
2. understand the current implementation
3. decide the safest next backlog
4. create a GitHub Project
5. create detailed GitHub issues/tickets
6. organise those tickets into a sensible delivery order

## Current architecture strengths

The recent upgrade added:

### Mix Canvas selection

Stage 1 concepts are now wrapped into structured Mix Canvases before Stage 2 receives them.

Each canvas carries:

- deterministic role candidates:
  - opener
  - groove-locker
  - builder
  - pivot
  - peak
  - closer
- contrast assets:
  - vocal moments
  - texture changes
  - darker turns
  - brighter lifts
  - lower-pressure resets
- risk notes:
  - weak opener pool
  - weak closer pool
  - BPM spread issues
  - artist repetition
  - label repetition
  - too-similar midsection

Up to 6 canvases are selected for Stage 2 using deterministic weighted scoring rather than random sampling from large pools.

### Three-pool BPM partitioning

Tracks are no longer hard-filtered at ±6 BPM.

Each cluster is partitioned into:

- core: within ±6 BPM from median
- bridge: within ±12 BPM from median
- wildcard: more than ±12 BPM from median

Core tracks are normal candidates.

Bridge and wildcard tracks are retained as canvas metadata and can be used by Stage 2 for structural roles such as opener, pivot, reset, or closer when BPM deviation is intentional.

### Concept history and novelty scoring

Each successful run writes to:

```text
.mixlab/concept-history.json
```

Future runs apply a novelty penalty using recent track overlap.

The implementation uses Jaccard similarity over a recent run window, with decay over time.

This helps repeated runs for the same genre move toward different corners of the collection.

### Post-Stage-2 validation

A warn-only validation pass runs after Stage 2 and before delivery.

It checks for:

- missing track IDs
- denylist violations
- played-track violations
- large Camelot jumps
- large BPM jumps
- repeated artists

Warnings appear in the Discord report and never abort a run.

## What is good about the current state

The system is now much better than the old version because:

- Stage 2 no longer receives random large technical pools
- Sonnet receives role candidates and risk notes, not flat track lists
- BPM is no longer acting as an absolute creative gate
- repeated runs have some memory
- validation makes problems visible
- the cost shape has stayed sensible
- the architecture still preserves Sonnet as the DJ brain

This is a good foundation.

## Areas that may still need work

Think carefully about these. Do not assume all are worth implementing now.

### 1. Canvas scoring observability

The current system may select good canvases, but I need to understand why.

Potential improvement:

- add optional debug or verbose output showing:
  - all candidate canvases
  - selected canvases
  - rejected canvases
  - score breakdowns
  - novelty penalties
  - overlap penalties
  - core / bridge / wildcard counts
  - risk notes
  - reason selected
  - reason rejected

The goal is to make Mix Canvas selection explainable without cluttering normal Discord reports.

### 2. Novelty currently focuses mainly on track overlap

The concept history currently penalises recent overlap using track IDs.

That is useful, but it may not catch repeated concept shapes.

Two different runs could use different tracks but still produce the same kind of concept:

```text
172-174 BPM dark rolling D&B
Wave energy path
minimal vocals
warehouse pressure
similar opener/peak/closer shape
```

Potential improvement:

- add concept-shape novelty using deterministic fields:
  - genre
  - BPM band
  - dominant Camelot zone
  - energy path
  - mood tags if available
  - role pattern
  - anchor artist or label overlap
  - opener/closer type
  - vocal density if available

This should be done cautiously and without embeddings or extra premium LLM calls unless there is a compelling reason.

### 3. Role candidates must remain hints

Deterministic role inference is useful but imperfect.

It cannot truly know:

- intro strength
- outro strength
- breakdown placement
- vocal placement
- bassline clash
- emotional resolution
- actual mixability

Potential improvement:

- ensure Stage 2 prompt treats role candidates as hints, not final truth
- allow Sonnet to override role candidates
- ask Sonnet to explain important overrides
- avoid making deterministic role inference too authoritative

### 4. Bridge and wildcard tracks may currently be too decorative

Bridge and wildcard tracks are available to Stage 2, but they may not influence the canvas concept enough.

Sometimes an outlier track could be a concept-defining opener, closer, reset, or leftfield anchor.

Potential improvement:

- add a special category such as:
  - `concept_anchor_candidates`
  - `structural_exception_candidates`
  - `wildcard_anchor_candidates`

These would be bridge or wildcard tracks that are unusual but potentially important.

The goal is not to make the system risky. The goal is to allow intentional creative exceptions.

### 5. Validation is currently mostly technical

The validation pass catches useful technical issues, but it does not yet fully validate the DJ-thinking structure.

Potential additional warn-only checks:

- missing opener role
- missing closer role
- no clear peak or payoff
- no clear resolution
- bridge or wildcard used in a non-structural role without justification
- three or more consecutive tracks with the same role family
- declared energy path does not roughly match available energy data
- final track was not a closer candidate
- too many high-energy tracks in a row
- all tracks sit in the same role or energy band

These should remain warnings only.

### 6. The architecture may still be technical-first

The current flow may still be:

```text
technical cluster
→ Stage 1 concept
→ Mix Canvas wrapper
→ Stage 2 creative selection
```

The long-term ideal may be closer to:

```text
musical possibility
→ technical feasibility
→ Mix Canvas
→ Stage 2 creative selection
```

Potential future idea:

- create deterministic canvas seeds before Stage 1 or before canvas building:
  - biggest safe technical cluster
  - strongest high-energy anchor cluster
  - strongest low-pressure or opener-led cluster
  - most distinctive artist / label / mood cluster
  - most bridge-heavy adventurous cluster
  - strongest closer-led cluster

This may be too much for now. Think carefully before recommending it.

## What I want from you

Please inspect the actual codebase before creating the GitHub Project and issues.

Start by understanding the current implementation, especially:

- MixCanvas models
- BPM partitioning
- role inference
- contrast detection
- risk note generation
- canvas scoring
- canvas selection
- concept history
- novelty calculation
- Stage 2 prompt construction
- validation
- Discord report generation
- tests
- existing GitHub issue conventions, labels, milestones, and project setup if any

Then create a GitHub Project and populate it with a safe, well-scoped backlog.

## Key instruction

I want you to think hard, but I do not want speculative architecture for its own sake.

The system is already good.

The next phase should reduce risk, improve explainability, and improve musical accountability.

Before creating tickets, ask yourself:

- Does this improve actual DJ output quality?
- Does this make the system easier to trust?
- Does this preserve the current working pipeline?
- Does this avoid extra required Sonnet calls?
- Does this avoid overfitting to deterministic heuristics?
- Does this keep Sonnet as the creative DJ brain?
- Is this a small, testable change?
- Can it fail safely?
- Can this be rolled back cleanly?
- Is this ticket small enough to implement and review independently?
- Have i thought if the impact on the three modes (unplayed tracks, all tracks, playlist mode)

## Questions to answer before creating tickets

Please answer these internally after inspecting the code, then use the answers to shape the backlog.

1. Does the current implementation match the plan that was intended?
2. Are there any bugs, edge cases, or mismatches in the Mix Canvas implementation?
3. Is the scoring model likely to over-favour large safe pools?
4. Is the novelty penalty strong enough, too strong, or too narrow?
5. Is the Stage 2 prompt using canvas metadata correctly?
6. Does the Stage 2 prompt clearly say role candidates are hints, not facts?
7. Does the Stage 2 prompt clearly require bridge and wildcard justification?
8. Does the Stage 2 prompt allow Sonnet to reject weak canvases?
9. Are validation warnings useful and not too noisy?
10. Are there obvious missing validation checks that would catch bad DJ structure?
11. Is concept history written at the right point in the run?
12. Is concept history robust if the file is missing, corrupt, or stale?
13. Are there enough tests around the new behaviour?
14. What should be tuned before adding any new features?
15. What should be explicitly avoided because it introduces too much risk?

## Priority order for the backlog

Prefer improvements in this order:

1. Observability and diagnostics
2. Stage 2 prompt tightening
3. Warn-only validation improvements
4. Concept history and novelty tuning
5. Scoring weight tuning
6. Bridge/wildcard concept-anchor handling
7. Any larger concept-first redesign

Do not jump straight to a large redesign.

## GitHub Project requirements

Create a GitHub Project for this next phase.

Suggested project title:

```text
MixLab: Mix Canvas Quality and Trust
```

If a better title emerges from inspecting the repo and existing project naming, use that.

The project should be organised around safe delivery.

Suggested fields or columns:

- Backlog
- Ready
- In Progress
- Review
- Done

If the repository already uses another GitHub Project layout, follow the existing conventions instead.

Suggested labels:

- `area:mix-canvas`
- `area:stage2`
- `area:validation`
- `area:history`
- `area:diagnostics`
- `type:enhancement`
- `type:refactor`
- `type:test`
- `risk:low`
- `risk:medium`
- `priority:p1`
- `priority:p2`
- `priority:p3`

Use existing labels if the repo already has suitable ones.

## Ticket requirements

Create detailed GitHub issues for the selected backlog.

Each ticket should include:

1. Title
2. Problem statement
3. Goal
4. Context
5. Scope
6. Out of scope
7. Proposed approach
8. Files likely to change
9. Acceptance criteria
10. Test plan
11. Risk level
12. Rollback or safety notes
13. Dependencies
14. Notes for future follow-up if relevant

Tickets should be implementation-ready, but not over-prescriptive.

Do not create huge umbrella tickets where one ticket hides many unrelated changes.

Prefer small, independently reviewable tickets.

## Suggested ticket themes

Use these as candidate backlog items, but inspect the code first and adjust.

### Ticket theme 1: Add Mix Canvas scoring diagnostics

Purpose:

Make canvas selection explainable.

Possible content:

- optional debug or verbose output
- selected and rejected canvas summaries
- score breakdowns
- novelty and overlap penalties
- core / bridge / wildcard counts
- risk notes
- no noisy output in normal mode

### Ticket theme 2: Tighten Stage 2 prompt around canvas metadata

Purpose:

Ensure Sonnet uses canvas metadata correctly without over-trusting deterministic heuristics.

Possible content:

- role candidates are hints, not facts
- Sonnet may override role candidates
- bridge and wildcard tracks require justification
- wildcard use should be exceptional
- weak canvases may be rejected
- transition claims must distinguish known data from inferred assumptions

### Ticket theme 3: Improve warn-only DJ-structure validation

Purpose:

Move validation beyond technical sanity checks.

Possible content:

- missing opener role
- missing closer role
- no clear peak or payoff
- no clear resolution
- bridge/wildcard used in inappropriate role
- repeated role family warnings
- energy path mismatch warnings where energy data exists
- final closer not in closer candidate list

### Ticket theme 4: Store richer concept history for future tuning

Purpose:

Make concept history useful for diagnostics and future preference learning.

Possible content:

- store canvas score breakdowns
- store selected/rejected canvas metadata where useful
- store energy path candidates
- store role patterns
- store risk notes
- preserve compatibility with existing history file
- handle missing old fields safely

### Ticket theme 5: Add concept-shape novelty scoring

Purpose:

Prevent repeated concept shapes even when track overlap is low.

Possible content:

- deterministic similarity based on BPM band, key zone, role pattern, energy path, mood tags where available
- combine track-overlap penalty with concept-shape penalty
- keep weights conservative
- add debug output showing novelty reasons
- no embeddings in first version

### Ticket theme 6: Tune canvas scoring to avoid over-favouring large safe pools

Purpose:

Ensure smaller distinctive canvases can beat large generic pools.

Possible content:

- review current weights
- consider reducing pool-size reward
- use a core-pool size sweet spot rather than linear growth
- increase anchor strength and distinctiveness where appropriate
- add tests showing small distinctive canvas can win

### Ticket theme 7: Add bridge/wildcard concept-anchor candidates

Purpose:

Allow unusual outlier tracks to shape a concept when musically justified.

Possible content:

- identify structural exception candidates
- expose them in canvas metadata
- keep rules conservative
- require Stage 2 justification
- warning if used without justification

### Ticket theme 8: Document Mix Canvas architecture

Purpose:

Make the architecture easier to maintain.

Possible content:

- document pipeline flow
- document core/bridge/wildcard semantics
- document score meanings
- document concept history behaviour
- document validation philosophy
- document what Stage 2 is responsible for versus deterministic code

## Ticket sizing guidance

Break tickets down by safe implementation slices.

A good ticket should be completable in a focused session.

Avoid combining:

- prompt changes and scoring changes
- diagnostics and validation
- history schema changes and scoring changes
- bridge/wildcard anchor logic and concept-shape novelty

Each ticket should be independently useful.

## Required project-level backlog structure

Please create tickets in these phases unless code inspection suggests a better order.

### Phase 1: Trust and observability

Goal:

Make the current system easier to understand before changing behaviour too much.

Likely tickets:

- Mix Canvas scoring diagnostics
- Stage 2 prompt tightening
- Documentation of current Mix Canvas architecture

### Phase 2: Safe validation improvements

Goal:

Catch obvious DJ-structure issues without blocking successful runs.

Likely tickets:

- Warn-only DJ-structure validation
- Tests for validation warning noise and edge cases

### Phase 3: History and novelty tuning

Goal:

Make repeated runs more meaningfully different.

Likely tickets:

- richer concept history
- concept-shape novelty scoring
- debug output for novelty decisions

### Phase 4: Scoring and creative exception tuning

Goal:

Improve canvas selection quality while preserving safety.

Likely tickets:

- scoring weight tuning
- core-pool sweet spot
- bridge/wildcard concept-anchor candidates

### Phase 5: Future exploration only

Goal:

Capture larger ideas without committing to them yet.

Likely tickets should be labelled as future or discovery only:

- concept-first canvas seed exploration
- optional feedback loop from accepted/rejected concepts
- optional deeper audio/arrangement metadata
- optional playlist mode adaptation

## Things not to do yet

Do not create implementation tickets for these unless you explicitly mark them as future discovery:

- full concept-first redesign
- audio analysis pipeline
- embeddings-based similarity
- extra required Sonnet critic or repair calls
- reinforcement learning
- replacing the current two-stage LLM design
- changing playlist completion mode
- large schema/database migration
- major prompt rewrite that changes the identity of the generated reports

## Deliverables

At the end of the task, provide:

1. GitHub Project link or identifier
2. List of created issues with links
3. Brief summary of why the backlog is ordered that way
4. Any questions that require my product or DJ taste judgement
5. Any explicit non-goals or risks captured in the project
6. Any issues you chose not to create and why

## Acceptance criteria for this task

This task is complete when:

- a GitHub Project exists for the next phase
- the project contains a clear, prioritised backlog
- each ticket has detailed acceptance criteria and a test plan
- tickets are small and independently reviewable
- no code feature implementation has been started
- no new required premium LLM usage has been introduced
- playlist mode remains out of scope unless clearly justified
- the backlog is grounded in the actual code
- risky future ideas are separated from low-risk next work
- the working MixLab pipeline is protected

---

# Optional note to Claude

If GitHub project creation is not possible from the current environment, create the GitHub issues in the repository and produce a Markdown backlog document that can be used to create the project manually. Do not abandon the task because project creation is unavailable.

