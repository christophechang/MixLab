# MixLab flags guide

A tutorial for getting the most out of MixLab's levers — organised by what you're
trying to do, not alphabetically. For exact wording of any flag, `./mixlab --help`
is always current; this guide is about *when and why*.

## The 30-second mental model

Every generative run flows through the same pipeline:

```
Rekordbox XML → pool scoping (genre/mode/filters)
             → Stage 1 (deterministic BPM/key/era partitioning)
             → canvases + directions (candidate concept frames)
             → Stage 2 (LLM curation: selection, then prose reports)
             → validation (warn-only) → self-revision → sequencer
             → Discord + HTML report + Rekordbox XML export
```

Flags act at specific stations. Pool flags (`--genre`, `--mode`, `--min-bpm`…)
change what the machine sees. Shape flags (`--directions`, `--risk`, `--intent`)
change what it tries to build. Pipeline flags (`--deep`, `--no-revise`,
`--resequence`) change how hard it checks its own work. Utility modes (`--genres`,
`--prep`, `--feedback`, `--export-unplayed`) don't run the pipeline at all — no
LLM calls, no cost.

---

## First runs

```bash
./mixlab                      # crate availability table — what's in the tank, no LLM
./mixlab --genre house        # the standard run: 3–6 house concepts from unplayed tracks
```

The default run uses **unplayed tracks only** (played tracks are fetched from your
play-history API and excluded), blends classic BPM-coherent canvases with 2–3
creative *directions*, runs one self-revision pass on flagged concepts, and
delivers Discord messages plus a self-contained HTML report with booth sheets.

---

## Recipes

### "I'm playing Saturday and I know the vibe"

```bash
./mixlab --genre house --intent "sunset terrace, warm and melodic, ends euphoric but never hard" --mix-length 90
```

- `--intent` is free text that shapes concept moods, energy arcs, and selection.
  It works best with 10–50 words covering occasion, emotional destination, and
  stylistic scope. Frame positively ("dark and driven", not "not melodic").
- `--mix-length` scales track counts to a set length (90 min → ~22 tracks),
  using real track durations when your XML has them.

### "Surprise me" / "keep it safe"

```bash
./mixlab --genre house --risk high     # wildcards promoted to featured picks, bolder jumps tolerated
./mixlab --genre house --risk low      # adjacent keys, gentle BPM steps, effortlessly mixable
```

`--risk` (genre mode only) moves both the deterministic canvas scoring and the
Stage 2 framing. At `high`, transitions the model *explicitly annotates as risky*
get relaxed validator thresholds (20 BPM / 5 Camelot) — unannotated jumps are
still held to the default, so bold moves must be justified, never accidental.

### "Complete this playlist I started"

```bash
./mixlab --playlist "Monday Night"                    # seed + library additions
./mixlab --playlist "Monday Night" --genre electronica  # additions stay in one genre
./mixlab --playlist "Monday Night" --locked --mix-length 60  # trim my pool, add nothing
```

Playlist mode reads your seed playlist, infers a DJ-intent brief from it (Stage 0),
and produces three strategy variants (practical / balanced / adventurous), picking
a winner. `--locked` forbids additions — Stage 2 may only cut and reorder, which is
the move when you've over-collected and need it trimmed to a set length.
`--intent` also works here and overrides the inferred brief where they conflict.

Note which levers are **ignored in playlist mode**: `--directions`, `--risk`
(playlist mode derives risk from the intent brief), and `--deep`.

### "Take me across genres"

```bash
./mixlab --genre 4x4              # the 4/4 spectrum: house↔electronica↔disco↔progressive↔techno
./mixlab --genre 170              # hardcore continuum: DnB + jungle at 165–175
./mixlab --genre 140              # UK underground: breaks + UK bass + UKG at 130–140
./mixlab --genre traverse         # whole collection — unlocks genre-traverse journeys
```

Custom genre labels pool multiple genres. `traverse` exists specifically to feed
the **genre traverse** direction: journey concepts that cross tempo regimes
(house → UKG → jungle/DnB) using pitch-locked ratio bridges — halftime,
double-time, 3:4, 4:3 — with each verified bridge pair named in the brief. The
direction only fires when your material genuinely splits into bridgeable regimes.

### "More ideas" vs "no gimmicks"

```bash
./mixlab --genre house --directions only    # creative briefs only (testing/exploration)
./mixlab --genre house --directions off     # classic BPM-coherent canvases only
```

Directions (default `mixed`) are cross-strata creative briefs — mood journeys,
era dialogues, label spotlights, artist threads, energy shapes, fresh-crate
showcases, genre traverses — proposed only when the library supports them and
rotated by the daily seed. `only` forces them to the front, which is the right
way to test whether a specific direction (like a traverse) fires for your pool.

### Narrowing the pool

```bash
./mixlab --genre house --mode played              # battle-tested tracks only
./mixlab --genre 4x4 --mode all                   # ignore play history entirely
./mixlab --genre house --min-bpm 122 --max-bpm 128
./mixlab --genre drum_and_bass --min-year 2020    # note: undated tracks are excluded
```

`--mode` picks the play-history slice (`unplayed` is the default — discovery is
the product's core loop). BPM/year filters apply straight after ingestion, before
anything else sees the pool.

### The cue-prep loop (this is the compounding one)

```bash
./mixlab --prep                        # rank every uncued track by expected payoff
./mixlab --prep --genre house --top 10
```

Offline, no LLM. Scores each track with missing cue data by how often your
concept history programs it, how harmonically central it is in its genre, and
whether it's an unplayed debut candidate. Cue the top entries in Rekordbox
(mix-in at position one, mix-out at the end — your conventions), re-export the
XML, and the next run's **booth sheets** gain clock times, pitch percentages, and
green tiers where there were scout warnings. Every cue you add makes every future
report smarter.

### Closing the feedback loop

```bash
./mixlab --feedback                                            # list recent concepts + state
./mixlab --feedback --concept "Night Bus N29" --verdict played --notes "peak section landed"
./mixlab --feedback --concept "Pillar Logic" --verdict rejected
```

Verdicts feed novelty scoring: `played` concepts make similar future concepts
*less* likely to repeat (×1.5 similarity penalty — you've done it), `rejected`
ones fade fast (×0.25). Recording even a few verdicts per week measurably changes
what the engine offers you.

### Trust and verification

```bash
./mixlab --genre house --deep          # peer-DJ critique per concept (2× Stage 2 cost)
./mixlab --genre house --no-revise     # skip the automatic repair pass
./mixlab --genre house --resequence    # let the sequencer's order improvements apply
```

By default a concept with 2+ hard validation findings gets one bounded
self-revision (minimal swap/reorder/drop; accepted only if it strictly reduces
findings, with its prose regenerated to match). `--no-revise` turns that off when
you want the raw first take. The deterministic sequencer always *suggests* order
improvements in the report; `--resequence` makes it apply them to the export.

### Reproducing and debugging

```bash
./mixlab --genre house --stage1-seed 20260707   # replay a prior day's exact run
./mixlab --genre house --debug                  # canvas scoring diagnostics on stderr
```

The seed (printed at every run start) drives Stage 1 windowing, direction
rotation, and naming-lens selection. Same seed + same library + same history →
the same deterministic skeleton, so a surprising run can be replayed and studied.

### Exports

```bash
./mixlab --genre house --export ~/Desktop     # merged Rekordbox XML to a directory
./mixlab --genre house --export-playlists     # same, to output/playlists/
./mixlab --export-unplayed                    # all unplayed tracks as XML + Discord (no LLM)
```

The merged XML carries one playlist per concept (plus an All Unplayed Tunes
playlist when history was used) — import it into Rekordbox and the night's
options are sitting there as playlists.

---

## Quick reference

| Flag | Mode | What it does |
|---|---|---|
| `--genre LABEL` | both | Pool scope: standard genre, custom (`170`/`140`/`4x4`/`traverse`), or raw Rekordbox tag |
| `--playlist NAME` | playlist | Complete a seed playlist into a set |
| `--mode unplayed\|all\|played` | both | Play-history slice (default unplayed) |
| `--min-bpm` / `--max-bpm` / `--min-year` / `--max-year` | both | Hard pool filters |
| `--intent TEXT` | both | Free-text creative direction for Stage 2 |
| `--mix-length MIN` | both | Scale track count to a set length |
| `--locked` | playlist | No library additions — trim/reorder only |
| `--directions mixed\|off\|only` | genre | Blend/exclude/force creative direction concepts |
| `--risk low\|medium\|high` | genre | Safety ↔ novelty knob for scoring, prompts, and validators |
| `--deep` | genre | Opt-in self-critique pass (2× Stage 2 cost) |
| `--no-revise` | genre | Skip the bounded self-repair pass |
| `--resequence` | genre | Apply (not just suggest) sequencer order improvements |
| `--stage1-seed N` | genre | Reproduce a run's deterministic skeleton |
| `--debug` | genre | Verbose scoring diagnostics |
| `--export DIR` / `--export-playlists` | both | Merged Rekordbox XML export |
| `--genres` | utility | Availability table from cache (no network) |
| `--export-unplayed` | utility | Unplayed-tracks XML + Discord, no LLM |
| `--prep` (+ `--top N`) | utility | Cue-prep payoff ranking, offline |
| `--feedback` (+ `--concept`, `--verdict`, `--notes`) | utility | Record concept verdicts into history |

## How flags compose

- **Pool flags stack**: `--genre 4x4 --mode played --min-bpm 120 --max-year 2015`
  is a valid, very specific pool.
- **Shape flags compose**: `--risk high --directions only --intent "..."` all
  apply simultaneously in genre mode.
- **Playlist mode wins**: when `--playlist` is present, genre-mode-only flags
  (`--directions`, `--risk`, `--deep`, `--stage1-seed`) are ignored; `--genre`
  changes meaning to "constrain added tracks to this scope".
- **Utility modes short-circuit**: `--genres`, `--prep`, `--feedback`, and
  `--export-unplayed` run their task and exit — no pipeline, no LLM cost, safe to
  run any time.
