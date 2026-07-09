# MixLab

[![GitHub release](https://img.shields.io/github/v/release/christophechang/MixLab)](https://github.com/christophechang/MixLab/releases)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

See [CHANGELOG.md](CHANGELOG.md) for version history.

> **Point it at your collection. Get a set worth playing.**

AI-powered DJ crate assistant. Point it at your Rekordbox collection, pick a genre, and get a set of ready-to-use mix concepts with Camelot-ordered track listings — delivered to Discord.

<!-- demo screenshot or terminal recording goes here -->

For background on why this was built and how it works in practice, read the [MixLab case study](https://www.soltechconsulting.co.uk/case-studies/mixlab).

> This project explores AI-assisted development workflows. My focus here was system design and delivery rather than idiomatic Python, which is not my primary stack.

---

## Quickstart

```bash
git clone https://github.com/christophechang/MixLab.git && cd MixLab
./setup.sh
cp .env.example .env   # fill in ANTHROPIC_API_KEY + at least one Stage 1 key

./mixlab               # crate availability table — no LLM cost
./mixlab --genre house # full mix report, delivered to Discord
```

---

## How it works

1. Parses your exported Rekordbox XML collection
2. If `CATALOG_API_URL` is set, fetches your play history and applies `--mode` filtering (`unplayed` by default, or `played` to restrict to battle-tested tracks); without the catalog API the full collection is used
3. Prints a crate availability table (no LLM cost)
4. If `--genre` is specified, scopes the collection to that genre (or custom cross-genre pool), runs Stage 1 shortlisting, wraps each shortlist into a Mix Canvas (BPM tiers, role candidates, contrast assets, anchors, era/label coherence, risk notes), optionally blends in cross-strata concept directions (`--directions`), then writes a full Stage 2 mix planning report — optionally steered with `--intent "..."`
5. If `--playlist` is specified, uses that Rekordbox playlist as the seed, infers the set's intent, builds natural BPM-zone shortlists around the seed tracks, generates three completion variants, then writes the best playlist-completion report
6. Optionally exports a Rekordbox-compatible merged XML file
7. Sends the report and any XML attachment to a Discord channel

---

## Requirements

- Python 3.12+
- A Rekordbox XML export (see [Setup](#4-export-your-rekordbox-collection))
- `ANTHROPIC_API_KEY` — required for default Stage 2 report generation
- At least one Stage 1 LLM key (Groq, Gemini, or Mistral)
- A catalog API URL + key (optional — for filtering already-played tracks)
- A Discord bot token (optional — report prints to stdout without it)

---

## Setup

### 1. Clone and run setup

```bash
git clone https://github.com/christophechang/MixLab.git
cd MixLab
./setup.sh
```

`setup.sh` creates the virtual environment, installs all dependencies, and copies `.env.example` to `.env` if it doesn't already exist.

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Default Stage 2 report generation (Claude Sonnet) |
| `CATALOG_API_URL` | No | Base URL of your catalog/play-history API |
| `CHANGSTA_API_KEY` | No | Bearer token for `CATALOG_API_URL` (if your API requires auth) |
| `GROQ_API_KEY` | No | Stage 1 provider (tried first) |
| `GEMINI_API_KEY` | No | Stage 1 provider (fallback 1) |
| `MISTRAL_API_KEY` | No | Stage 1 provider (fallback 2) |
| `DISCORD_BOT_TOKEN` | No | Discord delivery |
| `DISCORD_GUILD_ID` | No | Discord server ID |
| `MIXLAB_DISCORD_CHANNEL_ID` | No | Target channel ID (preferred over name) |
| `MIXLAB_DISCORD_CHANNEL` | No | Target channel name (default: `mix-lab`) |
| `MIXLAB_STAGE2_MODEL` | No | Override the Stage 2 Anthropic model id (default: `claude-sonnet-4-6`) |
| `MIXLAB_STAGE2_TEMPERATURE` | No | Override the Stage 2 selection-pass temperature (default: `0.5`) |

`ANTHROPIC_API_KEY` is required for Stage 2 report generation. Without a catalog API URL, played-track exclusion is skipped and the full collection is used. Without a Discord token the report is printed to stdout only. `MIXLAB_STAGE2_MODEL` and `MIXLAB_STAGE2_TEMPERATURE` let you point Stage 2 at a different Anthropic model or tune selection-pass creativity without editing code — both fall back to the current defaults when unset or invalid.

#### Catalog API (optional)

The catalog API is used to fetch your play history and exclude already-played tracks from recommendations. Without it, every track in your collection is treated as unplayed — you still get full mix concepts, but the tool cannot distinguish tracks you've played before from ones you haven't. The unplayed count in the crate table will equal the total collection count.

The catalog API MixLab integrates with is also open source: [soundcloud-ai-mix-recommender-api](https://github.com/christophechang/soundcloud-ai-mix-recommender-api). Set `CATALOG_API_URL` to your deployed instance to enable played-track exclusion.

### 3. Enrich your Rekordbox collection (recommended)

MixLab works from your Rekordbox XML export. The core fields — BPM and Camelot key — are required and must be set for every track. The following enrichment layers are optional but significantly improve the quality of the AI-generated concepts.

#### Camelot keys (required)

Every track must have a Camelot key set in Rekordbox (`Tonality` field). The easiest way to populate these at scale is **Mixed In Key** — it analyses your files and writes Camelot keys directly into Rekordbox.

#### Mixed In Key — energy scores and tags (strongly recommended)

[Mixed In Key](https://mixedinkey.com) writes two additional pieces of data into the Rekordbox `Comments` field that MixLab reads:

| Data | Format in Comments | What MixLab does with it |
|---|---|---|
| Energy score | `Energy 7` (1–8 scale) | Passed to the AI to inform the energy arc and peak placement |
| Genre/mood tags | `/* Deep House / Soulful / Melodic */` | Passed to the AI as track character descriptors |

To populate these, run Mixed In Key on your collection, enable **Write to Rekordbox**, and let it analyse. After analysis completes, re-export your Rekordbox XML.

Tracks without energy scores or tags still work — MixLab reasons from BPM, key, genre, and artist knowledge when supplementary data is absent.

#### Track colours — enrichment confidence

MixLab reads the Rekordbox `Colour` field as a match-confidence signal set by the enrichment pipeline:

| Colour | Confidence | Meaning |
|---|---|---|
| Green | High (≥ 0.85) | Auto-matched, safe to use |
| Orange | Medium (0.65–0.85) | LLM-assisted, worth a glance |
| Red | Low (< 0.65) | Heuristic label — inspect before relying on |
| Blank | None | No match found |

Tracks flagged low-confidence are marked `[unverified]` in prompts sent to the AI. You do not need to set colours manually — the enrichment project populates them.

#### The `/* tags */` block — an example tagging setup

MixLab reads whatever is inside the `/* ... */` block in the Comments field and passes it verbatim to the AI as track descriptors. There is no fixed schema — it is plain text and you can structure it however you like. The AI will reason from whatever you put there.

The following is the tagging structure used in this project, shared as a concrete example rather than a prescription.

**Tagging layers**

Tracks are described across four independent layers:

| Layer | Where | What it captures |
|---|---|---|
| Genre | Rekordbox Genre field | Primary genre (House, Drum & Bass, etc.) |
| Playlists | Rekordbox Playlists | Tracks used in a SoundCloud mix — "battle-tested" markers |
| Energy | MIK energy score in Comments | Intensity level on a 1–8 scale |
| Mood | `/* */` block in Comments | Vibe, character, and feel of the track |

**Mood tag vocabulary**

The mood tags in this collection are a vibe-based layer that sits on top of genre and energy — describing not what a track *is*, but what feeling it brings to a mix:

`acid` · `aggressive` · `big` · `brooding` · `builder` · `carnival` · `dark` · `dirty` · `driving` · `druggy` · `dreamy` · `emotional` · `energetic` · `funky` · `grimy` · `happy` · `heavy` · `in the groove` · `Latino` · `melancholic` · `old skool` · `raga` · `rave` · `soulful` · `street` · `trippy` · `vocal`

A typical Comments field with this setup looks like:

```
8A - Energy 7 /* Drum & Bass / dark / driving */
```

MIK writes the key and energy score; the `/* */` block is added manually in Rekordbox. You can use as many or as few tags as feel useful — the AI treats them as hints, not hard rules.

#### Record label (optional)

The `Label` field in Rekordbox is passed to the AI as context. Useful if your collection is tagged by label and you want the AI to reason about label character (e.g. Defected, Nervous, Peacefrog).

#### Summary — what to do before your first run

1. Analyse your collection with Mixed In Key and write Camelot keys, energy scores, and tags back to Rekordbox
2. Optionally run the enrichment pipeline — it sets track colours as confidence signals
3. Export your collection: **File → Export Collection in xml format**
4. Move the file to `import/rekordbox.xml`
5. Re-export after any library changes (new tracks, updated tags)

### 4. Export your Rekordbox collection

> **Rekordbox 6:** XML export must be enabled before it appears in the menu. Go to **Preferences → Advanced → rekordbox xml** and tick **Export rekordbox xml**. Restart Rekordbox if needed.

In Rekordbox:

1. Go to **File → Export Collection in xml format**
2. Choose a save location and export
3. Move or copy the exported file to `import/rekordbox.xml` in the project root

> **Note:** Rekordbox exports your full COLLECTION — all tracks, not just a playlist. Tracks not tagged to any playlist will still appear in the XML (MixLab uses genre tags, not playlist membership). SoundCloud cloud tracks are excluded automatically. Re-export whenever you add new tracks.

---

## Running MixLab

> **New to the levers?** [`docs/flags-guide.md`](docs/flags-guide.md) is a tutorial covering every flag by use case — gig prep, playlist completion, cross-genre journeys, the cue-prep loop, feedback verdicts, and how flags compose — with copy-paste examples for each. The sections below cover the core workflows; the guide covers everything.

### View crate availability (no LLM calls)

```bash
./mixlab
```

Prints unplayed vs total counts per genre, sorted by availability. Only the catalog API is called (if configured). Without a catalog API, shows total collection counts.

### Generate a mix report for a specific genre

```bash
./mixlab --genre house
./mixlab --genre house --mode played
./mixlab --genre 4x4 --mode all
```

Runs the full genre pipeline: parse → fetch played history (based on `--mode`) → scope to the requested genre → Stage 1 shortlist generation → Stage 2 report → Discord/stdout.

`--mode` controls which tracks are eligible for concepts. Each mode also tunes the canvas scoring weights (boost novelty for `unplayed`, anchor strength for `played`, cross-canvas distinctiveness for `all`) and adds mode-specific creative direction to the Stage 2 prompt:

| Mode | Behaviour |
|------|-----------|
| `unplayed` (default) | Only tracks never played live. Stage 2 framed as discovery — surface debuts worth introducing. Requires `CATALOG_API_URL`. |
| `played` | Only tracks that have appeared in your play history — battle-tested and SoundCloud-proven. Stage 2 framed as reassembly — bolder Camelot jumps and chapter pivots since familiarity is an asset. Requires `CATALOG_API_URL`. |
| `all` | Full collection, ignoring play history entirely. Stage 2 framed as interleave — concepts that combine played and unplayed material in deliberate ways; notes the lean (played-anchored / unplayed-anchored / balanced) in the thesis. |

The report starts with a context header so you can see exactly what kind of run produced it, for example:

```text
Report context: House (unplayed tracks)
Report context: House (played tracks)
Report context: 140 (custom genre, All Tracks)
```

For standard and custom genre runs, a Rekordbox-compatible merged XML file can be attached to the Discord message or written to disk. It contains one playlist per concept plus an **All Unplayed Tunes** playlist with the full scoped unplayed pool when played-track history was used.

### HTML report

Every genre and playlist run also writes a standalone HTML report to `output/reports/` (override the directory with the `MIXLAB_REPORT_DIR` environment variable) and attaches it to the Discord message. The file is fully self-contained and offline — inline CSS and JS, a system font stack, no images, CDNs, fonts, or any external request — so it opens straight from disk and renders identically in light and dark mode. Each concept gets a card carrying everything the text report holds (title, arc type, mood, name reason, track table, and the full prose) plus computed transition intelligence: an energy sparkline across the set, per-transition mechanism labels (halftime locks, energy lifts), blend headroom, and a colour-coded mixability score for each consecutive pair. Click any track row to copy its `artist — title`. The rendered output is deterministic for identical inputs.

#### Booth sheet

Each concept card carries a booth sheet: a per-transition execution plan computed deterministically from your cue points and beat grids — no LLM involved. Each step shows: the clock position of the outgoing track's mix-out cue (`Open the blend at 5:43`), the pitch-fader move (halftime/double-time aware), key move, bars available each side, a plan line in booth language, a fallback when the outro allows one, and colour coding (green = relaxed window, amber = tight, red = hard commit). Scout notes flag tracks with missing cue data — they double as a cue-prep to-do list for Rekordbox.

### Steer genre mode with a free-text intent

```bash
./mixlab --genre house --intent "warmup set for an outdoor afternoon, low pressure, melodic"
./mixlab --genre 4x4 --intent "peak time main-room, no warmup, hit hard fast"
./mixlab --genre techno --mode played --intent "tools-only set, no melody, sustained pressure"
```

`--intent` accepts a free-text creative direction that is injected verbatim into the Stage 2 prompt. There is no parsing or LLM extraction beyond a light heuristic signal pass — the model reads it as guidance and fills in everything you did not specify. If the intent conflicts with the candidate pool, Stage 2 picks the closest viable interpretation and notes the gap.

`--intent` also works in playlist mode (`--playlist`), which already runs its own Stage 0 intent-extraction pass over the seed playlist. There, `--intent` is layered on top of the inferred DJ intent brief and overrides it wherever the two conflict — the report's Assumptions section names any such conflict. If the intent text contains a risk-tolerance cue (e.g. "safe and cautious" or "surprise me, go bold"), it also overrides the Stage 0-inferred `risk_tolerance` used for winner selection (see below), and MixLab prints a note on stderr when this happens.

### Concept directions (`--directions`)

Classic Stage 1 slices a genre pool into BPM strata, so tempo is the only axis a concept can be built around. Concept directions add a second, cross-strata axis: deterministic creative briefs that deliberately span BPM tiers. Six direction types are enumerated over the whole genre-scoped pool, and only the ones the material actually supports are proposed:

- **mood_journey** — travels between contrasting mood-tag poles (e.g. dark → euphoric), bridged by neutral tracks
- **era_dialogue** — old-vs-new conversation across a year gap, eras alternated deliberately
- **label_spotlight** — one label's scene DNA, optionally braced by a few harmonically-adjacent outsiders
- **artist_thread** — one artist/remixer (2–3 tracks) as the structural spine; their tracks are chapter markers
- **energy_shape_first** — the pool is balanced across energy bands to realise a declared arc (wave, double-peak, dark-to-light)
- **fresh_crate** — a debut showcase of the newest additions, grounded by a couple of anchor tracks

Each proposed direction is feasibility-scored (pool fill, BPM-path viability, and a type-specific signal strength). Every direction requires a BPM-feasible path, so briefs that cannot actually be mixed are dropped. Feasible directions are then seed-rotated: the same seed reproduces the same picks, but different days (the seed defaults to the date, reproducible via `--stage1-seed`) surface different angles while the strongest directions still appear often. Each surviving direction becomes a Mix Canvas carrying a DIRECTION BRIEF that Stage 2 must honour as the concept's thesis.

```sh
./mixlab --genre house                    # mixed (default): directions blended with classic canvases
./mixlab --genre house --directions off   # classic BPM-stratum canvases only
./mixlab --genre house --directions only  # directions only (falls back to classic if none are feasible)
```

`--directions` is genre mode only and is ignored (with a stderr note) in playlist mode.

### Risk knob (`--risk`)

`--risk {low,medium,high}` (default `medium`) trades safety for novelty in genre-mode runs, on top of whatever `--directions` picks. It shifts both canvas scoring and the Stage 2 prompt framing:

- `high` reweights canvas selection toward `contrast_potential` and `novelty` (away from `role_coverage`/`anchor_strength`), nudges Stage 2 to feature flagged wildcard/concept-anchor tracks rather than treat them as exceptions, and relaxes the post-Stage-2 validator's jump thresholds to 20 BPM / 5 Camelot positions — but only for transitions the model has explicitly annotated `is_risky`; an unannotated jump is still held to the medium thresholds, so a bold move still requires Stage 2 to name the mechanism that makes it survivable.
- `low` reweights canvas selection toward `role_coverage` and `anchor_strength` (away from `contrast_potential`/`novelty`), nudges Stage 2 toward core tracks and gentle moves, and tightens the validator thresholds to 10 BPM / 3 Camelot positions.
- `medium` (default) is unchanged from prior behaviour — same weights, same prompt, same 15 BPM / 4 Camelot thresholds.

```sh
./mixlab --genre house --risk high   # promote wildcards/anchors, relax annotated-jump thresholds
./mixlab --genre house --risk low    # favour role-complete, low-risk canvases
```

`--risk` composes with `--directions`: it changes how canvases are scored and validated regardless of whether they came from classic BPM strata or a concept direction. It is genre mode only — playlist mode derives its own risk tolerance from the Stage 0 intent brief (see `--intent` above) and prints a stderr note if `--risk` is passed alongside `--playlist`.

### Mix Engine (genre mode)

When your Rekordbox export carries a beat grid and memory/hot cues, MixLab parses each track's structure into mix points (intro/outro bar counts, loop zones, cue count) and uses them across genre mode:

- **Owner cue conventions.** The first cue is read as the mix-in; the last cue is read as the mix-out when a track has at least two cues in the back half of the arrangement. Tracks with short outros get manual-loop language ("cut or manual loop likely") rather than being called unmixable, and cueless tracks stay neutral — absence of cue data never penalises a track.
- **Blend warnings.** The post-Stage-2 validator flags any consecutive pair whose outro/intro headroom is too tight to ride (unless the transition is already annotated as a justified risk). These count as hard findings and can trigger the self-revision pass.
- **Practicality blend component.** When enough of a concept's transitions carry cue data, the per-concept practicality line gains a `blend_feasibility` term and rebalances its weights to include it; cueless concepts keep the original formula unchanged.
- **Intro/outro prompt tokens.** Stage 2 candidate and report lines show `intro:16b/outro:32b` tokens so the model can reason about workable mix windows directly.
- **`--resequence`.** By default the deterministic sequencer only *suggests* order improvements in a `**Sequencer**` report block and leaves the exported order untouched. Pass `--resequence` to apply the suggested swaps to the exported concepts (opener and closer are never moved).

```sh
./mixlab --genre house --resequence   # apply the sequencer's suggested order swaps
```

### Complete a mix from an existing Rekordbox playlist

```bash
./mixlab --playlist "Monday Night"
./mixlab --playlist "Monday Night" --genre electronica
./mixlab --playlist "Sets/Monday Night"
./mixlab --playlist "Monday Night" --mode all
./mixlab --playlist "Monday Night" --mode played
./mixlab --playlist "Monday Night" --mix-length 60   # target ~15 tracks for a 1-hour set
./mixlab --playlist "Monday Night" --mix-length 90   # target ~22 tracks for a 90-minute set
```

Playlist mode is a different workflow from genre mode:

- The source Rekordbox playlist is treated as the seed and MixLab aims to complete or extend it, not replace it
- MixLab first runs an intent-analysis pass over the seed playlist to infer the overall vibe, energy shape, anchor tracks, and any missing set roles
- Seed tracks are clustered into natural BPM zones
- Each zone becomes a shortlist containing the seed tracks for that zone plus nearby library tracks
- Stage 2 generates exactly three completion variants (`practical`, `balanced`, and `adventurous`) and MixLab auto-selects the strongest one
- The final report explains which seed tracks were retained, which were dropped, which library tracks were added, and which alternative strategy was rejected

`--mix-length <minutes>` works in **both genre and playlist mode** and scales the number of tracks Stage 2 selects. The target is derived from the real durations of the candidate pool (`TotalTime` from the Rekordbox XML) — 60 minutes of ~4-minute house targets ~15 tracks, 60 minutes of ~6-minute progressive targets ~10. When no track in the pool carries a duration, it falls back to the old `max(10, round(minutes / 4))` heuristic. Without the flag, playlist mode targets 10–14 tracks and genre mode uses per-genre targets. Arc quality takes priority: Stage 2 will cut weak tracks rather than padding to hit the count. Reports show per-track durations and a `Runtime: ~NNm` footer whenever duration data exists.

Important playlist-mode rules:

- `--genre` in playlist mode constrains added library tracks to that genre scope; it does not filter the seed playlist itself
- `--mode` controls which library tracks are candidates: `unplayed` (default) biases towards unplayed tracks, `played` restricts the pool to battle-tested tracks only, `all` uses the full collection with no weighting
- Playlist names are matched case-insensitively
- If the same playlist name exists in multiple folders, pass the full path such as `Sets/Monday Night`
- Playlist mode requires at least 4 valid seed tracks with BPM and Camelot key after parsing

Winner selection is tolerance-aware: each variant's `fit` score blends its DJ Practicality Score with an "adventure dividend" that rewards a high density of justified risky transitions (a real mechanism named, not a bare cut), weighted by the run's `risk_tolerance` (`low` → 100% practicality / 0% adventure, `medium` → 80/20, `high` → 60/40). At `low` tolerance this reduces to plain practicality — the same ranking as before. At `high` tolerance the tie-break order also inverts (`adventurous` > `balanced` > `practical`), so a DJ who explicitly asked for adventure gets it when variants are otherwise close. The report's rejected-alternatives line records which tolerance was used for the run, e.g. `Selection tolerance: medium.`

Playlist runs use the same report context header as genre runs, for example:

```text
Report context: Monday Night playlist (Electronica, unplayed tracks)
Report context: Monday Night playlist (played tracks)
Report context: Monday Night playlist (Electronica, All Tracks)
```

Playlist runs also print a compact intent summary before the final report, for example:

```text
Intent brief: Deep, rolling warm-up with a late lift | energy: single_arc | risk: medium | anchors: 3 | missing roles: peak
```

If you export playlist mode, the merged XML contains the single completed concept only; it does not add an **All Unplayed Tunes** playlist.

To also write the XML to disk:

```bash
./mixlab --genre house --export-playlists
# writes to output/playlists/rekordbox_export.xml

./mixlab --genre house --export /path/to/dir
# writes to /path/to/dir/rekordbox_export.xml
```

### Narrow the candidate pool by BPM or year

```bash
./mixlab --genre house --min-bpm 122 --max-bpm 128
./mixlab --genre drum_and_bass --min-year 2020
./mixlab --genre 4x4 --max-year 2019
./mixlab --playlist "Monday Night" --min-bpm 130 --max-bpm 138
```

BPM and year filters apply after ingestion and BPM correction. In playlist mode they apply only to library additions — seed tracks are never filtered out. Tracks with no release year set are excluded when either year flag is active. Active filters appear in the Discord crate snapshot label.

### Export all unplayed tracks to Rekordbox

```bash
./mixlab --export-unplayed
```

Compares your full Rekordbox collection against your play history and exports every track you haven't played yet as a Rekordbox-compatible merged XML file. Use this when you want a complete picture of what's in your crates that you haven't touched — import the file into Rekordbox to browse, filter, or prepare for a session.

- Writes to `output/playlists/rekordbox_export.xml` with a dated folder name (`Mix Lab - All Unplayed - YYYY-MM-DD`)
- Posts a summary and the XML attachment to Discord
- Requires `CATALOG_API_URL` — without play history there is nothing to compare against
- Respects the `DO NOT RECOMMEND` exclusion list
- No LLM calls — fast and cheap

### Inspect canvas scoring with --debug

```bash
./mixlab --genre house --debug
# or: MIXLAB_DEBUG_SCORE=1 ./mixlab --genre house
```

Emits per-canvas scoring diagnostics to stderr: every weighted component, weakness penalty, floor multiplier, overlap penalty against already-picked canvases, novelty breakdown (track-overlap component + shape-similarity component + closest history match), era/label coherence values, and risk notes. Normal stdout output and Discord delivery are unchanged.

### Record feedback on generated concepts

```bash
./mixlab --feedback                                              # list the most recent run's concepts
./mixlab --feedback --concept "Ladbroke to Kaoz" --verdict played
./mixlab --feedback --concept "Glass Crate" --verdict rejected --notes "peak section didn't land"
```

Every run stores its concepts in `.mixlab/concept-history.json`. `--feedback` lets you tell MixLab what actually happened to them: `played`, `played_modified`, `rejected`, or `unused`. Verdicts feed straight into novelty scoring on future runs — a concept you **played** penalises similar candidates ~1.5× harder (you've used that idea), while a **rejected** concept's penalty is muted to 0.25× (you said no; don't let it block fresh attempts). Concepts are matched by case-insensitive title or ID prefix against the most recent run containing them; no LLM or network calls.

### View cached genre counts from the last run (no API calls at all)

```bash
./mixlab --genres
```

### Rank cue-prep targets (Cue-Prep Assistant)

```bash
./mixlab --prep
./mixlab --prep --genre house --top 10
```

Ranks every track with missing or partial cue-point data (no `mix_points` at all, or missing its mix-out point) by how much cueing it would pay off. The score blends: demand (how often concept history has already programmed the track), harmonic centrality (how well-connected it is to its genre-bucket peers via transition scoring), unplayed status, and gap severity (a fully uncued track outweighs one that's only missing its mix-out point). `--genre <label>` scopes to one standard genre label (custom pools like `170`/`140`/`4x4` and raw Rekordbox tags are rejected — standard `GENRE_MAP` labels only); `--top N` caps the number of rows shown (default 20). Fully offline: no API calls, no LLM, no Discord post.

```text
    #  Track                                      Bucket            BPM  Key  Gap          Score  Reason
    1  Overmono — So U Kno                         house          126.0  8A   uncued         5.50  in 2 planned concepts · unplayed · fully uncued
    2  Or:la — Rebound                              house          124.0  9A   no-mix-out     3.20  harmonically central · missing mix-out cue
    3  Peach — Fabric 92 Intro                      house          122.0  8B   uncued         2.50  unplayed · fully uncued

  house: 128 of 370 tracks lack cue data

Cue up the top entries in Rekordbox, re-export, and booth sheets gain clock times.
```

### MixLab Anywhere worker

Run MixLab as a remote worker that pulls queued runs from the MixLab Anywhere API, executes the normal pipeline as a subprocess, and uploads report + summary artifacts:

```bash
./mixlab --worker                 # persistent loop, polls API every 30s
./mixlab --worker-once            # one cycle, then exit (for cron/manual)
```

The worker mode is purely additive — your normal CLI workflow (`./mixlab --genre house`, etc.) is unchanged. All other pipeline flags are ignored when `--worker` is active; the run's flags come from the queued manifest instead.

**Required environment variables:**

- `MIXLAB_API_URL` — base URL of the MixLab Anywhere API (e.g. `https://api.example.com`)
- `MIXLAB_API_SECRET` — shared bearer secret for API authentication

**Optional variables:**

- `MIXLAB_WORKER_POLL_SECONDS` (default `30`) — empty-poll cadence in seconds
- `MIXLAB_WORKER_XML_PATH` (default `.mixlab/worker-collection.xml`) — where the worker writes each run's downloaded collection. Deliberately off `import/rekordbox.xml` so a remote run never overwrites the collection a local run uses; the worker points the pipeline at this path via `MIXLAB_COLLECTION_PATH`.
- `MIXLAB_WORKER_RUN_TIMEOUT` (default `1200`) — hard per-run subprocess timeout in seconds

The worker also syncs concept history and feedback with the API: it pulls pending feedback events, applies them to the local history, and pushes back any changes — bidirectional sync so verdicts and ratings you record in the web UI feed into future runs.

For production setup on macOS, see [`docs/ops/worker-launchd.md`](docs/ops/worker-launchd.md). Architecture and failure policies: [`docs/architecture/mixlab-anywhere.md`](docs/architecture/mixlab-anywhere.md).

---

## Available genres

### Standard genres

Pass the label (left column) to `--genre`. The right column shows the Rekordbox genre tags that map to it.

| Label | Rekordbox genre tags |
|---|---|
| `house` | House, Deep House, Tech House, Classic House, Afro House, Minimal / Deep Tech |
| `drum_and_bass` | Drum & Bass, DnB, Liquid DnB, Jungle/Drum'n'bass |
| `breakbeat` | Breakbeat, Breaks, Nu Skool Breaks, Hardcore |
| `electronica` | Electronica, Electronic, Downtempo, Trip Hop |
| `hip_hop` | Hip Hop, Funk, Hip Hop/Rap, Soul/Funk/Jazz |
| `jungle` | Jungle, Ragga Jungle, Rave |
| `uk_bass` | UK Bass |
| `progressive` | Progressive |
| `disco` | Disco |
| `techno` | Techno, Dark Techno, Industrial Techno, Dub Techno, Melodic House & Techno |
| `uk_garage` | UK Garage, UKG, 2-Step, UK Garage / Bassline |

You can also pass a Rekordbox genre tag directly (case-insensitive), e.g. `--genre "Deep House"`.

### Custom genres

Custom genres merge multiple standard genres into a single pool. They are designed for cross-genre sets where the interesting DJ work happens at the boundaries — moving from one sound to another and making it feel intentional.

```bash
./mixlab --genre 4x4
./mixlab --genre 170
./mixlab --genre 140
./mixlab --genre traverse
```

| Label | Sub-genres | BPM range | Rationale |
|---|---|---|---|
| `170` | drum_and_bass + jungle | 165–175 BPM | Hardcore continuum genres that live at the same tempo and share rhythmic DNA — the richest cross-genre territory in the collection |
| `140` | breakbeat + uk_bass + uk_garage | 130–140 BPM | UK underground genres that occupy the same tempo bracket; blends can range from technical to percussive to bass-heavy |
| `4x4` | house + electronica + disco + progressive + techno | none | The full 4/4 spectrum from deep house to techno, with a wide BPM range (~90–140); the creative challenge is the journey across that arc |
| `traverse` | all standard genres | none | Full-collection pool that unlocks the **genre traverse** direction: journey concepts that cross tempo regimes (house → UKG → jungle/DnB) via pitch-locked ratio bridges — halftime, double-time, 3:4 and 4:3 blends. The direction only fires when the pool actually splits into bridgeable regimes; each verified bridge pair is named in the Stage 2 brief with its mechanism |

Custom genres behave differently from standard genres in two key ways:

**1. BPM filtering.** `170` and `140` apply a hard BPM range filter — tracks outside those bounds are excluded before Stage 1. The range is part of what defines the genre. `4x4` has no hard BPM filter: the sub-genres span a wide range (~90–140 BPM) and Stage 1 is responsible for finding BPM-coherent groupings within the pool.

**2. Stage 2 cross-genre guidance.** The Stage 2 prompt is given the list of sub-genres and instructed to justify any move across genre boundaries — naming the specific mechanism that makes the transition work (BPM alignment, rhythmic character, harmonic relationship, or the energy state of the room). Cross-genre moves are not avoided; they are the point of using a custom genre. But every such move must be defensible.

#### How large pools are handled

Custom genre pools are large — `4x4` alone is ~800 tracks. Stage 1 partitions the pool deterministically by BPM peaks, Camelot connectivity, and era, then enforces the 15–25 track shortlist contract with **seeded windowing**: any oversized shortlist keeps its 15 most-central tracks as a fixed spine and fills the remaining 10 slots by sampling rotated by the run seed. The seed defaults to today's date and is printed at run start (`Stage 1 seed: 20260707 (reproduce with --stage1-seed 20260707)`) — so the same seed reproduces a run exactly, while different days explore different corners of a big pool. Tracks beyond the windows are reported per-shortlist on stderr and as a `Stage 1 overflow` line in the pipeline summary, never silently dropped. See `docs/architecture/deterministic-stage1.md` for the full algorithm.

---

## Business rules

### Track inclusion

- Source: `import/rekordbox.xml` — read fresh on every run, never cached
- Tracks missing BPM or Camelot key are excluded with a warning printed to stderr
- SoundCloud tracks (Location starting with `file://localhostsoundcloud`) are excluded silently
- Tracks in a Rekordbox playlist named `DO NOT RECOMMEND` are excluded from every run; the crate snapshot shows how many were excluded, and a warning fires if the playlist is missing from the XML
- If `CATALOG_API_URL` is set, tracks in your catalog play history are excluded — matched on normalised artist + title keys (unicode/dash normalisation, feat. stripping, version-suffix stripping); otherwise all tracks are treated as unplayed

### BPM correction

Drum & Bass tracks (genre tag `Drum & Bass` or `DnB`) with a recorded BPM below 100 are automatically doubled. Rekordbox sometimes stores DnB at half-time tempo.

### Genre clustering

- Tracks are grouped by Rekordbox genre tag, then aggregated under a canonical label via `GENRE_MAP` in `config.py`
- Tracks whose genre tag is not in `GENRE_MAP` and not in `IGNORED_GENRES` appear as **Outliers** in the Discord report
- Outlier tracks matching the requested genre name (case-insensitive) are passed to Stage 1 as a `Misc` cluster if there are 4 or more of them
- Custom genres (`170`, `140`, `4x4`) merge multiple standard genres into a single pool before Stage 1; see [Custom genres](#custom-genres)

### BPM filtering (per cluster)

- **Standard genres:** tracks are partitioned into three pools relative to the cluster median — core (±6 BPM), bridge (±12 BPM), and wildcard (>12 BPM). Core tracks are sent to Stage 1; bridge and wildcard tracks are retained as canvas metadata and are available to Stage 2 for structural roles such as opener, closer, or pivot where BPM deviation is intentional.
- **Custom genres with a defined BPM range** (`170`, `140`): a hard range filter is applied instead — tracks outside the defined range are excluded
- **Custom genres without a BPM range** (`4x4`): no BPM filter is applied; Stage 1 finds BPM-coherent groupings within the pool itself

### Camelot key ordering

Tracks within each concept are sorted for harmonic compatibility. The algorithm walks the Camelot wheel preferring adjacent moves (±1 same mode, or same number opposite mode), falling back to lowest BPM when no harmonically compatible track is available.

### LLM Stage 1 — concept generation

- Provider cascade tried in order, falling through on error or missing key:
  **Groq → Gemini → Mistral**
- **Standard genres:** clusters larger than 40 tracks are chunked; each chunk is called independently and concepts merged
- **Custom genres:** the pool is partitioned deterministically by BPM peaks, Camelot connectivity, and era (see `docs/architecture/deterministic-stage1.md`) — same shortlist algorithm as standard genres, applied to the merged cross-genre pool
- Track IDs are aliased to short positional keys (`T001`, `T002`, …) in the prompt; hallucinated IDs are structurally impossible and concepts with fewer than 4 resolvable aliases are discarded
- Stage 1 concepts are wrapped into **Mix Canvases** — structured objects that add role candidates (opener, groove-locker, builder, pivot, peak, closer), contrast assets (vocal moments, texture changes, darker/brighter turns), deterministic risk notes (weak opener/closer pool, BPM spread, artist repetition), an era window and dominant label when the core pool supports them, identity-defining `Anchors` from the core pool (provenance + library rarity + pool centrality), and `Concept anchors` tagging bridge/wildcard tracks as `peak`, `identity`, or `structural-exception`. Up to 6 canvases are forwarded to Stage 2, selected by a weighted scoring model covering technical viability, role coverage, anchor strength, contrast potential, cross-canvas distinctiveness, era coherence, label coherence, and novelty against recent run history. Weights are mode-aware — `unplayed` mode prioritises novelty, `played` mode prioritises anchor strength, `all` mode prioritises cross-canvas distinctiveness. Selection is deterministic given the same input — no random sampling.

### LLM Stage 2 — report generation

- Uses Claude Sonnet 4.6 (Anthropic-only, no fallback provider)
- Before sequencing, chooses an explicit energy path (Slow Climb, Wave, Plateau With Detail, Double Peak, Front-Loaded Hook, Dark to Light, Light to Dark) and assigns every track to one of five sections: Invitation, Groove Lock, Development, Peak/Payoff, Resolution
- Assigns each track a role from a focused 10-role vocabulary: opener, groove, hook, pivot, lift, vocal-moment, texture-change, peak, resolution, closer (a track may carry more than one)
- Each report includes: named energy path, structured `arc_type` field, section breakdown with track numbers, per-track role and transition risk, dedicated opener and closer rationale, excluded tracks with reasons, a `Bold moves:` summary of bridge/wildcard usage with the mechanism that justified each pick, and a one-line `Practicality:` score (bpm_smoothness, harmonic_ratio, risk_justified, overall) for triage
- If the catalog API returns existing mix names, Stage 2 is instructed to avoid reusing any words, tropes, or phrasing from them; each concept also includes a `name_reason` tying the name to the set's thesis
- Playlist mode generates three variants (`practical`, `balanced`, `adventurous`) and auto-selects the strongest by a tolerance-aware `fit` score (practicality blended with an adventure dividend for justified risk-taking, weighted by `risk_tolerance` — see "Complete a mix from an existing Rekordbox playlist" below); seed retention is enforced with a floor of 75% of anchor tracks and 40% of supporting tracks
- Appends shortfall warnings for concepts significantly below the recommended track count for their genre
- Appends the active report context and elapsed generation time to the final output
- After each successful run, concept history is written to `.mixlab/concept-history.json`. On subsequent runs, canvases are penalised on a combined novelty score: 65% track-overlap Jaccard plus 35% concept-shape similarity (BPM band, dominant Camelot zone, role pattern, `arc_type`). Both components decay at 0.8^age across a 10-run recency window. Catches "different tracks, same shape" repetition the old track-only signal missed. Stage 2 also sees a `RECENT CONCEPTS` block listing recent titles/arcs/moods so it can deliberately diverge.
- Post-Stage-2 validation is warn-only. Strong-tier checks (always fire): track IDs not found in the library, denylist or played-track violations, Camelot jumps greater than 4, BPM jumps greater than 15, artist repeats of 3 or more, opener/closer absent in expected positions, bridge/wildcard tracks used without a justified `risk_type`, and wildcard tracks used outside the canvas's concept-anchor list. Soft-tier checks (softened by genre family and `arc_type`): no peak in sequence, no wind-down before closer, three-or-more consecutive same-role-family tracks, all tracks high-energy, cross-concept track overlap above 50%, generic `[Adjective][Noun]` concept titles. Warnings appear under **⚠ Validation Notes** and never abort the run.

### Bounded self-revision (genre mode)

After validation, MixLab attempts one minimal repair per flagged concept. A concept is flagged when it has two or more *hard* findings (track not found, denylisted, played, BPM jump, Camelot jump, arc mismatch, or a bridge/wildcard track used without a justified transition), or — under `--deep` — when its critique verdict is `weak`, or `needs_attention` with a concrete suggested substitution. The revision call is a targeted repair, not a regeneration: the model may only swap, reorder, or drop tracks from the same canvas pool, and must preserve the title, thesis, and character. The result is accepted only if it strictly reduces the concept's hard-finding count; otherwise the original is kept. This is a hard one-pass cap — never a second round. Accepted repairs add a **Revised** annotation to the report noting how many findings were resolved (the prose still describes the pre-revision sequence; the exported playlist uses the revised order), and the **⚠ Validation Notes** section reflects the post-revision state. Pass `--no-revise` to skip the pass entirely. Cost: roughly one extra small Stage 2 call per flagged concept; combined with `--deep` the worst case is about 3× baseline Stage 2 cost.

### Shortfall thresholds (tracks per set)

| Genre | Minimum | Target |
|---|---|---|
| Drum & Bass / DnB | 10 | 14 |
| UK Garage | 10 | 13 |
| Jungle | 12 | 16 |
| House / Deep House / Techno | 8 | 12 |
| All others | 8 | 12 |

A shortfall warning fires when a concept has more than 4 tracks below the minimum.

---

## Development

### Run tests

```bash
pytest
pytest --tb=short -q   # terse output
```

### Lint and format

```bash
ruff format .
ruff check .
```

### Type checking

```bash
mypy .
```

All three must pass clean before committing. See `CLAUDE.md` for full coding conventions.

---

## Project structure

```
MixLab/
├── src/mixlab/
│   ├── __main__.py        # CLI entry point and pipeline orchestration
│   ├── reader.py          # Rekordbox XML parsing and BPM correction
│   ├── client.py          # Catalog API client (played track history, optional)
│   ├── matcher.py         # Fuzzy played-track exclusion
│   ├── clustering.py      # Genre grouping, BPM pool partitioning, Mix Canvas builder and scoring
│   ├── history.py         # Concept history read/write (.mixlab/concept-history.json)
│   ├── llm.py             # Stage 1 provider cascade + Stage 2 Anthropic report + post-run validation
│   ├── playlist_exporter.py # Rekordbox playlist XML export
│   ├── discord_client.py  # Discord delivery and report formatting
│   ├── cache.py           # Genre availability cache (.mixlab_genres.json)
│   ├── config.py          # GENRE_MAP, IGNORED_GENRES, TRACK_COUNT_TARGETS
│   └── models.py          # Pydantic models: Track, MixConcept, MixCanvas, BpmPools, CanvasScore
├── tests/                 # pytest suite mirroring src layout
├── import/
│   └── rekordbox.xml      # Your Rekordbox export (gitignored)
├── pyproject.toml
└── .env.example
```

---

## License

MIT — see [LICENSE](LICENSE).
