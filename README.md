# MixLab

[![GitHub release](https://img.shields.io/github/v/release/christophechang/MixLab)](https://github.com/christophechang/MixLab/releases)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Point it at your collection. Get a set worth playing.**

AI-powered DJ crate assistant. Point it at your Rekordbox collection, pick a genre, and get a set of ready-to-use mix concepts with Camelot-ordered track listings — delivered to Discord.

<!-- demo screenshot or terminal recording goes here -->

For background on why this was built and how it works in practice, read the [MixLab case study](https://www.soltechconsulting.co.uk/case-studies/mixlab).

---

## Quickstart

```bash
git clone <repo-url> && cd MixLab
./setup.sh
cp .env.example .env   # fill in ANTHROPIC_API_KEY + at least one Stage 1 key

./mixlab               # crate availability table — no LLM cost
./mixlab --genre house # full mix report, delivered to Discord
```

---

## How it works

1. Parses your exported Rekordbox XML collection
2. If `CATALOG_API_URL` is set, fetches your play history and filters out already-played tracks; otherwise uses the full collection
3. Prints a crate availability table (no LLM cost)
4. If `--genre` is specified, scopes the collection to that genre (or custom cross-genre pool), runs Stage 1 shortlisting, then writes a full Stage 2 mix planning report
5. If `--playlist` is specified, uses that Rekordbox playlist as the seed, infers the set's intent, builds natural BPM-zone shortlists around the seed tracks, generates three completion variants, then writes the best playlist-completion report
6. Optionally exports a Rekordbox-compatible merged XML file
7. Sends the report and any XML attachment to a Discord channel

---

## What's new in v0.2.0

- **BPM and year range filters.** Four new CLI flags — `--min-bpm`, `--max-bpm`, `--min-year`, `--max-year` — narrow the candidate pool before Stage 1. In playlist mode, filters apply only to library additions and never touch seed tracks. Active filters appear in the Discord crate snapshot label.
- **DO NOT RECOMMEND playlist exclusion.** Tracks in a Rekordbox playlist named `DO NOT RECOMMEND` are silently excluded from every run. The crate snapshot shows how many tracks were excluded, and a warning fires if the playlist is missing from the XML.
- **Catalogue name deduplication.** Stage 2 now receives a list of existing mix names from the catalogue and avoids repeating words, tropes, or phrasing from them. Each concept also carries a `name_reason` field — a one-sentence justification tying the name to the set's thesis rather than individual tracks.

---

## Requirements

- Python 3.12+
- A Rekordbox XML export (see [Setup](#4-export-your-rekordbox-collection))
- `ANTHROPIC_API_KEY` — required for default Stage 2 report generation
- At least one Stage 1 LLM key (Groq, Gemini, Mistral, or MiniMax)
- A catalog API URL + key (optional — for filtering already-played tracks)
- A Discord bot token (optional — report prints to stdout without it)

---

## Setup

### 1. Clone and run setup

```bash
git clone <repo-url>
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
| `MINIMAX_API_KEY` | No | Stage 1 provider (last fallback) and optional Stage 2 provider via `--stage2-provider minimax` |
| `OPENROUTER_API_KEY` | No | Reserved for future use |
| `DISCORD_BOT_TOKEN` | No | Discord delivery |
| `DISCORD_GUILD_ID` | No | Discord server ID |
| `MIXLAB_DISCORD_CHANNEL_ID` | No | Target channel ID (preferred over name) |
| `MIXLAB_DISCORD_CHANNEL` | No | Target channel name (default: `mix-lab`) |

`ANTHROPIC_API_KEY` is required for the default Stage 2 path. `MINIMAX_API_KEY` is additionally required if you want to run `--stage2-provider minimax`. Without a catalog API URL, played-track exclusion is skipped and the full collection is used. Without a Discord token the report is printed to stdout only.

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

#### Track colours (optional)

MixLab reads Rekordbox track colours and maps them to energy tiers passed to the AI:

| Colour | Tier | Meaning |
|---|---|---|
| Red | High energy | Peak / floor-filling track |
| Orange | Mid energy | Builder or transition track |
| Green | Chill | Opener, palette cleanser, or warm-up |

Colour-code your tracks in Rekordbox manually to give the AI an additional signal beyond the Mixed In Key energy score. Mixed In Key can also set colours automatically based on energy level if you configure it to do so.

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
2. Optionally colour-code tracks in Rekordbox (or let Mixed In Key do it by energy level)
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

### View crate availability (no LLM calls)

```bash
./mixlab
```

Prints unplayed vs total counts per genre, sorted by availability. Only the catalog API is called (if configured). Without a catalog API, shows total collection counts.

### Generate a mix report for a specific genre

```bash
./mixlab --genre house
./mixlab --genre house --stage2-provider minimax
./mixlab --genre 4x4 --all-tracks
```

Runs the full genre pipeline: parse → fetch played history (unless `--all-tracks`) → scope to the requested genre → Stage 1 shortlist generation → Stage 2 report → Discord/stdout.

The report starts with a context header so you can see exactly what kind of run produced it, for example:

```text
Report context: House (unplayed tracks)
Report context: House (unplayed tracks, stage 2: minimax)
Report context: 140 (custom genre, All Tracks)
```

For standard and custom genre runs, a Rekordbox-compatible merged XML file can be attached to the Discord message or written to disk. It contains one playlist per concept plus an **All Unplayed Tunes** playlist with the full scoped unplayed pool when played-track history was used.

### Complete a mix from an existing Rekordbox playlist

```bash
./mixlab --playlist "Monday Night"
./mixlab --playlist "Monday Night" --genre electronica
./mixlab --playlist "Sets/Monday Night"
./mixlab --playlist "Monday Night" --all-tracks
./mixlab --playlist "Monday Night" --stage2-provider minimax
```

Playlist mode is a different workflow from genre mode:

- The source Rekordbox playlist is treated as the seed and MixLab aims to complete or extend it, not replace it
- MixLab first runs an intent-analysis pass over the seed playlist to infer the overall vibe, energy shape, anchor tracks, and any missing set roles
- Seed tracks are clustered into natural BPM zones
- Each zone becomes a shortlist containing the seed tracks for that zone plus nearby library tracks
- Stage 2 generates exactly three completion variants (`practical`, `balanced`, and `adventurous`) and MixLab auto-selects the strongest one
- The final report explains which seed tracks were retained, which were dropped, which library tracks were added, and which alternative strategy was rejected

Important playlist-mode rules:

- `--genre` in playlist mode constrains added library tracks to that genre scope; it does not filter the seed playlist itself
- `--all-tracks` disables played-track weighting in playlist mode and uses the full collection as the candidate pool
- Playlist names are matched case-insensitively
- If the same playlist name exists in multiple folders, pass the full path such as `Sets/Monday Night`
- Playlist mode requires at least 4 valid seed tracks with BPM and Camelot key after parsing

Playlist runs use the same report context header as genre runs, for example:

```text
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

### View cached genre counts from the last run (no API calls at all)

```bash
./mixlab --genres
```

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
```

| Label | Sub-genres | BPM range | Rationale |
|---|---|---|---|
| `170` | drum_and_bass + jungle | 165–175 BPM | Hardcore continuum genres that live at the same tempo and share rhythmic DNA — the richest cross-genre territory in the collection |
| `140` | breakbeat + uk_bass + uk_garage | 130–140 BPM | UK underground genres that occupy the same tempo bracket; blends can range from technical to percussive to bass-heavy |
| `4x4` | house + electronica + disco + progressive + techno | none | The full 4/4 spectrum from deep house to techno, with a wide BPM range (~90–140); the creative challenge is the journey across that arc |

Custom genres behave differently from standard genres in two key ways:

**1. BPM filtering.** `170` and `140` apply a hard BPM range filter — tracks outside those bounds are excluded before Stage 1. The range is part of what defines the genre. `4x4` has no hard BPM filter: the sub-genres span a wide range (~90–140 BPM) and Stage 1 is responsible for finding BPM-coherent groupings within the pool.

**2. Stage 2 cross-genre guidance.** The Stage 2 prompt is given the list of sub-genres and instructed to justify any move across genre boundaries — naming the specific mechanism that makes the transition work (BPM alignment, rhythmic character, harmonic relationship, or the energy state of the room). Cross-genre moves are not avoided; they are the point of using a custom genre. But every such move must be defensible.

#### Why random selection?

Custom genre pools are large — `4x4` alone is ~800 tracks. Sending the whole pool to Stage 1 would mean 10–15 sequential API calls to a free-tier LLM provider, which hits rate limits on every run and produces the same output every time.

Instead, MixLab picks a **random 120-track window** from the BPM-sorted pool on each run:

- The pool is sorted by BPM, so adjacent tracks in the sorted list are in the same tempo zone. The window always lands on a BPM-coherent slice — house tracks at 120–125 one run, progressive/techno at 128–133 the next.
- Only 2 Stage 1 API calls are made per run (120 tracks / 60 per call), keeping the LLM load light and the rate limits safe.
- Each run explores a different section of the collection, so you get different concepts each time without manually curating which tracks to send.

Run the same custom genre multiple times. Each run will focus on a different corner of the pool and produce different concepts.

---

## Business rules

### Track inclusion

- Source: `import/rekordbox.xml` — read fresh on every run, never cached
- Tracks missing BPM or Camelot key are excluded with a warning printed to stderr
- SoundCloud tracks (Location starting with `file://localhostsoundcloud`) are excluded silently
- Tracks in a Rekordbox playlist named `DO NOT RECOMMEND` are excluded from every run; the crate snapshot shows how many were excluded, and a warning fires if the playlist is missing from the XML
- If `CATALOG_API_URL` is set, tracks in your catalog play history are excluded — fuzzy-matched on artist + title with unicode normalisation, dash normalisation, and feat. stripping; otherwise all tracks are treated as unplayed

### BPM correction

Drum & Bass tracks (genre tag `Drum & Bass` or `DnB`) with a recorded BPM below 100 are automatically doubled. Rekordbox sometimes stores DnB at half-time tempo.

### Genre clustering

- Tracks are grouped by Rekordbox genre tag, then aggregated under a canonical label via `GENRE_MAP` in `config.py`
- Tracks whose genre tag is not in `GENRE_MAP` and not in `IGNORED_GENRES` appear as **Outliers** in the Discord report
- Outlier tracks matching the requested genre name (case-insensitive) are passed to Stage 1 as a `Misc` cluster if there are 4 or more of them
- Custom genres (`170`, `140`, `4x4`) merge multiple standard genres into a single pool before Stage 1; see [Custom genres](#custom-genres)

### BPM filtering (per cluster)

- **Standard genres:** any track more than ±6 BPM from the cluster median is removed before Stage 1
- **Custom genres with a defined BPM range** (`170`, `140`): a hard range filter is applied instead — tracks outside the defined range are excluded
- **Custom genres without a BPM range** (`4x4`): no BPM filter is applied; Stage 1 finds BPM-coherent groupings within the pool itself

### Camelot key ordering

Tracks within each concept are sorted for harmonic compatibility. The algorithm walks the Camelot wheel preferring adjacent moves (±1 same mode, or same number opposite mode), falling back to lowest BPM when no harmonically compatible track is available.

### LLM Stage 1 — concept generation

- Provider cascade tried in order, falling through on error or missing key:
  **Groq → Gemini → Mistral → MiniMax**
- **Standard genres:** clusters larger than 40 tracks are chunked; each chunk is called independently and concepts merged
- **Custom genres:** a random 120-track window is selected from the BPM-sorted pool each run (see [Why random selection?](#why-random-selection)); 60 tracks per call, 2 calls maximum; shortlist target is 20–25 tracks per concept (vs 15–25 for standard)
- Concepts with fewer than 4 valid track IDs (after stripping hallucinated IDs) are discarded
- Up to 6 concepts are forwarded to Stage 2, randomly sampled from the top 12 by pool size to ensure variety across runs

### LLM Stage 2 — report generation

- Uses Claude Sonnet 4.6 by default
- `--stage2-provider minimax` switches Stage 2 to MiniMax M2.7 explicitly
- If the default Anthropic Stage 2 path fails mid-run, MixLab can fall back to MiniMax M2.7 when that key is available
- Writes a peer-to-peer mix planning narrative with track listings in Camelot order and notes on transitions
- If the catalog API returns existing mix names, Stage 2 is instructed to avoid reusing any words, tropes, or phrasing from them; each concept also includes a `name_reason` tying the name to the set's thesis
- Playlist mode generates three variants (`practical`, `balanced`, `adventurous`) and auto-selects the strongest; seed retention is enforced with a floor of 75% of anchor tracks and 40% of supporting tracks
- Appends shortfall warnings for concepts significantly below the recommended track count for their genre
- Appends the active report context and elapsed generation time to the final output

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
│   ├── clustering.py      # Genre grouping, BPM filtering, Camelot sort
│   ├── llm.py             # Stage 1 provider cascade + Stage 2 Anthropic report
│   ├── discord_client.py  # Discord delivery and report formatting
│   ├── cache.py           # Genre availability cache (.mixlab_genres.json)
│   ├── config.py          # GENRE_MAP, IGNORED_GENRES, TRACK_COUNT_TARGETS
│   └── models.py          # Pydantic models: Track, PlayedTrack, MixConcept
├── tests/                 # pytest suite mirroring src layout
├── import/
│   └── rekordbox.xml      # Your Rekordbox export (gitignored)
├── pyproject.toml
└── .env.example
```

---

## License

MIT — see [LICENSE](LICENSE).
