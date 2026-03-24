# MixLab

AI-powered DJ crate assistant. Point it at your Rekordbox collection, pick a genre, and get a set of ready-to-use mix concepts with Camelot-ordered track listings — delivered to Discord.

For background on why this was built and how it works in practice, read the [MixLab case study](https://www.soltechconsulting.co.uk/case-studies/mixlab).

---

## How it works

1. Parses your exported Rekordbox XML collection
2. If `CATALOG_API_URL` is set, fetches your play history and filters out already-played tracks; otherwise uses the full collection
3. Prints a crate availability table (no LLM cost)
4. If a genre is specified, clusters the tracks, runs them through an LLM cascade to generate mix concepts, then writes a full mix planning report via a second LLM call
5. Exports a Rekordbox-compatible XML file containing one playlist per concept plus an **All Unplayed Tunes** playlist with the full genre pool
6. Sends the report and XML to a Discord channel

---

## Requirements

- Python 3.12+
- A Rekordbox XML export (see [Setup](#4-export-your-rekordbox-collection))
- `ANTHROPIC_API_KEY` — required for Stage 2 report generation
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
| `ANTHROPIC_API_KEY` | Yes | Stage 2 report generation (Claude Sonnet) |
| `CATALOG_API_URL` | No | Base URL of your catalog/play-history API |
| `CHANGSTA_API_KEY` | No | Bearer token for `CATALOG_API_URL` (if your API requires auth) |
| `GROQ_API_KEY` | No | Stage 1 provider (tried first) |
| `GEMINI_API_KEY` | No | Stage 1 provider (fallback 1) |
| `MISTRAL_API_KEY` | No | Stage 1 provider (fallback 2) |
| `MINIMAX_API_KEY` | No | Stage 1 provider (last fallback) + Stage 2 alternative (`--stage2-provider minimax`) |
| `OPENROUTER_API_KEY` | No | Reserved for future use |
| `DISCORD_BOT_TOKEN` | No | Discord delivery |
| `DISCORD_GUILD_ID` | No | Discord server ID |
| `MIXLAB_DISCORD_CHANNEL_ID` | No | Target channel ID (preferred over name) |
| `MIXLAB_DISCORD_CHANNEL` | No | Target channel name (default: `mix-lab`) |

`ANTHROPIC_API_KEY` is the only required key. Without a catalog API URL, played-track exclusion is skipped and the full collection is used. Without a Discord token the report is printed to stdout only.

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
```

Runs the full pipeline: parse → filter played → cluster → Stage 1 concepts → Stage 2 report → Discord. A Rekordbox-compatible XML file is attached to the Discord message containing one playlist per concept plus an **All Unplayed Tunes** playlist with the full genre pool.

To also write the XML to disk:

```bash
./mixlab --genre house --export-playlists
# writes to output/playlists/rekordbox_export.xml

./mixlab --genre house --export /path/to/dir
# writes to /path/to/dir/rekordbox_export.xml
```

### View cached genre counts from the last run (no API calls at all)

```bash
./mixlab --genres
```

---

## Available genres

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

Run `./mixlab` first to see your availability counts — genres with the most unplayed tracks produce the richest concepts.

---

## Business rules

### Track inclusion

- Source: `import/rekordbox.xml` — read fresh on every run, never cached
- Tracks missing BPM or Camelot key are excluded with a warning printed to stderr
- SoundCloud tracks (Location starting with `file://localhostsoundcloud`) are excluded silently
- If `CATALOG_API_URL` is set, tracks in your catalog play history are excluded — fuzzy-matched on artist + title with unicode normalisation, dash normalisation, and feat. stripping; otherwise all tracks are treated as unplayed

### BPM correction

Drum & Bass tracks (genre tag `Drum & Bass` or `DnB`) with a recorded BPM below 100 are automatically doubled. Rekordbox sometimes stores DnB at half-time tempo.

### Genre clustering

- Tracks are grouped by Rekordbox genre tag, then aggregated under a canonical label via `GENRE_MAP` in `config.py`
- Tracks whose genre tag is not in `GENRE_MAP` and not in `IGNORED_GENRES` appear as **Outliers** in the Discord report
- Outlier tracks matching the requested genre name (case-insensitive) are passed to Stage 1 as a `Misc` cluster if there are 4 or more of them

### BPM filtering (per cluster)

Before sending tracks to the LLM, any track more than ±6 BPM from the cluster median is removed. This keeps generated concepts mixable without extreme pitch shifting.

### Camelot key ordering

Tracks within each concept are sorted for harmonic compatibility. The algorithm walks the Camelot wheel preferring adjacent moves (±1 same mode, or same number opposite mode), falling back to lowest BPM when no harmonically compatible track is available.

### LLM Stage 1 — concept generation

- Provider cascade tried in order, falling through on error or missing key:
  **Groq → Gemini → Mistral → MiniMax**
- Clusters larger than 40 tracks are chunked; each chunk is called independently and concepts merged
- Concepts with fewer than 4 valid track IDs (after stripping hallucinated IDs) are discarded
- Up to 6 concepts, ranked by track count, are forwarded to Stage 2

### LLM Stage 2 — report generation

- Always uses Anthropic (Claude Sonnet); falls back to MiniMax M2.7 if Anthropic fails mid-run
- Writes a peer-to-peer mix planning narrative with track listings in Camelot order and notes on transitions
- Appends shortfall warnings for concepts significantly below the recommended track count for their genre
- Elapsed generation time is appended to the report and included in the Discord message

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
