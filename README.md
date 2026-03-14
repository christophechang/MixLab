# MixLab

AI-powered DJ crate assistant. Point it at your Rekordbox collection, pick a genre, and get a set of ready-to-use mix concepts with Camelot-ordered track listings — delivered to Discord.

---

## How it works

1. Parses your exported Rekordbox XML collection
2. Fetches your play history from the Changsta API and filters out already-played tracks
3. Prints a crate availability table (no LLM cost)
4. If a genre is specified, clusters the unplayed tracks, runs them through an LLM cascade to generate mix concepts, then writes a full mix planning report via a second LLM call
5. Sends the report to a Discord channel

---

## Requirements

- Python 3.12+
- A Rekordbox XML export (see below)
- A Changsta API key
- At least one LLM API key (Anthropic required for Stage 2; any cascade provider works for Stage 1)
- A Discord bot token (optional — report still prints to stdout without it)

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd MixLab
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `CHANGSTA_API_KEY` | Yes | Fetches your play history for track exclusion |
| `ANTHROPIC_API_KEY` | Yes | Stage 2 report generation (Claude) |
| `MINIMAX_API_KEY` | No | Stage 1 provider (tried first) |
| `GROQ_API_KEY` | No | Stage 1 provider (fallback) |
| `GEMINI_API_KEY` | No | Stage 1 provider (fallback) |
| `OPENROUTER_API_KEY` | No | Stage 1 provider (fallback) |
| `DISCORD_BOT_TOKEN` | No | Discord delivery |
| `DISCORD_GUILD_ID` | No | Discord server ID |
| `MIXLAB_DISCORD_CHANNEL_ID` | No | Target channel ID (preferred over name) |
| `MIXLAB_DISCORD_CHANNEL` | No | Target channel name (default: `mix-lab`) |

At minimum you need `CHANGSTA_API_KEY` and `ANTHROPIC_API_KEY` to run a full report. Without a Discord token the report is printed to stdout only.

### 4. Export your Rekordbox collection

In Rekordbox:

1. Go to **File → Export Collection in xml format**
2. Select the playlist you want to export (typically your master "All" playlist)
3. Save the file to `import/rekordbox.xml`

> **Note:** Rekordbox always exports the full COLLECTION regardless of which playlist you select. Tracks not assigned to any playlist will still appear in the XML but will be filtered out automatically (SoundCloud cloud tracks are also excluded). Re-export whenever you add new tracks to your collection.

---

## Running MixLab

### View crate availability (no LLM calls)

```bash
python -m mixlab
```

Prints unplayed vs total counts per genre, sorted by availability. Only the Changsta API is called.

### Generate a mix report for a specific genre

```bash
python -m mixlab --genre house
```

Runs the full pipeline: parse → filter played → cluster → Stage 1 concepts → Stage 2 report → Discord.

### View cached genre counts from the last run (no API calls at all)

```bash
python -m mixlab --genres
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

**Best genres to test with** — run `python -m mixlab` first to see your current availability counts. The genres with the most unplayed tracks will produce the richest concepts. Based on a typical collection size:

- `house` — largest crate, most concept variety
- `drum_and_bass` — second largest, good for testing BPM correction logic
- `breakbeat` — solid depth, distinct from the above

---

## Business rules

### Track inclusion

- Source: `import/rekordbox.xml` — read fresh on every run, never cached
- Tracks missing BPM or Camelot key are excluded with a warning printed to stderr
- SoundCloud tracks (Location starting with `file://localhostsoundcloud`) are excluded silently
- Tracks in your Changsta play history are excluded — fuzzy-matched on artist + title with unicode normalisation, dash normalisation, and feat. stripping

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
  **MiniMax → Groq → Gemini → OpenRouter → Anthropic**
- Clusters larger than 40 tracks are chunked; each chunk is called independently and concepts merged
- Concepts with fewer than 4 valid track IDs (after stripping hallucinated IDs) are discarded
- Up to 6 concepts, ranked by track count, are forwarded to Stage 2

### LLM Stage 2 — report generation

- Always uses Anthropic (Claude Sonnet) — no fallback
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
│   ├── client.py          # Changsta API (played track history)
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
