# H3 — DJ-Workability Output: Schema Prep

Read-only investigation of Rekordbox XML cue/grid data available in `import/rekordbox.xml`. Goal: enumerate what data H3 (#25 H3) could draw on, and flag gaps before any conversion to a real ticket.

**Not a spec.** Prep only. Convert H3 to a real issue per #25's gate before implementing.

---

## What the XML contains

Two relevant elements appear inside `<TRACK>`:

### `<TEMPO>` — beat grid

Attributes observed:

| Attribute | Type | Example | Meaning |
|-----------|------|---------|---------|
| `Inizio` | float (seconds) | `0.196` | Beat anchor time |
| `Bpm` | float | `126.00` | BPM at this anchor |
| `Metro` | string | `4/4` | Time signature |
| `Battito` | int | `1` | Beat number within bar |

Multiple `<TEMPO>` per track allowed → variable-BPM grid. Most tracks observed have 1–2 anchors.

Useful for: beat-accurate phrase timing if downstream code computes `time → bar/beat` from the anchors.

### `<POSITION_MARK>` — cue points / loops

Attributes observed:

| Attribute | Type | Example | Notes |
|-----------|------|---------|-------|
| `Name` | string | `""` | **Always empty in the source collection** (1467/1467 marks) |
| `Type` | int | `0` or `4` | `0` = hot/memory cue, `4` = loop (has `End` attribute) |
| `Start` | float (seconds) | `0.667` | Cue position |
| `End` | float (seconds) | `18.539` | Present only on `Type="4"` (loops) |
| `Num` | int | `-1`, `0`, `1`, `2` | `0` = memory cue (red), `1+` = hot cues, `-1` = ungrouped/auto |
| `Red`, `Green`, `Blue` | int 0–255 | `255`, `55`, `111` | Cue color RGB |

Distinct `Type` values in collection: only `0` (1454 occurrences) and `4` (13 occurrences). **No Rekordbox fade-in (`Type=1`) or fade-out (`Type=2`) markers in use.**

### Coverage

- Total tracks: 2128
- Tracks with ≥1 `POSITION_MARK`: 767 (~36%)
- Distribution among the 767: 290 have 1 cue, 333 have 2, 102 have 3, 42 have 4+
- Tracks with named cues: **0**

---

## Mapping to H3 requirements

H3 (from #25) describes: "Surface per-transition workability: intro length, outro length, vocal-entry time, recommended mix-out cue."

| H3 signal | Available in XML? | How |
|-----------|-------------------|-----|
| Intro length | **Partial heuristic only.** No labeled intro end. Could approximate as `time of first Num=0 cue` (often placed at "real start" by DJ convention). Fragile — depends on user's cue-placement habit. | Read first `<POSITION_MARK Type="0" Num="0">.Start` |
| Outro length | **Not available.** No outro markers. Last cue is not reliably an outro start. | — |
| Vocal-entry time | **Not available.** Would require audio analysis (vocal detection) or named cues which don't exist here. | — |
| Recommended mix-out cue | **Not available.** No semantic labels. Loops (`Type=4`) might mark loopable sections but only 13 in the whole collection. | — |
| Phrase boundaries (16/32-bar) | **Computable from beat grid.** `<TEMPO>` anchors + `AverageBpm` → time-to-bar conversion → align to 16/32-bar phrases. | Derive from `Inizio` + `Bpm` |
| First-beat offset | **Available.** `<TEMPO Battito="1">.Inizio` is the downbeat anchor. | Direct |

### Verdict

The XML alone is **insufficient** for H3 as scoped. Three blockers:

1. **No named cues.** Every `POSITION_MARK.Name=""`. Without "intro end" / "vocal in" / "drop" labels there is no direct intro/outro/vocal-entry signal.
2. **No fade-in/fade-out markers.** `Type=1` and `Type=2` are unused. Rekordbox's own intro/outro convention is not being applied to this collection.
3. **Sparse coverage.** Only ~36% of tracks have any cue data. The other 64% would yield no workability output at all.

### Paths if H3 converts

1. **Pure beat-grid path** — Compute phrase boundaries from `<TEMPO>` anchors and bar/beat alignment. Output: "track is 8 bars to first phrase boundary at 7.6s, then 32-bar phrases through 213.4s, last 8 bars from 252.8s". Works on 100% of tracks with `<TEMPO>` data. **Does NOT identify intro/outro/vocal.** Just gives phrase math.
2. **Audio-analysis path** — Use librosa or similar to detect vocal-onset, segment boundaries, energy contour. Major engineering: new dependency, batch-precompute pipeline, caching. Most accurate but slowest to ship.
3. **User-convention path** — Document a cue convention (e.g. "Num=0 = phrase one; Num=7 = mix-out target") and ask the user to mark tracks accordingly. Zero engineering, but only useful on tracks the user actually marks. Compound benefit over time but no immediate payoff.

---

## Recommendation when H3 converts

If H3 converts, lead with path 1 (beat-grid phrase math). It is the only no-dependency, no-workflow-change option that touches 100% of tracks. Report would surface phrase-aligned mix-window hints rather than semantic intro/outro labels.

Path 2 (audio analysis) is a separate, larger initiative and should be its own ticket. Path 3 (user convention) is a documentation-only adjunct that does not block path 1.

## Out of scope for this prep

- Parser changes in `reader.py` (today's `reader.py` reads `<TRACK>` attrs but ignores `<TEMPO>` and `<POSITION_MARK>` — adding parsing is implementation work, not prep)
- Performance/caching considerations for path 2
- Conversion criteria evaluation (see #25's gate; user demand signal must accumulate first)

---

## Source

- XML file: `import/rekordbox.xml` (2128 tracks, 1467 position marks, ~36% cue coverage)
- Investigation date: 2026-05-17
- Related: #25 H3 line item
