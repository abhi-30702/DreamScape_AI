# Rater Study Stats (Phase 3) — Design

**Goal:** Convert the rater study JSON files (Phase 2) and Phase 1 automated metrics into thesis-ready statistics and tables via a single CLI command.

**Scope:** Stats + thesis tables (CSV + Markdown) for the final dissertation. One-shot analysis run at the end of the rater study.

**Prerequisite:** Operator has run `huggingface-cli download --repo-type dataset <repo> --local-dir responses_dump/` to materialize the rater submissions locally. Phase 1 metric output JSONs are also present locally under `eval_runs/`.

---

## Architecture

A small Python package `eval/stats/` with six focused modules. Each module owns one concern and stays under ~80 lines.

```
eval/stats/
├── __init__.py
├── loader.py        # walk responses_dump/, parse JSON, join manifest + auto metrics
├── mos.py           # MOS per video/dim + sentiment/style breakdowns + summary
├── kappa.py         # Fleiss' kappa per dimension
├── correlations.py  # Pearson/Spearman + p-values, auto ↔ subjective
├── formatters.py    # CSV + Markdown table writers
└── cli.py           # argparse + orchestrator
```

### Data flow

```
responses_dump/responses/<rater>/<video>.json     study_videos/manifest.json     eval_runs/*.json
                            │                                  │                          │
                            └──────────────────┬───────────────┴──────────────────────────┘
                                               ▼
                                       loader.load_all(...)
                                               │
                                ┌──────────────┼──────────────┐
                                ▼              ▼              ▼
                          ratings_df       auto_df       overall_comments
                                │              │              │
                                │              │              └─► (loaded but not aggregated; just counted)
                                │              │
                ┌───────────────┼──────────────┼───────────────┐
                ▼               ▼              ▼               ▼
            mos.py          kappa.py    correlations.py    summary
                │               │              │               │
                └───────────────┴──────────────┴───────────────┘
                                               │
                                               ▼
                                          formatters
                                               │
                                               ▼
                                       thesis_tables/{csv,md}
```

### Why a package (not a single file)

Phase 1's `metrics.py` is 222 lines for 4 functions. Phase 3 has a loader + 3 stat modules + a formatter + a CLI orchestrator — approximately 350 lines total. Splitting into six small files keeps each unit under ~80 lines and matches the helpers/handlers/UI split used by `ui/rater_study.py`. Tests can mock at the loader boundary and exercise each computation against synthetic DataFrames.

### Stack additions

- `pandas` (tidy DataFrames, groupby/pivot, CSV I/O)
- `scipy.stats` (`pearsonr`, `spearmanr` with two-sided p-values)
- `statsmodels.stats.inter_rater.fleiss_kappa`

All three are widely used, well-tested, and standard to cite in a research thesis.

---

## Loader & tidy DataFrame schema

### Inputs to discover

1. `responses_dump/responses/<rater_id>/<video_id>.json` — every per-video submission written by Phase 2's `ui/rater_storage.save_response`.
2. `responses_dump/responses/<rater_id>/_overall.json` — final-comment submission per rater.
3. `study_videos/manifest.json` — 20-entry source of truth for sentiment/style/prompt/run_id.
4. `eval_runs/<run_id>.json` (or whatever `evaluate_run` writes) — Phase 1 automated metrics keyed by `run_id`.

### Output: one tidy long-format DataFrame

`ratings_df`, keyed by `(rater_id, video_id, dimension)`:

| column | type | source |
|---|---|---|
| `rater_id` | str | filename |
| `video_id` | str | filename |
| `dimension` | str | one of the 6 dimension keys |
| `rating` | int (1–5) | `payload["ratings"][dimension]` |
| `video_sentiment` | str | manifest join |
| `video_style` | str | manifest join |
| `video_run_id` | str/None | payload (denormalized in Phase 2) |
| `submitted_at_utc` | str | payload |

One row per `(rater, video, dimension)` → 20 raters × 20 videos × 6 dims = 2,400 rows at full coverage. Long format because every stat library (groupby for MOS, pivot for kappa input, scipy correlations) consumes it cleanly.

### Auto-metrics DataFrame

`auto_df`, returned alongside ratings_df, keyed by `video_id`:

| column | type | source |
|---|---|---|
| `video_id` | str | manifest |
| `clip_score` | float | eval_runs JSON |
| `wer` | float | eval_runs JSON |
| `sync_error_ms` | float | eval_runs JSON |

Joined via `video_run_id` → manifest `run_id` → `eval_runs/<run_id>.json`. If a video has no matching run (e.g. `run_id` is `null` in manifest), its row gets `NaN`; correlations drop those rows.

### Manifest join behavior

- Submissions whose `video_id` is not in the manifest are dropped with a console warning naming the offending id. Schema mismatch should never crash analysis.
- A rater may submit any subset; nothing requires all 20.

### Public surface

```python
def load_all(
    responses_dir: Path,
    manifest_path: Path,
    eval_runs_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (ratings_df, auto_df, overall_comments)."""
```

Three returns instead of one because the three are consumed by different downstream modules.

---

## Stat computations

Three small modules. Each one is a thin wrapper over a well-known formula.

### `mos.py` — Mean Opinion Score

- **`compute_mos_per_video(ratings_df) -> DataFrame`** — groupby `(video_id, dimension)`, compute `mean`/`std`/`n`. Pivot to wide: rows = `video_id`, columns = the 6 dimensions, one mean per cell. Add `n_raters` (distinct rater count per video) so the thesis table shows coverage.
- **`compute_mos_by_sentiment(ratings_df) -> DataFrame`** — same pattern with `video_sentiment` groupby → 3-row × 6-col table.
- **`compute_mos_by_style(ratings_df) -> DataFrame`** — `video_style` groupby → 5-row × 6-col table.
- **`compute_summary(ratings_df) -> dict`** — overall MOS per dimension across all 2,400 rows + N raters + N videos + mean Fleiss' kappa across dimensions. Drives the headline summary table.

### `kappa.py` — Fleiss' Kappa

**`compute_fleiss_kappa_per_dimension(ratings_df) -> DataFrame`**

For each of the 6 dimensions:
- Build the per-video N×5 count matrix expected by `statsmodels.stats.inter_rater.fleiss_kappa` (rows = videos, cols = ratings 1..5, cells = count of raters who gave that rating).
- For each video, count distinct raters who supplied a rating for *this dimension*. Drop videos with `n < min_raters_per_video` (kappa is undefined for singletons; default minimum = 2, CLI-configurable). Different dimensions may end up with different `n_videos_used` if raters skipped fields — that's expected and reported in the table.
- Returns a 6-row table:

| column | type |
|---|---|
| `dimension` | str |
| `kappa` | float |
| `n_videos_used` | int |
| `interpretation` | str |

`interpretation` maps Landis & Koch bands: `<0.0` poor, `0.0–0.2` slight, `0.21–0.4` fair, `0.41–0.6` moderate, `0.61–0.8` substantial, `>0.8` almost perfect.

### `correlations.py` — Auto vs. human

**Pairings to compute** (auto metric ↔ subjective dimension):

| auto metric | subjective dimension | sign |
|---|---|---|
| `clip_score` | `visual_quality` | + |
| `wer` | `narration_clarity` | negated (lower WER = better) |
| `sync_error_ms` | `av_sync` | negated (lower error = better) |
| composite (see below) | `overall_quality` | exploratory |

**Composite definition for the overall_quality pairing:** z-score each of `clip_score`, `−wer`, `−sync_error_ms` independently across the 20 videos (so each becomes mean-0, std-1), then sum the three z-scores. Correlate the summed composite against per-video `overall_quality` MOS.

**`compute_correlations(per_video_mos: DataFrame, auto_df: DataFrame) -> DataFrame`**

- For each pairing, take per-video mean subjective rating vs the auto metric (N = up to 20 minus any missing).
- Compute both Pearson r and Spearman ρ with their two-sided p-values via `scipy.stats.pearsonr` / `spearmanr`.
- Output columns:

| column | type |
|---|---|
| `pair` | str (e.g. `"clip_score ↔ visual_quality"`) |
| `n` | int |
| `pearson_r` | float |
| `pearson_p` | float |
| `spearman_rho` | float |
| `spearman_p` | float |
| `sig` | str (`"*"` if either p < 0.05, else `""`) |

On N=20 with possible missing values, correlations are noisy; the Markdown output's footnote explicitly states N and caveats interpretation.

---

## Output tables

CLI writes everything under `--out` (default `thesis_tables/`). Every table is emitted in **both** formats so the thesis can paste either:

| File stem | What it shows | Shape |
|---|---|---|
| `summary` | Overall MOS per dimension, N raters, N videos, mean Fleiss' kappa | 1 × ~10 |
| `mos_per_video` | One row per video × 6 dim columns + `n_raters` | 20 × 8 |
| `mos_by_sentiment` | sentiment × dimension MOS matrix | 3 × 6 |
| `mos_by_style` | style × dimension MOS matrix | 5 × 6 |
| `kappa_per_dimension` | dimension, kappa, n_videos, interpretation | 6 × 4 |
| `auto_vs_human_correlations` | pair, n, pearson r/p, spearman ρ/p, sig | 4 × 7 |

Each writes two files: `<stem>.csv` (raw, comma-separated, full precision) and `<stem>.md` (rounded to 2 dp, column headers in Title Case).

### Formatting choices baked into `formatters.py`

- CSV: pandas defaults; no index column when the rowkey is meaningful (e.g. `video_id` becomes a regular column).
- Markdown: GitHub-flavored pipe tables (works in the thesis preview and in pasted blockquotes). Numeric columns right-aligned with `---:`.

### Provenance file

A `provenance.txt` written alongside the tables, recording: CLI args, package versions (`pandas`, `scipy`, `statsmodels`), input file counts, and a UTC timestamp. Lets any thesis number be re-defended ("we computed this with scipy 1.13.0 on 2026-07-12").

### Out of scope (v1)

- Excel output — CSV + Markdown is enough.
- PNG charts — a future subagent task can render box-plots from the CSVs; kept out to preserve YAGNI.

---

## CLI

```
python -m eval.stats \
  --responses responses_dump/ \
  --manifest study_videos/manifest.json \
  --metrics eval_runs/ \
  --out thesis_tables/ \
  [--min-raters-per-video 2]
```

- `--manifest` defaults to `study_videos/manifest.json` (the bundled one).
- `--metrics` is optional. If omitted, the correlations table is skipped with a single console line (`auto metrics not provided; skipping correlations`) and all subjective stats still emit.
- `--min-raters-per-video` defaults to 2 (kappa minimum). Raising it tightens the kappa N at the user's discretion.
- On success, prints a one-line summary per table: `mos_per_video.csv  (20 rows)` etc.

---

## Error handling

| Situation | Behavior |
|---|---|
| `--responses` directory missing or empty | Fail fast with clear message; exit 1 |
| A JSON file fails to parse | Skip it, log warning with path, continue |
| `video_id` in submission not in manifest | Skip the submission, log warning, continue |
| All submissions for one dimension are missing | Emit that dimension's MOS as NaN; kappa row drops to `n_videos_used=0`, `kappa=NaN`, `interpretation="insufficient data"` |
| `--metrics` provided but no `run_id` matches | Correlations table emits all rows with `n=0` and NaN coefficients (still useful as a placeholder so the thesis section structure is in place) |
| `--out` exists with prior files | Overwrite without prompting (CI-friendly); print `overwriting` once |

**Principle:** never crash on partial study data. Phase 3 is run live during thesis writing — every regenerable table beats a crash.

---

## Testing

- `tests/test_stats_loader.py` — synthetic 3-rater × 4-video × 2-dim fixtures via tmp_path. Covers: happy join, manifest mismatch, missing dimension, malformed JSON skipped.
- `tests/test_stats_mos.py` — synthetic DataFrame; asserts means, breakdown shapes, `n_raters`.
- `tests/test_stats_kappa.py` — Fleiss' textbook example (the original 6-rater data published with the 1971 paper) → assert against the canonical 0.430 value; second test with perfect agreement → assert ~1.0.
- `tests/test_stats_correlations.py` — synthetic perfectly-correlated input → assert r≈1, p<0.001. Mismatched lengths → assert skip.
- `tests/test_stats_formatters.py` — write to tmp_path, assert CSV roundtrips through pandas, assert Markdown contains expected headers/values.
- `tests/test_stats_cli.py` — one end-to-end test on a fixture directory; assert all 6 file pairs created and `provenance.txt` non-empty.

Target: ~25 tests total. All run in <2s without network or GPU.

---

## Out of scope (deferred)

- **Demographic analysis** — Phase 2 does not collect rater demographics (anonymous IDs only).
- **Confidence intervals on MOS** — could add bootstrap CIs in a follow-up if the thesis defense calls for it.
- **PNG charts / box-plots** — could render from the CSVs in a follow-up subagent task.
- **Auto-vs-human regression model** — current scope is correlation only; a regression would be a separate analysis.
- **Inter-style or inter-sentiment significance tests** — N≈4 videos per cell is too small to support ANOVA; the breakdown tables are descriptive only.
