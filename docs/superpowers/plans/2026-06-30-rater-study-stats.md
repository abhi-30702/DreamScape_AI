# Rater Study Stats (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `eval/stats/`, a six-module Python package + CLI that turns the rater study JSON files (Phase 2) and Phase 1 automated metrics into six thesis-ready statistic tables (CSV + Markdown).

**Architecture:** Small Python package `eval/stats/` with one module per concern: `loader.py` (file discovery + tidy DataFrame), `mos.py` (means + breakdowns), `kappa.py` (Fleiss'), `correlations.py` (Pearson/Spearman + p-values), `formatters.py` (CSV + Markdown writers), `cli.py` (argparse orchestrator). One CLI command emits all six tables under `--out` (default `thesis_tables/`).

**Tech Stack:** Python 3.11, `pandas` (DataFrames + CSV I/O), `scipy.stats` (correlations + p-values), `statsmodels.stats.inter_rater.fleiss_kappa`, pytest. Adds `pandas`, `scipy`, `statsmodels` to `requirements.txt`.

**Spec:** `docs/superpowers/specs/2026-06-30-rater-study-stats-design.md`

**Branch:** `feat/rater-study-stats` (already created from `main`).

---

## File Structure

```
eval/
├── __init__.py            # existing (empty)
├── metrics.py             # existing (Phase 1 — unmodified)
└── stats/                 # NEW
    ├── __init__.py        # NEW (Task 1)
    ├── loader.py          # NEW (Task 2)
    ├── mos.py             # NEW (Task 3)
    ├── kappa.py           # NEW (Task 4)
    ├── correlations.py    # NEW (Task 5)
    ├── formatters.py      # NEW (Task 6)
    └── cli.py             # NEW (Task 7)

tests/
├── test_stats_loader.py        # NEW (Task 2)
├── test_stats_mos.py           # NEW (Task 3)
├── test_stats_kappa.py         # NEW (Task 4)
├── test_stats_correlations.py  # NEW (Task 5)
├── test_stats_formatters.py    # NEW (Task 6)
└── test_stats_cli.py           # NEW (Task 7)

requirements.txt           # MODIFY (Task 1)
```

---

## Notes for the Implementer

- Branch `feat/rater-study-stats` is already created. Do **not** commit on `main`.
- Phase 1 metric JSONs live under `eval_results/<run_id>.json` (the default in `eval/metrics.py` line 209). Each file has shape `{run_id, prompt, duration_target_s, clip_score: {per_scene, mean}, wer: {per_scene, mean}, sync_error_ms: {per_entry, mean, max, pass}}`. The loader extracts `["clip_score"]["mean"]`, `["wer"]["mean"]`, `["sync_error_ms"]["mean"]`.
- The CLI flag is `--metrics` but the default directory name is `eval_results/` (Phase 1's convention).
- Phase 2's per-submission JSON shape is in `docs/superpowers/specs/2026-06-30-rater-study-ui-design.md` — keys are `rater_id`, `video_id`, `video_sentiment`, `video_style`, `video_run_id`, `ratings` (dict of 6 dim keys → 1..5), `comment`, `video_order_index`, `submitted_at_utc`, `app_version`, `schema_version`.
- Each dimension's key list (matches Phase 2): `visual_quality`, `narration_clarity`, `music_mood_fit`, `av_sync`, `narrative_coherence`, `overall_quality`.
- All tests use pytest `tmp_path` for filesystem fixtures. No network. No real HF API. Synthetic DataFrames for stat-module tests.

---

## Task 1: Dependencies + package skeleton

Adds `pandas`, `scipy`, `statsmodels` to `requirements.txt` and creates an empty `eval/stats/__init__.py` so the package imports cleanly.

**Files:**
- Modify: `requirements.txt`
- Create: `eval/stats/__init__.py`

- [ ] **Step 1: Append dependencies to `requirements.txt`**

Append these lines to the end of `requirements.txt`:

```
# Phase 3 (rater study statistics)
pandas>=2.0.0
scipy>=1.11.0
statsmodels>=0.14.0
tabulate>=0.9.0
```

(`tabulate` is the backend `pandas.DataFrame.to_markdown` uses; pinning it explicitly avoids an `ImportError` at Task 6 runtime.)

- [ ] **Step 2: Install the new packages locally**

Run: `pip install "pandas>=2.0.0" "scipy>=1.11.0" "statsmodels>=0.14.0" "tabulate>=0.9.0"`
Expected: installs cleanly. Verify with `python -c "import pandas, scipy.stats, statsmodels.stats.inter_rater, tabulate; print('ok')"` → prints `ok`.

- [ ] **Step 3: Create the empty package init**

Create an empty file at `eval/stats/__init__.py`. Its presence marks the directory as a Python package; no content is needed.

- [ ] **Step 4: Confirm the package is importable**

Run: `python -c "import eval.stats; print('ok')"`
Expected: prints `ok` with no traceback.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt eval/stats/__init__.py
git commit -m "feat(stats): add pandas/scipy/statsmodels deps and stats package skeleton"
```

---

## Task 2: `eval/stats/loader.py` — file discovery + tidy DataFrame

Walks `responses_dump/responses/<rater>/<video>.json`, joins each row against the manifest, optionally joins per-`video_id` against `eval_results/<run_id>.json`, and returns `(ratings_df, auto_df, overall_comments)`.

**Files:**
- Create: `eval/stats/loader.py`
- Test: `tests/test_stats_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats_loader.py`:

```python
import json
from pathlib import Path

import pandas as pd
import pytest

from eval.stats import loader


DIMS = [
    "visual_quality", "narration_clarity", "music_mood_fit",
    "av_sync", "narrative_coherence", "overall_quality",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_submission(rater_id: str, video_id: str, ratings: dict, run_id: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "rater_id": rater_id,
        "video_id": video_id,
        "video_filename": f"{video_id}.mp4",
        "video_prompt": "p",
        "video_sentiment": "neutral",
        "video_style": "cinematic",
        "video_run_id": run_id,
        "ratings": ratings,
        "comment": "",
        "video_order_index": 0,
        "submitted_at_utc": "2026-07-01T00:00:00Z",
        "app_version": "0.2.0",
    }


def _make_manifest(entries: list[dict]) -> list[dict]:
    return entries


def _full_ratings(value: int = 4) -> dict:
    return {k: value for k in DIMS}


@pytest.fixture
def fake_study(tmp_path):
    responses = tmp_path / "responses_dump" / "responses"
    manifest_path = tmp_path / "manifest.json"
    eval_runs = tmp_path / "eval_results"

    manifest = _make_manifest([
        {"id": "v01", "filename": "v01.mp4", "prompt": "p1",
         "sentiment": "sad", "style": "cinematic", "run_id": "run_a"},
        {"id": "v02", "filename": "v02.mp4", "prompt": "p2",
         "sentiment": "happy", "style": "anime", "run_id": "run_b"},
        {"id": "v03", "filename": "v03.mp4", "prompt": "p3",
         "sentiment": "neutral", "style": "noir", "run_id": None},
    ])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _write_json(responses / "rater_a" / "v01.json", _make_submission("rater_a", "v01", _full_ratings(5), "run_a"))
    _write_json(responses / "rater_a" / "v02.json", _make_submission("rater_a", "v02", _full_ratings(3), "run_b"))
    _write_json(responses / "rater_a" / "_overall.json", {"rater_id": "rater_a", "overall_comment": "ok"})
    _write_json(responses / "rater_b" / "v01.json", _make_submission("rater_b", "v01", _full_ratings(4), "run_a"))

    _write_json(eval_runs / "run_a.json", {
        "run_id": "run_a",
        "clip_score": {"per_scene": [0.3], "mean": 0.31},
        "wer": {"per_scene": [0.1], "mean": 0.10},
        "sync_error_ms": {"per_entry": [80.0], "mean": 80.0, "max": 80.0, "pass": True},
    })
    _write_json(eval_runs / "run_b.json", {
        "run_id": "run_b",
        "clip_score": {"per_scene": [0.25], "mean": 0.25},
        "wer": {"per_scene": [0.20], "mean": 0.20},
        "sync_error_ms": {"per_entry": [120.0], "mean": 120.0, "max": 120.0, "pass": True},
    })

    return {
        "responses_dir": tmp_path / "responses_dump",
        "manifest_path": manifest_path,
        "eval_runs_dir": eval_runs,
    }


def test_load_all_returns_three_objects(fake_study):
    ratings_df, auto_df, overall_comments = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    assert isinstance(ratings_df, pd.DataFrame)
    assert isinstance(auto_df, pd.DataFrame)
    assert isinstance(overall_comments, list)


def test_load_all_ratings_df_has_long_format_shape(fake_study):
    ratings_df, _, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    # 3 submissions × 6 dimensions = 18 rows
    assert len(ratings_df) == 18
    expected_cols = {
        "rater_id", "video_id", "dimension", "rating",
        "video_sentiment", "video_style", "video_run_id", "submitted_at_utc",
    }
    assert expected_cols.issubset(set(ratings_df.columns))


def test_load_all_joins_manifest_sentiment_and_style(fake_study):
    ratings_df, _, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    v01_rows = ratings_df[ratings_df["video_id"] == "v01"]
    assert (v01_rows["video_sentiment"] == "sad").all()
    assert (v01_rows["video_style"] == "cinematic").all()


def test_load_all_overall_comments_captured(fake_study):
    _, _, overall_comments = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    assert len(overall_comments) == 1
    assert overall_comments[0]["rater_id"] == "rater_a"
    assert overall_comments[0]["overall_comment"] == "ok"


def test_load_all_auto_df_uses_metric_means(fake_study):
    _, auto_df, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    # auto_df is keyed by video_id, joining via manifest's run_id
    row_v01 = auto_df[auto_df["video_id"] == "v01"].iloc[0]
    assert row_v01["clip_score"] == pytest.approx(0.31)
    assert row_v01["wer"] == pytest.approx(0.10)
    assert row_v01["sync_error_ms"] == pytest.approx(80.0)


def test_load_all_auto_df_has_nan_when_run_id_null(fake_study):
    _, auto_df, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    row_v03 = auto_df[auto_df["video_id"] == "v03"].iloc[0]
    assert pd.isna(row_v03["clip_score"])
    assert pd.isna(row_v03["wer"])
    assert pd.isna(row_v03["sync_error_ms"])


def test_load_all_skips_submission_with_unknown_video_id(tmp_path, fake_study):
    rogue = fake_study["responses_dir"] / "responses" / "rater_a" / "v99.json"
    rogue.write_text(json.dumps(_make_submission("rater_a", "v99", _full_ratings())), encoding="utf-8")
    ratings_df, _, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    assert "v99" not in set(ratings_df["video_id"])


def test_load_all_skips_malformed_json(tmp_path, fake_study):
    bad = fake_study["responses_dir"] / "responses" / "rater_a" / "v04.json"
    bad.write_text("not json at all{", encoding="utf-8")
    ratings_df, _, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], fake_study["eval_runs_dir"],
    )
    # 3 valid submissions stay, malformed one is dropped
    assert ratings_df["video_id"].nunique() == 2  # v01, v02 (rater_a's v01, v02 + rater_b's v01 → 2 distinct videos)


def test_load_all_works_without_eval_runs_dir(fake_study):
    ratings_df, auto_df, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], eval_runs_dir=None,
    )
    assert len(ratings_df) == 18
    # auto_df is still a DataFrame (possibly empty or all-NaN); shape is well-defined
    assert isinstance(auto_df, pd.DataFrame)
    assert {"video_id", "clip_score", "wer", "sync_error_ms"}.issubset(set(auto_df.columns))


def test_load_all_raises_when_responses_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="responses"):
        loader.load_all(
            tmp_path / "does_not_exist",
            tmp_path / "manifest.json",
            None,
        )
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_stats_loader.py -v`
Expected: All fail with `ModuleNotFoundError: No module named 'eval.stats.loader'`.

- [ ] **Step 3: Implement the loader**

Create `eval/stats/loader.py`:

```python
import json
from pathlib import Path

import pandas as pd

DIMENSIONS = [
    "visual_quality", "narration_clarity", "music_mood_fit",
    "av_sync", "narrative_coherence", "overall_quality",
]


def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["id"]: e for e in entries}


def _read_submission(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: skipping malformed submission {path}: {exc}")
        return None


def _load_auto_metrics_by_run_id(eval_runs_dir: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for path in sorted(eval_runs_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warning: skipping malformed metric file {path}: {exc}")
            continue
        run_id = data.get("run_id")
        if not run_id:
            continue
        metrics[run_id] = {
            "clip_score": (data.get("clip_score") or {}).get("mean"),
            "wer": (data.get("wer") or {}).get("mean"),
            "sync_error_ms": (data.get("sync_error_ms") or {}).get("mean"),
        }
    return metrics


def load_all(
    responses_dir: Path,
    manifest_path: Path,
    eval_runs_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (ratings_df, auto_df, overall_comments)."""
    responses_root = Path(responses_dir) / "responses"
    if not responses_root.is_dir():
        raise FileNotFoundError(f"responses directory not found: {responses_root}")

    manifest_by_id = _load_manifest(Path(manifest_path))

    rows: list[dict] = []
    overall_comments: list[dict] = []

    for rater_dir in sorted(responses_root.iterdir()):
        if not rater_dir.is_dir():
            continue
        for submission_path in sorted(rater_dir.glob("*.json")):
            payload = _read_submission(submission_path)
            if payload is None:
                continue
            stem = submission_path.stem
            if stem == "_overall":
                overall_comments.append(payload)
                continue
            video_id = payload.get("video_id")
            if video_id not in manifest_by_id:
                print(f"warning: skipping submission for unknown video_id={video_id!r} ({submission_path})")
                continue
            manifest_entry = manifest_by_id[video_id]
            ratings = payload.get("ratings") or {}
            for dim in DIMENSIONS:
                rating = ratings.get(dim)
                if rating is None:
                    continue
                rows.append({
                    "rater_id": payload.get("rater_id"),
                    "video_id": video_id,
                    "dimension": dim,
                    "rating": int(rating),
                    "video_sentiment": manifest_entry["sentiment"],
                    "video_style": manifest_entry["style"],
                    "video_run_id": manifest_entry.get("run_id"),
                    "submitted_at_utc": payload.get("submitted_at_utc"),
                })

    ratings_df = pd.DataFrame(rows, columns=[
        "rater_id", "video_id", "dimension", "rating",
        "video_sentiment", "video_style", "video_run_id", "submitted_at_utc",
    ])

    metrics_by_run_id: dict[str, dict[str, float]] = {}
    if eval_runs_dir is not None:
        metrics_by_run_id = _load_auto_metrics_by_run_id(Path(eval_runs_dir))

    auto_rows: list[dict] = []
    for video_id, entry in manifest_by_id.items():
        run_id = entry.get("run_id")
        metrics = metrics_by_run_id.get(run_id, {}) if run_id else {}
        auto_rows.append({
            "video_id": video_id,
            "clip_score": metrics.get("clip_score"),
            "wer": metrics.get("wer"),
            "sync_error_ms": metrics.get("sync_error_ms"),
        })
    auto_df = pd.DataFrame(auto_rows, columns=["video_id", "clip_score", "wer", "sync_error_ms"])

    return ratings_df, auto_df, overall_comments
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_stats_loader.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/stats/loader.py tests/test_stats_loader.py
git commit -m "feat(stats): add loader for rater submissions, manifest, and auto metrics"
```

---

## Task 3: `eval/stats/mos.py` — Mean Opinion Score

Computes per-video MOS, sentiment/style breakdowns, and the headline summary.

**Files:**
- Create: `eval/stats/mos.py`
- Test: `tests/test_stats_mos.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats_mos.py`:

```python
import pandas as pd
import pytest

from eval.stats import mos


DIMS = [
    "visual_quality", "narration_clarity", "music_mood_fit",
    "av_sync", "narrative_coherence", "overall_quality",
]


def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "rater_id", "video_id", "dimension", "rating",
        "video_sentiment", "video_style",
    ])


def _fully_rated(rater: str, video: str, sentiment: str, style: str, value: int) -> list[tuple]:
    return [(rater, video, dim, value, sentiment, style) for dim in DIMS]


def test_compute_mos_per_video_means_match():
    rows = _fully_rated("ra", "v01", "sad", "cinematic", 5) + _fully_rated("rb", "v01", "sad", "cinematic", 3)
    df = _df(rows)
    out = mos.compute_mos_per_video(df)
    row = out[out["video_id"] == "v01"].iloc[0]
    for dim in DIMS:
        assert row[dim] == pytest.approx(4.0)
    assert row["n_raters"] == 2


def test_compute_mos_per_video_columns():
    df = _df(_fully_rated("ra", "v01", "neutral", "anime", 4))
    out = mos.compute_mos_per_video(df)
    expected = {"video_id", "n_raters"} | set(DIMS)
    assert expected.issubset(set(out.columns))


def test_compute_mos_by_sentiment_shape():
    rows = (
        _fully_rated("ra", "v01", "sad", "cinematic", 5)
        + _fully_rated("rb", "v02", "happy", "anime", 4)
        + _fully_rated("rc", "v03", "neutral", "noir", 3)
    )
    out = mos.compute_mos_by_sentiment(_df(rows))
    assert set(out["video_sentiment"]) == {"sad", "happy", "neutral"}
    assert len(out) == 3
    for dim in DIMS:
        assert dim in out.columns


def test_compute_mos_by_style_means():
    rows = (
        _fully_rated("ra", "v01", "sad", "cinematic", 5)
        + _fully_rated("rb", "v02", "sad", "cinematic", 3)
        + _fully_rated("rc", "v03", "happy", "anime", 4)
    )
    out = mos.compute_mos_by_style(_df(rows))
    cine_row = out[out["video_style"] == "cinematic"].iloc[0]
    for dim in DIMS:
        assert cine_row[dim] == pytest.approx(4.0)


def test_compute_summary_includes_counts_and_means():
    rows = _fully_rated("ra", "v01", "sad", "cinematic", 4) + _fully_rated("rb", "v02", "happy", "anime", 5)
    summary = mos.compute_summary(_df(rows))
    assert summary["n_raters"] == 2
    assert summary["n_videos"] == 2
    for dim in DIMS:
        assert summary[f"mos_{dim}"] == pytest.approx(4.5)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_stats_mos.py -v`
Expected: All fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement MOS**

Create `eval/stats/mos.py`:

```python
import pandas as pd

from eval.stats.loader import DIMENSIONS


def _pivot_means(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    means = (
        df.groupby([index_col, "dimension"])["rating"].mean().reset_index()
    )
    wide = means.pivot(index=index_col, columns="dimension", values="rating").reset_index()
    # Ensure all expected dimension columns exist even if some are empty
    for dim in DIMENSIONS:
        if dim not in wide.columns:
            wide[dim] = float("nan")
    return wide[[index_col] + DIMENSIONS]


def compute_mos_per_video(ratings_df: pd.DataFrame) -> pd.DataFrame:
    wide = _pivot_means(ratings_df, "video_id")
    n_raters = (
        ratings_df.groupby("video_id")["rater_id"].nunique().reset_index(name="n_raters")
    )
    return wide.merge(n_raters, on="video_id", how="left")


def compute_mos_by_sentiment(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return _pivot_means(ratings_df, "video_sentiment")


def compute_mos_by_style(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return _pivot_means(ratings_df, "video_style")


def compute_summary(ratings_df: pd.DataFrame) -> dict:
    summary = {
        "n_raters": int(ratings_df["rater_id"].nunique()),
        "n_videos": int(ratings_df["video_id"].nunique()),
    }
    per_dim_means = ratings_df.groupby("dimension")["rating"].mean()
    for dim in DIMENSIONS:
        summary[f"mos_{dim}"] = float(per_dim_means.get(dim, float("nan")))
    return summary
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `pytest tests/test_stats_mos.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/stats/mos.py tests/test_stats_mos.py
git commit -m "feat(stats): add MOS per-video, sentiment/style breakdowns, and summary"
```

---

## Task 4: `eval/stats/kappa.py` — Fleiss' Kappa per dimension

Computes Fleiss' kappa for each of the 6 dimensions independently.

**Files:**
- Create: `eval/stats/kappa.py`
- Test: `tests/test_stats_kappa.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats_kappa.py`:

```python
import pandas as pd
import pytest

from eval.stats import kappa


DIMS = [
    "visual_quality", "narration_clarity", "music_mood_fit",
    "av_sync", "narrative_coherence", "overall_quality",
]


def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["rater_id", "video_id", "dimension", "rating"])


def _perfect_agreement_df(n_videos: int = 5, n_raters: int = 4) -> pd.DataFrame:
    rows = []
    for v in range(n_videos):
        rating = (v % 5) + 1
        for r in range(n_raters):
            for dim in DIMS:
                rows.append((f"r{r}", f"v{v:02d}", dim, rating))
    return _df(rows)


def test_kappa_per_dimension_returns_one_row_per_dimension():
    df = _perfect_agreement_df()
    out = kappa.compute_fleiss_kappa_per_dimension(df, min_raters_per_video=2)
    assert set(out["dimension"]) == set(DIMS)
    assert len(out) == 6


def test_kappa_perfect_agreement_is_one():
    df = _perfect_agreement_df()
    out = kappa.compute_fleiss_kappa_per_dimension(df, min_raters_per_video=2)
    assert out["kappa"].min() == pytest.approx(1.0)


def test_kappa_interpretation_band():
    df = _perfect_agreement_df()
    out = kappa.compute_fleiss_kappa_per_dimension(df, min_raters_per_video=2)
    assert (out["interpretation"] == "almost perfect").all()


def test_kappa_drops_videos_below_min_raters(monkeypatch):
    # 3 videos, 1 with only 1 rater → that video should be dropped
    rows = []
    for v in ["v01", "v02"]:
        for r in ["ra", "rb", "rc"]:
            for dim in DIMS:
                rows.append((r, v, dim, 3))
    for dim in DIMS:
        rows.append(("ra", "v03", dim, 5))  # only 1 rater for v03
    df = _df(rows)
    out = kappa.compute_fleiss_kappa_per_dimension(df, min_raters_per_video=2)
    assert (out["n_videos_used"] == 2).all()


def test_kappa_returns_nan_with_insufficient_data_message():
    # only 1 rater on 1 video
    rows = [("ra", "v01", dim, 4) for dim in DIMS]
    out = kappa.compute_fleiss_kappa_per_dimension(_df(rows), min_raters_per_video=2)
    assert out["n_videos_used"].max() == 0
    assert (out["interpretation"] == "insufficient data").all()
    assert out["kappa"].isna().all()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_stats_kappa.py -v`
Expected: All fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement Kappa**

Create `eval/stats/kappa.py`:

```python
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

from eval.stats.loader import DIMENSIONS

RATING_VALUES = [1, 2, 3, 4, 5]


def _interpret(kappa_value: float) -> str:
    if kappa_value < 0.0:
        return "poor"
    if kappa_value <= 0.20:
        return "slight"
    if kappa_value <= 0.40:
        return "fair"
    if kappa_value <= 0.60:
        return "moderate"
    if kappa_value <= 0.80:
        return "substantial"
    return "almost perfect"


def _per_dimension_kappa(ratings_df: pd.DataFrame, dimension: str, min_raters_per_video: int) -> tuple[float, int]:
    df = ratings_df[ratings_df["dimension"] == dimension]
    if df.empty:
        return float("nan"), 0

    matrix_rows: list[list[int]] = []
    for _video_id, grp in df.groupby("video_id"):
        counts = [int((grp["rating"] == val).sum()) for val in RATING_VALUES]
        if sum(counts) < min_raters_per_video:
            continue
        matrix_rows.append(counts)

    if len(matrix_rows) < 2:
        return float("nan"), len(matrix_rows)

    # statsmodels' fleiss_kappa requires every row to have the same total (n raters).
    # If raters skipped this dimension on some videos, some rows may sum to fewer than others.
    # Drop rows whose total differs from the modal total so the input is well-formed.
    totals = [sum(r) for r in matrix_rows]
    modal_total = max(set(totals), key=totals.count)
    matrix_rows = [r for r, t in zip(matrix_rows, totals) if t == modal_total]
    if len(matrix_rows) < 2:
        return float("nan"), len(matrix_rows)

    arr = np.array(matrix_rows, dtype=int)
    return float(fleiss_kappa(arr, method="fleiss")), int(arr.shape[0])


def compute_fleiss_kappa_per_dimension(
    ratings_df: pd.DataFrame,
    min_raters_per_video: int = 2,
) -> pd.DataFrame:
    rows: list[dict] = []
    for dim in DIMENSIONS:
        kappa_value, n_videos_used = _per_dimension_kappa(ratings_df, dim, min_raters_per_video)
        if n_videos_used < 2 or pd.isna(kappa_value):
            interpretation = "insufficient data"
            kappa_out = float("nan")
        else:
            interpretation = _interpret(kappa_value)
            kappa_out = kappa_value
        rows.append({
            "dimension": dim,
            "kappa": kappa_out,
            "n_videos_used": n_videos_used,
            "interpretation": interpretation,
        })
    return pd.DataFrame(rows, columns=["dimension", "kappa", "n_videos_used", "interpretation"])
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `pytest tests/test_stats_kappa.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/stats/kappa.py tests/test_stats_kappa.py
git commit -m "feat(stats): add per-dimension Fleiss' kappa with Landis-Koch interpretation"
```

---

## Task 5: `eval/stats/correlations.py` — Auto vs. human

Computes Pearson and Spearman correlations (with two-sided p-values) between automated metrics and matching subjective dimensions.

**Files:**
- Create: `eval/stats/correlations.py`
- Test: `tests/test_stats_correlations.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats_correlations.py`:

```python
import numpy as np
import pandas as pd
import pytest

from eval.stats import correlations


DIMS = [
    "visual_quality", "narration_clarity", "music_mood_fit",
    "av_sync", "narrative_coherence", "overall_quality",
]


def _per_video_mos(values: dict[str, list[float]], video_ids: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"video_id": video_ids})
    for dim in DIMS:
        df[dim] = values.get(dim, [3.0] * len(video_ids))
    df["n_raters"] = 3
    return df


def _auto_df(video_ids: list[str], clip: list[float], wer: list[float], sync: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "video_id": video_ids,
        "clip_score": clip,
        "wer": wer,
        "sync_error_ms": sync,
    })


def test_correlations_returns_four_pairs():
    video_ids = [f"v{i:02d}" for i in range(1, 11)]
    per_video = _per_video_mos({}, video_ids)
    auto = _auto_df(video_ids, [0.3] * 10, [0.1] * 10, [80.0] * 10)
    out = correlations.compute_correlations(per_video, auto)
    assert len(out) == 4
    expected_pairs = {
        "clip_score ↔ visual_quality",
        "wer ↔ narration_clarity",
        "sync_error_ms ↔ av_sync",
        "composite ↔ overall_quality",
    }
    assert set(out["pair"]) == expected_pairs


def test_correlations_perfect_positive():
    video_ids = [f"v{i:02d}" for i in range(1, 11)]
    visual_ratings = [1.0 + 0.4 * i for i in range(10)]
    per_video = _per_video_mos({"visual_quality": visual_ratings}, video_ids)
    # CLIP scores perfectly correlated with visual ratings
    clip = [0.1 + 0.05 * i for i in range(10)]
    auto = _auto_df(video_ids, clip, [0.1] * 10, [80.0] * 10)
    out = correlations.compute_correlations(per_video, auto)
    row = out[out["pair"] == "clip_score ↔ visual_quality"].iloc[0]
    assert row["pearson_r"] == pytest.approx(1.0)
    assert row["pearson_p"] < 0.001
    assert row["sig"] == "*"
    assert row["n"] == 10


def test_correlations_negated_metric_pair_uses_sign_correction():
    # WER lower is better — narration_clarity rating higher is better.
    # If WER decreases as narration_clarity increases, correlation between -WER and clarity is +1.
    # The function should report a POSITIVE coefficient for this perfectly aligned pair.
    video_ids = [f"v{i:02d}" for i in range(1, 11)]
    clarity = [1.0 + 0.4 * i for i in range(10)]
    per_video = _per_video_mos({"narration_clarity": clarity}, video_ids)
    wer = [0.5 - 0.04 * i for i in range(10)]  # decreases as clarity rises
    auto = _auto_df(video_ids, [0.3] * 10, wer, [80.0] * 10)
    out = correlations.compute_correlations(per_video, auto)
    row = out[out["pair"] == "wer ↔ narration_clarity"].iloc[0]
    assert row["pearson_r"] == pytest.approx(1.0)


def test_correlations_drops_rows_with_nan_in_auto():
    video_ids = [f"v{i:02d}" for i in range(1, 11)]
    per_video = _per_video_mos({}, video_ids)
    clip = [0.3 if i < 8 else float("nan") for i in range(10)]
    auto = _auto_df(video_ids, clip, [0.1] * 10, [80.0] * 10)
    out = correlations.compute_correlations(per_video, auto)
    row = out[out["pair"] == "clip_score ↔ visual_quality"].iloc[0]
    assert row["n"] == 8


def test_correlations_sig_flag_blank_when_high_p():
    # near-zero correlation across 10 random points → p large
    video_ids = [f"v{i:02d}" for i in range(1, 11)]
    rng = np.random.default_rng(seed=42)
    visual = list(rng.uniform(1, 5, 10))
    clip = list(rng.uniform(0.2, 0.4, 10))
    per_video = _per_video_mos({"visual_quality": visual}, video_ids)
    auto = _auto_df(video_ids, clip, [0.1] * 10, [80.0] * 10)
    out = correlations.compute_correlations(per_video, auto)
    row = out[out["pair"] == "clip_score ↔ visual_quality"].iloc[0]
    # Don't assert exact r — just check sig flag is blank when p >= 0.05
    if max(row["pearson_p"], row["spearman_p"]) >= 0.05:
        assert row["sig"] == ""
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/test_stats_correlations.py -v`
Expected: All fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement correlations**

Create `eval/stats/correlations.py`:

```python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# (auto_metric_column, dimension_column, sign): sign=-1 means lower metric = better.
PAIRINGS: list[tuple[str, str, int]] = [
    ("clip_score", "visual_quality", +1),
    ("wer", "narration_clarity", -1),
    ("sync_error_ms", "av_sync", -1),
]


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([float("nan")] * len(series), index=series.index)
    return (series - series.mean()) / std


def _composite_auto(auto_df: pd.DataFrame) -> pd.Series:
    z_clip = _zscore(auto_df["clip_score"])
    z_neg_wer = _zscore(-auto_df["wer"])
    z_neg_sync = _zscore(-auto_df["sync_error_ms"])
    return z_clip + z_neg_wer + z_neg_sync


def _correlate_pair(metric_values: pd.Series, dim_values: pd.Series, sign: int) -> dict:
    paired = pd.concat(
        [metric_values.rename("metric"), dim_values.rename("dim")], axis=1
    ).dropna()
    n = int(len(paired))
    if n < 3:
        return {
            "n": n,
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }
    metric = sign * paired["metric"].to_numpy()
    dim = paired["dim"].to_numpy()
    r, p_r = pearsonr(metric, dim)
    rho, p_rho = spearmanr(metric, dim)
    return {
        "n": n,
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
    }


def compute_correlations(per_video_mos: pd.DataFrame, auto_df: pd.DataFrame) -> pd.DataFrame:
    joined = per_video_mos.merge(auto_df, on="video_id", how="inner")
    rows: list[dict] = []

    for metric_col, dim_col, sign in PAIRINGS:
        result = _correlate_pair(joined[metric_col], joined[dim_col], sign)
        rows.append({
            "pair": f"{metric_col} ↔ {dim_col}",
            **result,
            "sig": "*" if (
                not np.isnan(result["pearson_p"]) and not np.isnan(result["spearman_p"])
                and min(result["pearson_p"], result["spearman_p"]) < 0.05
            ) else "",
        })

    composite = _composite_auto(joined[["clip_score", "wer", "sync_error_ms"]])
    composite_result = _correlate_pair(composite, joined["overall_quality"], +1)
    rows.append({
        "pair": "composite ↔ overall_quality",
        **composite_result,
        "sig": "*" if (
            not np.isnan(composite_result["pearson_p"]) and not np.isnan(composite_result["spearman_p"])
            and min(composite_result["pearson_p"], composite_result["spearman_p"]) < 0.05
        ) else "",
    })

    return pd.DataFrame(rows, columns=[
        "pair", "n",
        "pearson_r", "pearson_p",
        "spearman_rho", "spearman_p",
        "sig",
    ])
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `pytest tests/test_stats_correlations.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/stats/correlations.py tests/test_stats_correlations.py
git commit -m "feat(stats): add auto-vs-human correlations with p-values and composite metric"
```

---

## Task 6: `eval/stats/formatters.py` — CSV + Markdown writers

Writes each computed DataFrame to both `.csv` (full precision) and `.md` (rounded, GitHub-flavored).

**Files:**
- Create: `eval/stats/formatters.py`
- Test: `tests/test_stats_formatters.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats_formatters.py`:

```python
import pandas as pd
import pytest

from eval.stats import formatters


def test_write_table_creates_both_files(tmp_path):
    df = pd.DataFrame({"video_id": ["v01", "v02"], "mos": [4.0, 3.5]})
    formatters.write_table(df, tmp_path, stem="mos_test")
    assert (tmp_path / "mos_test.csv").is_file()
    assert (tmp_path / "mos_test.md").is_file()


def test_write_table_csv_roundtrips(tmp_path):
    df = pd.DataFrame({"video_id": ["v01"], "mos": [4.0]})
    formatters.write_table(df, tmp_path, stem="t")
    loaded = pd.read_csv(tmp_path / "t.csv")
    assert loaded["video_id"].iloc[0] == "v01"
    assert loaded["mos"].iloc[0] == pytest.approx(4.0)


def test_write_table_markdown_contains_pipe_table(tmp_path):
    df = pd.DataFrame({"video_id": ["v01"], "mos": [4.0]})
    formatters.write_table(df, tmp_path, stem="t")
    md = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert "| video_id" in md
    assert "v01" in md
    assert "4.00" in md  # rounded to 2 dp


def test_write_summary_writes_csv_and_md(tmp_path):
    summary = {"n_raters": 20, "n_videos": 20, "mos_visual_quality": 3.85}
    formatters.write_summary(summary, tmp_path)
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "summary.md").is_file()
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "n_raters" in md
    assert "20" in md


def test_write_provenance_writes_text_file(tmp_path):
    formatters.write_provenance(tmp_path, args={"responses": "responses_dump/"}, input_counts={"n_submissions": 350})
    prov_path = tmp_path / "provenance.txt"
    assert prov_path.is_file()
    text = prov_path.read_text(encoding="utf-8")
    assert "pandas" in text
    assert "scipy" in text
    assert "statsmodels" in text
    assert "responses_dump/" in text
    assert "350" in text
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/test_stats_formatters.py -v`
Expected: All fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement formatters**

Create `eval/stats/formatters.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _df_to_markdown(df: pd.DataFrame, decimals: int = 2) -> str:
    rounded = df.copy()
    for col in rounded.select_dtypes(include="number").columns:
        rounded[col] = rounded[col].round(decimals)
    return rounded.to_markdown(index=False, floatfmt=f".{decimals}f")


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, decimals: int = 2) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_df_to_markdown(df, decimals=decimals) + "\n", encoding="utf-8")


def write_summary(summary: dict, out_dir: Path) -> None:
    df = pd.DataFrame([summary])
    write_table(df, out_dir, stem="summary")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("pandas", "scipy", "statsmodels"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def write_provenance(out_dir: Path, args: dict, input_counts: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = _package_versions()
    lines = [
        f"# DreamScapeAI Phase 3 stats provenance",
        f"# generated_at_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "[arguments]",
    ]
    for key, value in args.items():
        lines.append(f"{key} = {value}")
    lines.append("")
    lines.append("[input_counts]")
    for key, value in input_counts.items():
        lines.append(f"{key} = {value}")
    lines.append("")
    lines.append("[package_versions]")
    for name, version in versions.items():
        lines.append(f"{name} = {version}")
    (out_dir / "provenance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `pytest tests/test_stats_formatters.py -v`
Expected: 5 passed. (`tabulate` was installed in Task 1, so `to_markdown` works out of the box.)

- [ ] **Step 5: Commit**

```bash
git add eval/stats/formatters.py tests/test_stats_formatters.py
git commit -m "feat(stats): add CSV/Markdown table writers and provenance file"
```

---

## Task 7: `eval/stats/cli.py` — argparse + orchestrator + end-to-end test

Ties everything together. One CLI command emits all six tables and `provenance.txt`.

**Files:**
- Create: `eval/stats/cli.py`
- Test: `tests/test_stats_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest


DIMS = [
    "visual_quality", "narration_clarity", "music_mood_fit",
    "av_sync", "narrative_coherence", "overall_quality",
]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def study_dirs(tmp_path):
    responses = tmp_path / "responses_dump" / "responses"
    manifest = tmp_path / "manifest.json"
    eval_runs = tmp_path / "eval_results"
    out = tmp_path / "thesis_tables"

    manifest_data = [
        {"id": f"v{i:02d}", "filename": f"v{i:02d}.mp4", "prompt": "p",
         "sentiment": ["sad", "happy", "neutral"][i % 3],
         "style": ["cinematic", "anime", "noir"][i % 3],
         "run_id": f"run_{i:02d}"}
        for i in range(1, 6)  # 5 videos
    ]
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    for r in ["ra", "rb", "rc"]:
        for entry in manifest_data:
            ratings = {dim: 4 for dim in DIMS}
            _write(responses / r / f"{entry['id']}.json", {
                "schema_version": 1,
                "rater_id": r,
                "video_id": entry["id"],
                "video_filename": entry["filename"],
                "video_prompt": entry["prompt"],
                "video_sentiment": entry["sentiment"],
                "video_style": entry["style"],
                "video_run_id": entry["run_id"],
                "ratings": ratings,
                "comment": "",
                "video_order_index": 0,
                "submitted_at_utc": "2026-07-01T00:00:00Z",
                "app_version": "0.2.0",
            })

    for entry in manifest_data:
        _write(eval_runs / f"{entry['run_id']}.json", {
            "run_id": entry["run_id"],
            "clip_score": {"per_scene": [0.3], "mean": 0.3 + 0.01 * int(entry["id"][1:])},
            "wer": {"per_scene": [0.1], "mean": 0.1},
            "sync_error_ms": {"per_entry": [80.0], "mean": 80.0, "max": 80.0, "pass": True},
        })

    return {
        "responses": tmp_path / "responses_dump",
        "manifest": manifest,
        "eval_runs": eval_runs,
        "out": out,
    }


def test_cli_creates_all_six_table_pairs(study_dirs):
    result = subprocess.run(
        [sys.executable, "-m", "eval.stats",
         "--responses", str(study_dirs["responses"]),
         "--manifest", str(study_dirs["manifest"]),
         "--metrics", str(study_dirs["eval_runs"]),
         "--out", str(study_dirs["out"])],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    expected_stems = [
        "summary", "mos_per_video", "mos_by_sentiment", "mos_by_style",
        "kappa_per_dimension", "auto_vs_human_correlations",
    ]
    for stem in expected_stems:
        assert (study_dirs["out"] / f"{stem}.csv").is_file(), f"missing {stem}.csv"
        assert (study_dirs["out"] / f"{stem}.md").is_file(), f"missing {stem}.md"
    assert (study_dirs["out"] / "provenance.txt").is_file()


def test_cli_skips_correlations_when_no_metrics(study_dirs):
    out_dir = study_dirs["out"]
    result = subprocess.run(
        [sys.executable, "-m", "eval.stats",
         "--responses", str(study_dirs["responses"]),
         "--manifest", str(study_dirs["manifest"]),
         "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    # subjective tables still produced
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "mos_per_video.csv").is_file()
    assert (out_dir / "kappa_per_dimension.csv").is_file()
    # correlations table skipped
    assert not (out_dir / "auto_vs_human_correlations.csv").is_file()
    assert "skipping correlations" in (result.stdout + result.stderr).lower()


def test_cli_fails_when_responses_dir_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "eval.stats",
         "--responses", str(tmp_path / "does_not_exist"),
         "--manifest", str(tmp_path / "manifest.json"),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/test_stats_cli.py -v`
Expected: All fail with `No module named eval.stats.__main__` (or `cli`).

- [ ] **Step 3: Implement the CLI**

Create `eval/stats/cli.py`:

```python
import argparse
import sys
from pathlib import Path

from eval.stats import correlations, formatters, kappa, loader, mos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.stats",
        description="Compute Phase 3 rater study statistics and write thesis tables",
    )
    parser.add_argument("--responses", type=Path, required=True,
                        help="Directory containing responses/<rater_id>/<video_id>.json")
    parser.add_argument("--manifest", type=Path,
                        default=Path("study_videos/manifest.json"),
                        help="Path to study_videos/manifest.json")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Directory of Phase 1 metric JSONs (eval_results/<run_id>.json)")
    parser.add_argument("--out", type=Path, default=Path("thesis_tables"),
                        help="Output directory for tables and provenance")
    parser.add_argument("--min-raters-per-video", type=int, default=2,
                        help="Minimum raters per video for kappa (default 2)")
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        ratings_df, auto_df, overall_comments = loader.load_all(
            args.responses, args.manifest, args.metrics,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out.exists() and any(args.out.iterdir()):
        print(f"overwriting existing files in {args.out}")

    per_video = mos.compute_mos_per_video(ratings_df)
    by_sentiment = mos.compute_mos_by_sentiment(ratings_df)
    by_style = mos.compute_mos_by_style(ratings_df)
    summary = mos.compute_summary(ratings_df)

    kappa_df = kappa.compute_fleiss_kappa_per_dimension(
        ratings_df, min_raters_per_video=args.min_raters_per_video,
    )
    # add mean kappa across dimensions to the summary
    valid_kappas = kappa_df["kappa"].dropna()
    summary["mean_kappa"] = float(valid_kappas.mean()) if not valid_kappas.empty else float("nan")

    formatters.write_summary(summary, args.out)
    formatters.write_table(per_video, args.out, stem="mos_per_video")
    formatters.write_table(by_sentiment, args.out, stem="mos_by_sentiment")
    formatters.write_table(by_style, args.out, stem="mos_by_style")
    formatters.write_table(kappa_df, args.out, stem="kappa_per_dimension")

    if args.metrics is not None:
        corr_df = correlations.compute_correlations(per_video, auto_df)
        formatters.write_table(corr_df, args.out, stem="auto_vs_human_correlations")
    else:
        print("auto metrics not provided; skipping correlations")

    formatters.write_provenance(
        args.out,
        args={
            "responses": str(args.responses),
            "manifest": str(args.manifest),
            "metrics": str(args.metrics) if args.metrics else "(none)",
            "out": str(args.out),
            "min_raters_per_video": args.min_raters_per_video,
        },
        input_counts={
            "n_submissions": int(ratings_df[["rater_id", "video_id"]].drop_duplicates().shape[0]),
            "n_overall_comments": len(overall_comments),
            "n_videos_in_manifest": int(auto_df.shape[0]),
        },
    )

    for stem in ["summary", "mos_per_video", "mos_by_sentiment",
                 "mos_by_style", "kappa_per_dimension"]:
        n_rows = sum(1 for _ in (args.out / f"{stem}.csv").open(encoding="utf-8")) - 1
        print(f"{stem}.csv  ({n_rows} rows)")
    if args.metrics is not None:
        n_rows = sum(1 for _ in (args.out / "auto_vs_human_correlations.csv").open(encoding="utf-8")) - 1
        print(f"auto_vs_human_correlations.csv  ({n_rows} rows)")

    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create the package `__main__.py` so `python -m eval.stats` works**

Create `eval/stats/__main__.py`:

```python
from eval.stats.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `pytest tests/test_stats_cli.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the full stats test suite for regressions**

Run: `pytest tests/test_stats_loader.py tests/test_stats_mos.py tests/test_stats_kappa.py tests/test_stats_correlations.py tests/test_stats_formatters.py tests/test_stats_cli.py -v`
Expected: all stats tests pass; no regressions.

- [ ] **Step 7: Run the project-wide test suite as a final regression check**

Run: `pytest -q`
Expected: previously passing tests still pass; new tests pass.

- [ ] **Step 8: Commit**

```bash
git add eval/stats/cli.py eval/stats/__main__.py tests/test_stats_cli.py
git commit -m "feat(stats): add Phase 3 stats CLI and end-to-end test"
```

---

## Final Checks

- [ ] **Run full stats test suite:** `pytest tests/test_stats_*.py -q` — all ~25 tests pass.
- [ ] **Smoke-launch the CLI:** if you have a `responses_dump/` from a live study, run `python -m eval.stats --responses responses_dump/ --manifest study_videos/manifest.json --metrics eval_results/ --out thesis_tables/`. Inspect each generated `.md` for correctness.
- [ ] **Inspect the diff:** `git log --oneline main..HEAD` — should show 7 focused commits, one per task.

Once green, hand off to `superpowers:finishing-a-development-branch`.
