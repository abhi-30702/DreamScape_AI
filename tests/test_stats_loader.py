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
    assert ratings_df["video_id"].nunique() == 2  # v01 and v02


def test_load_all_works_without_eval_runs_dir(fake_study):
    ratings_df, auto_df, _ = loader.load_all(
        fake_study["responses_dir"], fake_study["manifest_path"], eval_runs_dir=None,
    )
    assert len(ratings_df) == 18
    assert isinstance(auto_df, pd.DataFrame)
    assert {"video_id", "clip_score", "wer", "sync_error_ms"}.issubset(set(auto_df.columns))


def test_load_all_raises_when_responses_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="responses"):
        loader.load_all(
            tmp_path / "does_not_exist",
            tmp_path / "manifest.json",
            None,
        )
