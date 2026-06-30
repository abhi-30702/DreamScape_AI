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
