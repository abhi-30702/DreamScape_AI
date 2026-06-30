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
    rows = [("ra", "v01", dim, 4) for dim in DIMS]
    out = kappa.compute_fleiss_kappa_per_dimension(_df(rows), min_raters_per_video=2)
    assert out["n_videos_used"].max() == 0
    assert (out["interpretation"] == "insufficient data").all()
    assert out["kappa"].isna().all()
