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
    clip = [0.1 + 0.05 * i for i in range(10)]
    auto = _auto_df(video_ids, clip, [0.1] * 10, [80.0] * 10)
    out = correlations.compute_correlations(per_video, auto)
    row = out[out["pair"] == "clip_score ↔ visual_quality"].iloc[0]
    assert row["pearson_r"] == pytest.approx(1.0)
    assert row["pearson_p"] < 0.001
    assert row["sig"] == "*"
    assert row["n"] == 10


def test_correlations_negated_metric_pair_uses_sign_correction():
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
    video_ids = [f"v{i:02d}" for i in range(1, 11)]
    rng = np.random.default_rng(seed=42)
    visual = list(rng.uniform(1, 5, 10))
    clip = list(rng.uniform(0.2, 0.4, 10))
    per_video = _per_video_mos({"visual_quality": visual}, video_ids)
    auto = _auto_df(video_ids, clip, [0.1] * 10, [80.0] * 10)
    out = correlations.compute_correlations(per_video, auto)
    row = out[out["pair"] == "clip_score ↔ visual_quality"].iloc[0]
    if max(row["pearson_p"], row["spearman_p"]) >= 0.05:
        assert row["sig"] == ""
