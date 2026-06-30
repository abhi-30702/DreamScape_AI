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
    # ddof=0 (population std) for descriptive z-scores; values feed into the composite
    # signal and are not used for separate inferential claims.
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
