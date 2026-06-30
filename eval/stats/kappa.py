import logging

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

from eval.stats.loader import DIMENSIONS

_log = logging.getLogger(__name__)

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

    totals = [sum(r) for r in matrix_rows]
    modal_total = max(set(totals), key=totals.count)
    n_dropped = sum(1 for t in totals if t != modal_total)
    if n_dropped:
        _log.warning(
            "kappa dimension=%s: dropped %d video(s) whose rater count differed "
            "from the modal total (%d); statsmodels.fleiss_kappa requires equal row sums",
            dimension, n_dropped, modal_total,
        )
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
