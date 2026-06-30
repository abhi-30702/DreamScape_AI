import pandas as pd

from eval.stats.loader import DIMENSIONS


def _pivot_means(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    means = (
        df.groupby([index_col, "dimension"])["rating"].mean().reset_index()
    )
    wide = means.pivot(index=index_col, columns="dimension", values="rating").reset_index()
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
