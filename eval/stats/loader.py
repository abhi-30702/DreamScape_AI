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
