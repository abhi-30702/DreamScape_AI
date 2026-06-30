import json
import re
from datetime import datetime, timezone
from pathlib import Path

APP_VERSION = "0.2.0"

DIMENSIONS = [
    ("visual_quality", "Visual quality", "How sharp / coherent the imagery is"),
    ("narration_clarity", "Narration clarity", "How clearly the voice can be understood"),
    ("music_mood_fit", "Music–mood fit", "How well the music matches the story's tone"),
    ("av_sync", "A/V sync", "How well visuals, narration, and subtitles line up"),
    ("narrative_coherence", "Narrative coherence", "Whether the story makes sense from start to finish"),
    ("overall_quality", "Overall quality", "Your holistic impression of the video"),
]

_ID_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_start(rater_id: str, consent: bool) -> dict:
    if not consent:
        return {"ok": False, "error": "You must give consent to start."}
    if not _ID_RE.match(rater_id or ""):
        return {
            "ok": False,
            "error": "ID must be 3–32 characters, letters/numbers/underscore only.",
        }
    return {"ok": True, "error": None}


def compute_start_state(
    rater_id: str,
    completed_ids: set[str],
    has_overall: bool,
    manifest: list[dict],
) -> dict:
    total = len(manifest)
    if has_overall:
        return {
            "rater_id": rater_id,
            "current_index": total,
            "total": total,
            "status": "all_done",
        }
    # First index in manifest order whose id is not in completed_ids.
    next_index = total
    for i, entry in enumerate(manifest):
        if entry["id"] not in completed_ids:
            next_index = i
            break
    status = "overall_pending" if next_index == total else "rating"
    return {
        "rater_id": rater_id,
        "current_index": next_index,
        "total": total,
        "status": status,
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_submission_payload(
    rater_id: str,
    manifest_entry: dict,
    order_index: int,
    ratings: dict,
    comment: str,
) -> dict:
    return {
        "schema_version": 1,
        "rater_id": rater_id,
        "video_id": manifest_entry["id"],
        "video_filename": manifest_entry["filename"],
        "video_prompt": manifest_entry["prompt"],
        "video_sentiment": manifest_entry["sentiment"],
        "video_style": manifest_entry["style"],
        "video_run_id": manifest_entry.get("run_id"),
        "ratings": dict(ratings),
        "comment": comment or "",
        "video_order_index": order_index,
        "submitted_at_utc": _now_utc_iso(),
        "app_version": APP_VERSION,
    }


def build_overall_payload(rater_id: str, comment: str) -> dict:
    return {
        "schema_version": 1,
        "rater_id": rater_id,
        "overall_comment": comment or "",
        "submitted_at_utc": _now_utc_iso(),
        "app_version": APP_VERSION,
    }
