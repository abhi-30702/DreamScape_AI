import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ui import rater_storage

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


def on_start(rater_id: str, consent: bool, manifest: list[dict]) -> dict:
    """Handle the Start click. Returns {ok, error, state}."""
    check = validate_start(rater_id, consent)
    if not check["ok"]:
        return {"ok": False, "error": check["error"], "state": None}
    try:
        completed = rater_storage.list_completed(rater_id)
        has_overall = rater_storage.has_completed_overall(rater_id)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not load your progress: {exc}",
            "state": None,
        }
    state = compute_start_state(rater_id, completed, has_overall, manifest)
    return {"ok": True, "error": None, "state": state}


def _all_ratings_present(ratings: dict) -> bool:
    keys = {k for k, _, _ in DIMENSIONS}
    return all(ratings.get(k) is not None for k in keys)


def on_submit(state: dict, ratings: dict, comment: str, manifest: list[dict]) -> dict:
    """Handle the Submit & next click. Returns {ok, error, state}."""
    if not _all_ratings_present(ratings):
        return {
            "ok": False,
            "error": "Please rate every dimension before submitting.",
            "state": state,
        }
    idx = state["current_index"]
    entry = manifest[idx]
    payload = build_submission_payload(
        rater_id=state["rater_id"],
        manifest_entry=entry,
        order_index=idx,
        ratings=ratings,
        comment=comment,
    )
    try:
        rater_storage.save_response(state["rater_id"], entry["id"], payload)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not save your rating — please try again. ({exc})",
            "state": state,  # unchanged
        }
    next_idx = idx + 1
    next_status = "overall_pending" if next_idx >= state["total"] else "rating"
    new_state = {**state, "current_index": next_idx, "status": next_status}
    return {"ok": True, "error": None, "state": new_state}


def on_overall_submit(state: dict, overall_comment: str) -> dict:
    """Handle the final overall-comment submit. Returns {ok, error, state}."""
    payload = build_overall_payload(state["rater_id"], overall_comment)
    try:
        rater_storage.save_response(state["rater_id"], "_overall", payload)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not save your final comment — please try again. ({exc})",
            "state": state,
        }
    new_state = {**state, "status": "all_done"}
    return {"ok": True, "error": None, "state": new_state}
