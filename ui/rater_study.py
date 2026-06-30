import json
import re
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

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
    if state.get("status") == "all_done":
        return {"ok": True, "error": None, "state": state}
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


_INFO_SHEET = """\
### About this study

You will watch up to 20 short AI-generated videos (each 30–90 seconds) and rate
each on six dimensions. The full session takes about 20 minutes. Your ratings
are stored in a private dataset for academic analysis. You may stop at any time
by simply closing the browser tab — any ratings you already submitted are kept.

By starting, you confirm you are at least 18 years old and consent to participate.

Questions or concerns: contact the researcher.
"""

_STUDY_VIDEOS_DIR = Path(__file__).resolve().parent.parent / "study_videos"


def build_rater_tab() -> gr.Group:
    """Construct the Rate-videos tab content. Mount inside a `gr.TabItem`."""
    manifest = load_manifest(_STUDY_VIDEOS_DIR / "manifest.json")

    with gr.Group() as tab_root:
        # ---- Screen A: welcome / consent ----
        with gr.Group(visible=True) as screen_welcome:
            gr.Markdown(_INFO_SHEET)
            consent_chk = gr.Checkbox(label="I am 18+ and consent to participate", value=False)
            rater_id_tb = gr.Textbox(
                label="Choose an anonymous ID",
                placeholder="letters, numbers, underscore (e.g. reviewer_a)",
                max_lines=1,
            )
            welcome_error = gr.Markdown(visible=False)
            start_btn = gr.Button("Start", variant="primary", interactive=False)

        # ---- Screen B: rating loop ----
        with gr.Group(visible=False) as screen_rating:
            progress_md = gr.Markdown("Video 1 of 20")
            video_player = gr.Video(autoplay=False, show_label=False)
            gr.Markdown("Rate this video on each dimension (1 = Poor, 5 = Excellent):")
            radios: list[gr.Radio] = []
            for key, label, hint in DIMENSIONS:
                r = gr.Radio(choices=[1, 2, 3, 4, 5], label=label, info=hint)
                radios.append(r)
            comment_tb = gr.Textbox(
                label="Optional: anything specific you noticed?",
                lines=2,
            )
            rating_error = gr.Markdown(visible=False)
            submit_btn = gr.Button("Submit & next", variant="primary", interactive=False)

        # ---- Screen C: thank-you / overall ----
        with gr.Group(visible=False) as screen_thanks:
            thanks_md = gr.Markdown("### Thank you! Your ratings have been saved.")
            overall_tb = gr.Textbox(label="Any overall feedback?", lines=4)
            overall_error = gr.Markdown(visible=False)
            overall_btn = gr.Button("Submit final comment", variant="primary")
            done_md = gr.Markdown("Done — you can close this tab.", visible=False)

        state = gr.State({"status": "welcome"})

        # ---- Start-button enable logic ----
        def _toggle_start(consent, rid):
            ok = bool(consent) and bool(_ID_RE.match(rid or ""))
            return gr.update(interactive=ok)

        consent_chk.change(_toggle_start, [consent_chk, rater_id_tb], start_btn)
        rater_id_tb.change(_toggle_start, [consent_chk, rater_id_tb], start_btn)

        # ---- Submit-button enable logic ----
        def _toggle_submit(*values):
            ok = all(v is not None for v in values)
            return gr.update(interactive=ok)

        for r in radios:
            r.change(_toggle_submit, radios, submit_btn)

        # ---- Start click ----
        def _start_click(rater_id, consent):
            result = on_start(rater_id, consent, manifest)
            if not result["ok"]:
                return (
                    gr.update(visible=True, value=f"**{result['error']}**"),  # welcome_error
                    gr.update(),  # screen_welcome
                    gr.update(),  # screen_rating
                    gr.update(),  # screen_thanks
                    gr.update(),  # video_player
                    gr.update(),  # progress_md
                    gr.update(),  # overall_btn
                    gr.update(),  # done_md
                    result.get("state") or {"status": "welcome"},  # state
                )
            new_state = result["state"]
            if new_state["status"] == "all_done":
                # Rater has already finished — show done message, hide overall form.
                return (
                    gr.update(visible=False),                # welcome_error
                    gr.update(visible=False),                # screen_welcome
                    gr.update(visible=False),                # screen_rating
                    gr.update(visible=True),                 # screen_thanks
                    gr.update(),                             # video_player
                    gr.update(),                             # progress_md
                    gr.update(interactive=False),            # overall_btn
                    gr.update(visible=True),                 # done_md
                    new_state,
                )
            if new_state["status"] == "overall_pending":
                return (
                    gr.update(visible=False),  # welcome_error
                    gr.update(visible=False),  # screen_welcome
                    gr.update(visible=False),  # screen_rating
                    gr.update(visible=True),   # screen_thanks
                    gr.update(),               # video_player
                    gr.update(),               # progress_md
                    gr.update(),               # overall_btn
                    gr.update(),               # done_md
                    new_state,
                )
            # status == "rating"
            idx = new_state["current_index"]
            entry = manifest[idx]
            video_path = str(_STUDY_VIDEOS_DIR / entry["filename"])
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=video_path),
                gr.update(value=f"Video {idx + 1} of {new_state['total']}"),
                gr.update(),
                gr.update(),
                new_state,
            )

        start_btn.click(
            _start_click,
            [rater_id_tb, consent_chk],
            [welcome_error, screen_welcome, screen_rating, screen_thanks,
             video_player, progress_md, overall_btn, done_md, state],
        )

        # ---- Submit click ----
        def _submit_click(state_val, comment_val, *rating_values):
            ratings = {k: v for (k, _, _), v in zip(DIMENSIONS, rating_values)}
            result = on_submit(state_val, ratings, comment_val, manifest)
            if not result["ok"]:
                return (
                    gr.update(visible=True, value=f"**{result['error']}**"),  # rating_error
                    gr.update(),  # screen_rating
                    gr.update(),  # screen_thanks
                    gr.update(),  # video_player
                    gr.update(),  # progress_md
                    gr.update(),  # comment_tb
                    *[gr.update() for _ in DIMENSIONS],
                    gr.update(interactive=True),  # submit_btn — keep enabled so rater can retry
                    result["state"],
                )
            new_state = result["state"]
            if new_state["status"] == "overall_pending":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(),
                    gr.update(value=""),
                    *[gr.update(value=None) for _ in DIMENSIONS],
                    gr.update(interactive=False),
                    new_state,
                )
            # next video
            idx = new_state["current_index"]
            entry = manifest[idx]
            video_path = str(_STUDY_VIDEOS_DIR / entry["filename"])
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=video_path),
                gr.update(value=f"Video {idx + 1} of {new_state['total']}"),
                gr.update(value=""),
                *[gr.update(value=None) for _ in DIMENSIONS],
                gr.update(interactive=False),
                new_state,
            )

        submit_btn.click(
            _submit_click,
            [state, comment_tb, *radios],
            [rating_error, screen_rating, screen_thanks, video_player, progress_md,
             comment_tb, *radios, submit_btn, state],
        )

        # ---- Overall submit click ----
        def _overall_click(state_val, overall_val):
            result = on_overall_submit(state_val, overall_val)
            if not result["ok"]:
                return (
                    gr.update(visible=True, value=f"**{result['error']}**"),  # overall_error
                    gr.update(),  # done_md
                    gr.update(),  # overall_btn
                    result["state"],
                )
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(interactive=False),
                result["state"],
            )

        overall_btn.click(
            _overall_click,
            [state, overall_tb],
            [overall_error, done_md, overall_btn, state],
        )

    return tab_root
