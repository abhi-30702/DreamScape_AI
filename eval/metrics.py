import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import jiwer

from app.cache import Cache
from app.config import CACHE_DIR
from app.models.schemas import (
    PipelineRun,
    SubtitleOutput,
    VideoOutput,
    VisualOutput,
)


def compute_wer(hypotheses: list[str], references: list[str]) -> dict:
    per_scene = [jiwer.wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    mean = sum(per_scene) / len(per_scene) if per_scene else 0.0
    return {"per_scene": per_scene, "mean": round(mean, 4)}


def compute_clip_score(image_paths: list[str], texts: list[str]) -> dict:
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    scores = []
    with torch.no_grad():
        for path, text in zip(image_paths, texts):
            image = preprocess(Image.open(path)).unsqueeze(0)
            tokens = tokenizer([text])
            image_feats = model.encode_image(image)
            text_feats = model.encode_text(tokens)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            score = float((image_feats @ text_feats.T).squeeze())
            scores.append(score)

    mean = sum(scores) / len(scores) if scores else 0.0
    return {"per_scene": scores, "mean": round(mean, 4)}


def compute_sync_error(video_path: str, srt_entries: list) -> dict:
    import os
    import tempfile
    import whisper
    from moviepy.editor import VideoFileClip

    tmp_audio = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio = tmp.name

        with VideoFileClip(video_path) as clip:
            clip.audio.write_audiofile(tmp_audio, logger=None)

        model = whisper.load_model("base")
        result = whisper.transcribe(model, tmp_audio, word_timestamps=True)

        words = [
            w
            for seg in result.get("segments", [])
            for w in seg.get("words", [])
        ]

        errors_ms = []
        for i, entry in enumerate(srt_entries):
            if i >= len(words):
                break
            error_ms = abs(words[i]["start"] - entry.start_s) * 1000
            errors_ms.append(error_ms)

        if not errors_ms:
            return {"per_entry": [], "mean": 0.0, "max": 0.0, "pass": True}

        mean_ms = sum(errors_ms) / len(errors_ms)
        max_ms = max(errors_ms)
        return {
            "per_entry": [round(e, 1) for e in errors_ms],
            "mean": round(mean_ms, 1),
            "max": round(max_ms, 1),
            "pass": max_ms < 200.0,
        }
    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.unlink(tmp_audio)


def _load_run(cache: Cache, run_id: str) -> PipelineRun:
    params = cache.load_run_params(run_id)
    data: dict = {
        "run_id": run_id,
        "prompt_hash": "",
        "prompt": params["prompt"],
        "duration_target_s": int(params.get("duration", 60)),
        "style": params.get("style", "cinematic"),
        "voice": params.get("voice", "female"),
        "status": "complete",
    }
    stage_fields = {
        1: "parsed_prompt",
        2: "scene_plan",
        3: "visual_output",
        4: "narration_output",
        5: "subtitle_output",
        7: "video_output",
    }
    for stage_num, field in stage_fields.items():
        try:
            data[field] = cache.load_stage_output(run_id, stage_num)
        except KeyError:
            pass
    return PipelineRun.model_validate(data)


def _list_complete_run_ids(cache: Cache) -> list[str]:
    with sqlite3.connect(cache.db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM stage_outputs WHERE stage_num=7"
        ).fetchall()
    return [row[0] for row in rows]


def evaluate_run(run_id: str, out_path: Path, cache: Cache) -> dict:
    run = _load_run(cache, run_id)
    result: dict = {
        "run_id": run_id,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "prompt": run.prompt,
        "duration_target_s": run.duration_target_s,
        "clip_score": None,
        "wer": None,
        "sync_error_ms": None,
    }

    if run.visual_output and run.scene_plan:
        image_paths = [img.path for img in run.visual_output.images]
        texts = [s.narration_text for s in run.scene_plan.scenes]
        result["clip_score"] = compute_clip_score(image_paths, texts)

    if run.subtitle_output and run.narration_output and run.scene_plan:
        audio_durations = [a.duration_s for a in run.narration_output.audio]
        scene_starts: list[float] = []
        t = 0.0
        for d in audio_durations:
            scene_starts.append(t)
            t += d

        hyp_per_scene = []
        for i, scene in enumerate(run.scene_plan.scenes):
            t_start = scene_starts[i] if i < len(scene_starts) else 0.0
            t_end = t_start + (audio_durations[i] if i < len(audio_durations) else 0.0)
            entries = [
                e.text
                for e in run.subtitle_output.entries
                if t_start <= e.start_s < t_end
            ]
            hyp_per_scene.append(" ".join(entries))

        refs = [s.narration_text for s in run.scene_plan.scenes]
        result["wer"] = compute_wer(hyp_per_scene, refs)

    if run.video_output and run.subtitle_output:
        result["sync_error_ms"] = compute_sync_error(
            run.video_output.path,
            run.subtitle_output.entries,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute automated metrics for a DreamScapeAI pipeline run"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Evaluate a single run by ID")
    group.add_argument("--batch", action="store_true", help="Evaluate all complete runs in cache")
    parser.add_argument("--out", type=Path, help="Output JSON path (single run only)")
    args = parser.parse_args()

    cache = Cache(db_path=CACHE_DIR / "runs.db", asset_dir=CACHE_DIR)

    if args.run_id:
        out = args.out or Path("eval_results") / f"{args.run_id}.json"
        result = evaluate_run(args.run_id, out, cache)
        print(json.dumps(result, indent=2))
    else:
        run_ids = _list_complete_run_ids(cache)
        print(f"Found {len(run_ids)} complete run(s)")
        for rid in run_ids:
            out = Path("eval_results") / f"{rid}.json"
            evaluate_run(rid, out, cache)
            print(f"  {rid} → {out}")


if __name__ == "__main__":
    main()
