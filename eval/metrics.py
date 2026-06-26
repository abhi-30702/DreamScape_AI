import jiwer


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
