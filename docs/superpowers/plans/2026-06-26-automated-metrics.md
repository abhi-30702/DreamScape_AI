# Automated Metrics Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI script (`eval/metrics.py`) that loads any completed DreamScapeAI pipeline run from the SQLite cache, computes CLIPScore, WER, and audio-visual sync error, and writes a JSON result file.

**Architecture:** Single `eval/metrics.py` file with three pure metric functions (plain-type interfaces for testability) and a CLI entry point that loads pipeline runs from the existing `Cache` class. Heavy models (CLIP, Whisper) are lazy-imported inside their respective functions to keep startup fast. Tests mock all models — no downloads required.

**Tech Stack:** `jiwer>=3.0.0` (WER), `open-clip-torch>=2.20.0` (CLIPScore), `whisper` + `moviepy` already in requirements (sync error), `argparse` (CLI), `app.cache.Cache` + `app.models.schemas.PipelineRun` (data loading).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `eval/__init__.py` | Create | Empty package marker |
| `eval/metrics.py` | Create | All metric logic + CLI entry point |
| `tests/test_metrics.py` | Create | 8 unit tests, all models mocked |
| `requirements.txt` | Modify | Add `jiwer>=3.0.0`, `open-clip-torch>=2.20.0` |
| `.gitignore` | Modify | Add `eval_results/` |

---

## Context for all tasks

**How pipeline data is stored in the cache:**

`Cache.load_run_params(run_id)` returns the original pipeline input dict:
```python
{"prompt": "A warrior stands tall", "duration": 60, "style": "cinematic", "voice": "female"}
```

`Cache.load_stage_output(run_id, stage_num)` returns the stage's raw output dict:
- Stage 1 (parsed_prompt): `{"prompt": str, "sentiment": str, "duration_target_s": int, "style": str, "key_entities": list[str]}`
- Stage 2 (scene_plan): `{"scenes": [{"id": int, "description": str, "narration_text": str, "mood": str, "duration_estimate_s": float}, ...]}`
- Stage 3 (visual_output): `{"images": [{"scene_id": int, "path": str, "width": int, "height": int}, ...]}`
- Stage 4 (narration_output): `{"audio": [{"scene_id": int, "path": str, "duration_s": float}, ...], "total_duration_s": float}`
- Stage 5 (subtitle_output): `{"srt_path": str, "entries": [{"index": int, "start_s": float, "end_s": float, "text": str}, ...]}`
- Stage 7 (video_output): `{"path": str, "duration_s": float, "file_size_bytes": int}`

Raises `KeyError` if that stage hasn't been saved (incomplete run).

`PipelineRun.model_validate(data_dict)` accepts nested dicts and coerces them to nested Pydantic models automatically (Pydantic v2).

---

## Task 1: Package scaffold and dependencies

**Files:**
- Create: `eval/__init__.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create `eval/__init__.py`**

Create `eval/__init__.py` as an empty file (just a newline):

```python

```

- [ ] **Step 2: Add dependencies to `requirements.txt`**

Open `requirements.txt`. Add these two lines after the `huggingface_hub>=0.22.0` line:

```
jiwer>=3.0.0
open-clip-torch>=2.20.0
```

The updated block should read:
```
huggingface_hub>=0.22.0
jiwer>=3.0.0
open-clip-torch>=2.20.0
```

- [ ] **Step 3: Add `eval_results/` to `.gitignore`**

Open `.gitignore`. If the file does not exist, create it. Append:
```
eval_results/
```

- [ ] **Step 4: Install new dependencies**

```
pip install jiwer>=3.0.0 "open-clip-torch>=2.20.0"
```

Expected: installs without error.

- [ ] **Step 5: Run existing tests to confirm no regressions**

```
pytest tests/ -q --ignore=tests/test_metrics.py
```

Expected: 78 passed (or close — the flaky moviepy Windows test may intermittently fail).

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py requirements.txt .gitignore
git commit -m "feat: scaffold eval package and add jiwer/open-clip-torch deps"
```

---

## Task 2: WER metric

**Files:**
- Create: `eval/metrics.py` (initial — just `compute_wer`)
- Create: `tests/test_metrics.py` (first 2 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import pytest
from unittest.mock import MagicMock, patch


def test_wer_perfect_match():
    from eval.metrics import compute_wer
    result = compute_wer(["hello world"], ["hello world"])
    assert result["per_scene"] == [0.0]
    assert result["mean"] == 0.0


def test_wer_partial_match():
    # "hello earth" vs "hello world" — 1 substitution out of 2 words = 0.5
    from eval.metrics import compute_wer
    result = compute_wer(["hello earth"], ["hello world"])
    assert result["per_scene"][0] == pytest.approx(0.5)
    assert result["mean"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_metrics.py -v
```

Expected: 2 failures with `ModuleNotFoundError` (file doesn't exist yet).

- [ ] **Step 3: Create `eval/metrics.py` with `compute_wer`**

Create `eval/metrics.py`:

```python
import jiwer


def compute_wer(hypotheses: list[str], references: list[str]) -> dict:
    per_scene = [jiwer.wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    mean = sum(per_scene) / len(per_scene) if per_scene else 0.0
    return {"per_scene": per_scene, "mean": round(mean, 4)}
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_metrics.py::test_wer_perfect_match tests/test_metrics.py::test_wer_partial_match -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_metrics.py
git commit -m "feat: add compute_wer metric function"
```

---

## Task 3: CLIPScore metric

**Files:**
- Modify: `eval/metrics.py` (add `compute_clip_score`)
- Modify: `tests/test_metrics.py` (add 2 tests)

- [ ] **Step 1: Append the failing tests to `tests/test_metrics.py`**

Append to the bottom of `tests/test_metrics.py`:

```python
import torch
from PIL import Image


def test_clip_score_returns_per_scene_and_mean(tmp_path):
    img_paths = []
    for i in range(2):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (64, 64)).save(str(p))
        img_paths.append(str(p))

    feat = torch.tensor([[1.0, 0.0]])
    mock_model = MagicMock()
    mock_model.encode_image.return_value = feat
    mock_model.encode_text.return_value = feat
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
    mock_tokenizer = MagicMock(return_value=torch.zeros(1, 77, dtype=torch.long))

    with patch("open_clip.create_model_and_transforms", return_value=(mock_model, None, mock_preprocess)), \
         patch("open_clip.get_tokenizer", return_value=mock_tokenizer):
        from eval.metrics import compute_clip_score
        result = compute_clip_score(img_paths, ["a scene", "another scene"])

    assert len(result["per_scene"]) == 2
    assert all(isinstance(s, float) for s in result["per_scene"])
    assert isinstance(result["mean"], float)


def test_clip_score_single_scene(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (64, 64)).save(str(p))

    feat = torch.tensor([[1.0, 0.0]])
    mock_model = MagicMock()
    mock_model.encode_image.return_value = feat
    mock_model.encode_text.return_value = feat
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
    mock_tokenizer = MagicMock(return_value=torch.zeros(1, 77, dtype=torch.long))

    with patch("open_clip.create_model_and_transforms", return_value=(mock_model, None, mock_preprocess)), \
         patch("open_clip.get_tokenizer", return_value=mock_tokenizer):
        from eval.metrics import compute_clip_score
        result = compute_clip_score([str(p)], ["a scene"])

    assert len(result["per_scene"]) == 1
    assert result["mean"] == result["per_scene"][0]
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_metrics.py::test_clip_score_returns_per_scene_and_mean tests/test_metrics.py::test_clip_score_single_scene -v
```

Expected: 2 failures with `ImportError` (function not defined yet).

- [ ] **Step 3: Add `compute_clip_score` to `eval/metrics.py`**

Append to `eval/metrics.py`:

```python


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
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_metrics.py::test_clip_score_returns_per_scene_and_mean tests/test_metrics.py::test_clip_score_single_scene -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_metrics.py
git commit -m "feat: add compute_clip_score metric function"
```

---

## Task 4: Sync error metric

**Files:**
- Modify: `eval/metrics.py` (add `compute_sync_error`)
- Modify: `tests/test_metrics.py` (add 2 tests)

- [ ] **Step 1: Append the failing tests to `tests/test_metrics.py`**

Append to the bottom of `tests/test_metrics.py`:

```python
from app.models.schemas import SubtitleEntry


def test_sync_error_within_threshold():
    entries = [
        SubtitleEntry(index=0, start_s=0.0, end_s=1.0, text="hello"),
        SubtitleEntry(index=1, start_s=1.0, end_s=2.0, text="world"),
    ]
    # Whisper re-transcription offset by 50ms per word
    mock_whisper_result = {
        "segments": [{
            "words": [
                {"word": "hello", "start": 0.05, "end": 1.0},
                {"word": "world", "start": 1.05, "end": 2.0},
            ]
        }]
    }

    mock_clip = MagicMock()
    mock_clip.__enter__ = MagicMock(return_value=mock_clip)
    mock_clip.__exit__ = MagicMock(return_value=False)
    mock_clip.audio.write_audiofile = MagicMock()

    with patch("moviepy.editor.VideoFileClip", return_value=mock_clip), \
         patch("whisper.load_model", return_value=MagicMock()), \
         patch("whisper.transcribe", return_value=mock_whisper_result):
        from eval.metrics import compute_sync_error
        result = compute_sync_error("/fake/video.mp4", entries)

    assert result["pass"] is True
    assert result["max"] < 200.0
    assert len(result["per_entry"]) == 2
    assert result["per_entry"][0] == pytest.approx(50.0, abs=1.0)


def test_sync_error_exceeds_threshold():
    entries = [SubtitleEntry(index=0, start_s=0.0, end_s=1.0, text="hello")]
    # Whisper returns 300ms off → exceeds 200ms threshold
    mock_whisper_result = {
        "segments": [{"words": [{"word": "hello", "start": 0.3, "end": 1.0}]}]
    }

    mock_clip = MagicMock()
    mock_clip.__enter__ = MagicMock(return_value=mock_clip)
    mock_clip.__exit__ = MagicMock(return_value=False)
    mock_clip.audio.write_audiofile = MagicMock()

    with patch("moviepy.editor.VideoFileClip", return_value=mock_clip), \
         patch("whisper.load_model", return_value=MagicMock()), \
         patch("whisper.transcribe", return_value=mock_whisper_result):
        from eval.metrics import compute_sync_error
        result = compute_sync_error("/fake/video.mp4", entries)

    assert result["pass"] is False
    assert result["max"] >= 200.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_metrics.py::test_sync_error_within_threshold tests/test_metrics.py::test_sync_error_exceeds_threshold -v
```

Expected: 2 failures with `ImportError`.

- [ ] **Step 3: Add `compute_sync_error` to `eval/metrics.py`**

Append to `eval/metrics.py`:

```python


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
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_metrics.py::test_sync_error_within_threshold tests/test_metrics.py::test_sync_error_exceeds_threshold -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_metrics.py
git commit -m "feat: add compute_sync_error metric function"
```

---

## Task 5: Cache loader and CLI

**Files:**
- Modify: `eval/metrics.py` (add `_load_run`, `_list_complete_run_ids`, `evaluate_run`, `main`)
- Modify: `tests/test_metrics.py` (add 4 tests)

- [ ] **Step 1: Append the failing tests to `tests/test_metrics.py`**

Append to the bottom of `tests/test_metrics.py`:

```python
import json
import sqlite3
from pathlib import Path

from app.cache import Cache
from app.models.schemas import PipelineRun


def test_load_run_assembles_pipeline_run(tmp_path):
    from eval.metrics import _load_run

    mock_cache = MagicMock()
    mock_cache.load_run_params.return_value = {
        "prompt": "A warrior stands tall",
        "duration": 60,
        "style": "cinematic",
        "voice": "female",
    }

    def stage_side_effect(run_id, stage_num):
        data = {
            1: {
                "prompt": "A warrior stands tall",
                "sentiment": "happy",
                "duration_target_s": 60,
                "style": "cinematic",
                "key_entities": ["warrior"],
            },
            2: {
                "scenes": [{
                    "id": 0,
                    "description": "A warrior",
                    "narration_text": "A warrior stands tall",
                    "mood": "happy",
                    "duration_estimate_s": 15.0,
                }]
            },
        }
        if stage_num in data:
            return data[stage_num]
        raise KeyError(stage_num)

    mock_cache.load_stage_output.side_effect = stage_side_effect

    run = _load_run(mock_cache, "abc123")

    assert run.run_id == "abc123"
    assert run.prompt == "A warrior stands tall"
    assert run.parsed_prompt.sentiment == "happy"
    assert run.scene_plan.scenes[0].narration_text == "A warrior stands tall"
    assert run.video_output is None


def test_batch_mode_skips_incomplete_runs(tmp_path):
    from eval.metrics import evaluate_run

    mock_cache = MagicMock()
    mock_cache.load_run_params.return_value = {
        "prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"
    }
    mock_cache.load_stage_output.side_effect = KeyError("not found")

    out_path = tmp_path / "result.json"
    result = evaluate_run("run001", out_path, mock_cache)

    assert result["clip_score"] is None
    assert result["wer"] is None
    assert result["sync_error_ms"] is None
    assert out_path.exists()


def test_list_complete_run_ids(tmp_path):
    from eval.metrics import _list_complete_run_ids

    cache = Cache(db_path=tmp_path / "runs.db", asset_dir=tmp_path / "assets")
    run_id = cache.create_run("hash1", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    cache.save_stage_output(run_id, 7, {"path": "/fake.mp4", "duration_s": 60.0, "file_size_bytes": 1000})

    run_id2 = cache.create_run("hash2", {"prompt": "test2", "duration": 60, "style": "cinematic", "voice": "female"})

    ids = _list_complete_run_ids(cache)

    assert run_id in ids
    assert run_id2 not in ids


def test_json_output_written_to_eval_results(tmp_path):
    from eval.metrics import evaluate_run

    mock_cache = MagicMock()
    mock_cache.load_run_params.return_value = {
        "prompt": "a test prompt", "duration": 60, "style": "cinematic", "voice": "female"
    }
    mock_cache.load_stage_output.side_effect = KeyError("not found")

    out_path = tmp_path / "run001.json"
    evaluate_run("run001", out_path, mock_cache)

    written = json.loads(out_path.read_text())
    assert written["run_id"] == "run001"
    assert written["prompt"] == "a test prompt"
    assert "computed_at" in written
    assert "clip_score" in written
    assert "wer" in written
    assert "sync_error_ms" in written
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_metrics.py::test_load_run_assembles_pipeline_run tests/test_metrics.py::test_batch_mode_skips_incomplete_runs tests/test_metrics.py::test_list_complete_run_ids tests/test_metrics.py::test_json_output_written_to_eval_results -v
```

Expected: 4 failures with `ImportError`.

- [ ] **Step 3: Rewrite `eval/metrics.py` with the complete final content**

Replace the entire contents of `eval/metrics.py` with the following (this consolidates all imports at the top in correct order and keeps all previously-written functions intact):

```python
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
```

- [ ] **Step 4: Run the 4 new tests to confirm they pass**

```
pytest tests/test_metrics.py::test_load_run_assembles_pipeline_run tests/test_metrics.py::test_batch_mode_skips_incomplete_runs tests/test_metrics.py::test_list_complete_run_ids tests/test_metrics.py::test_json_output_written_to_eval_results -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full metrics test suite**

```
pytest tests/test_metrics.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Run the full project test suite to confirm no regressions**

```
pytest tests/ -q
```

Expected: 86 passed (78 existing + 8 new). The flaky Windows/moviepy test may intermittently fail — that is pre-existing and unrelated.

- [ ] **Step 7: Commit and push**

```bash
git add eval/metrics.py tests/test_metrics.py
git commit -m "feat: add cache loader, evaluate_run orchestrator, and CLI entry point"
git push origin main
```

---

## Usage after implementation

```bash
# Evaluate a single completed run
python eval/metrics.py --run-id abc123

# Batch evaluate all complete runs in cache
python eval/metrics.py --batch

# Custom output location
python eval/metrics.py --run-id abc123 --out results/my_run.json
```

Output JSON:
```json
{
  "run_id": "abc123",
  "computed_at": "2026-06-26T12:00:00+00:00",
  "prompt": "A warrior stands tall at dawn",
  "duration_target_s": 60,
  "clip_score": { "per_scene": [0.82, 0.79, 0.85], "mean": 0.82 },
  "wer": { "per_scene": [0.05, 0.08, 0.03], "mean": 0.053 },
  "sync_error_ms": { "per_entry": [45.0, 88.0, 120.0], "mean": 84.3, "max": 120.0, "pass": true }
}
```
