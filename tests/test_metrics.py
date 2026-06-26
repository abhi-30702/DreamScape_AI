import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_model_caches():
    import eval.metrics
    eval.metrics._clip_model = None
    eval.metrics._clip_preprocess = None
    eval.metrics._clip_tokenizer = None
    eval.metrics._whisper_model = None
    yield


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
