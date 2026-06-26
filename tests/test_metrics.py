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
