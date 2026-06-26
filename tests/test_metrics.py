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
