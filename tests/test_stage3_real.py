import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import app.stages.stage3_visual as _s3_mod
from app.stages.stage3_visual import Stage3Visual


def _make_mock_pipe(img_color=(128, 64, 192)):
    """Returns a mock diffusers pipeline that produces a colored 1024x576 image."""
    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [Image.new("RGB", (1024, 576), img_color)]
    return mock_pipe


def test_stage3_real_creates_one_image_per_scene(tmp_path):
    _s3_mod._PIPE = None
    mock_pipe = _make_mock_pipe()

    with patch("app.stages.stage3_visual._get_pipe", return_value=mock_pipe):
        stage = Stage3Visual(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "scenes": [
                {"id": 0, "description": "Wide establishing shot of a snow-covered mountain at dawn", "mood": "neutral"},
                {"id": 1, "description": "Close-up of a lone wolf silhouetted against moonlight", "mood": "neutral"},
            ],
            "sentiment": "neutral",
            "negative_prompt": "",
            "style": "cinematic",
            "asset_dir": str(tmp_path / "imgs"),
        })

    assert len(result["images"]) == 2
    for img in result["images"]:
        assert Path(img["path"]).exists()
        assert img["width"] == 1024
        assert img["height"] == 576


def test_stage3_real_passes_negative_prompt_to_pipe(tmp_path):
    _s3_mod._PIPE = None
    mock_pipe = _make_mock_pipe()

    with patch("app.stages.stage3_visual._get_pipe", return_value=mock_pipe):
        stage = Stage3Visual(cache_dir=tmp_path, stub_stages=set())
        stage.run({
            "scenes": [{"id": 0, "description": "A wolf in the forest", "mood": "sad"}],
            "sentiment": "sad",
            "negative_prompt": "bright, colorful, cheerful, optimistic",
            "style": "cinematic",
            "asset_dir": str(tmp_path / "imgs"),
        })

    call_kwargs = mock_pipe.call_args_list[0][1]
    assert "bright" in call_kwargs["negative_prompt"]


def test_stage3_real_retries_on_black_image(tmp_path):
    _s3_mod._PIPE = None
    # Return an all-black image (simulating safety checker) for all 3 retries
    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [Image.new("RGB", (1024, 576), (0, 0, 0))]

    with patch("app.stages.stage3_visual._get_pipe", return_value=mock_pipe):
        stage = Stage3Visual(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "scenes": [{"id": 0, "description": "A scene", "mood": "neutral"}],
            "sentiment": "neutral",
            "negative_prompt": "",
            "style": "cinematic",
            "asset_dir": str(tmp_path / "imgs"),
        })

    assert mock_pipe.call_count == 3  # 3 retry attempts
    assert len(result["images"]) == 1  # fallback placeholder still produced
    placeholder_path = Path(result["images"][0]["path"])
    assert placeholder_path.exists()
    assert placeholder_path.stat().st_size > 0


def test_stage3_real_uses_style_in_prompt(tmp_path):
    _s3_mod._PIPE = None
    mock_pipe = _make_mock_pipe()

    with patch("app.stages.stage3_visual._get_pipe", return_value=mock_pipe):
        stage = Stage3Visual(cache_dir=tmp_path, stub_stages=set())
        stage.run({
            "scenes": [{"id": 0, "description": "A dark city street at night", "mood": "sad"}],
            "sentiment": "sad",
            "negative_prompt": "",
            "style": "noir",
            "asset_dir": str(tmp_path / "imgs"),
        })

    call_kwargs = mock_pipe.call_args_list[0][1]
    assert "noir" in call_kwargs["prompt"]
