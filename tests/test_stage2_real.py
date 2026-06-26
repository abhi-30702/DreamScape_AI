import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.stages.stage2_expand import Stage2Expand


def _mock_ollama_post(response_dict: dict):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": json.dumps(response_dict)}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _parsed_prompt(sentiment="neutral", duration=60):
    return {
        "prompt": "A wolf howls at the moon",
        "sentiment": sentiment,
        "duration_target_s": duration,
        "style": "cinematic",
        "key_entities": ["wolf", "moon"],
    }


def test_stage2_real_returns_scenes(tmp_path):
    fake_scenes = [
        {
            "id": i,
            "description": f"Wide shot of scene {i} with dramatic lighting and atmospheric fog",
            "narration_text": f"The story unfolds in scene {i} of the night.",
            "mood": "neutral",
            "duration_estimate_s": 15.0,
        }
        for i in range(4)
    ]
    with patch("app.stages._ollama.httpx.post", return_value=_mock_ollama_post({"scenes": fake_scenes})):
        stage = Stage2Expand(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"parsed_prompt": _parsed_prompt()})
    assert len(result["scenes"]) == 4
    for s in result["scenes"]:
        assert "description" in s
        assert "narration_text" in s
        assert "mood" in s
        assert isinstance(s["duration_estimate_s"], float)


def test_stage2_real_falls_back_to_stub_on_empty_scenes(tmp_path):
    with patch("app.stages._ollama.httpx.post", return_value=_mock_ollama_post({"scenes": []})):
        stage = Stage2Expand(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"parsed_prompt": _parsed_prompt()})
    assert len(result["scenes"]) == 4


def test_stage2_real_coerces_invalid_mood_to_parsed_sentiment(tmp_path):
    fake_scenes = [
        {
            "id": i,
            "description": "A dramatic scene unfolds in shadow",
            "narration_text": "The darkness spreads.",
            "mood": "angry",
            "duration_estimate_s": 15.0,
        }
        for i in range(4)
    ]
    with patch("app.stages._ollama.httpx.post", return_value=_mock_ollama_post({"scenes": fake_scenes})):
        stage = Stage2Expand(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"parsed_prompt": _parsed_prompt(sentiment="sad")})
    for s in result["scenes"]:
        assert s["mood"] == "sad"


def test_stage2_real_caps_at_8_scenes(tmp_path):
    fake_scenes = [
        {"id": i, "description": f"Scene {i}", "narration_text": "Narration.", "mood": "neutral", "duration_estimate_s": 10.0}
        for i in range(12)  # LLM returns 12, should be capped at 8
    ]
    with patch("app.stages._ollama.httpx.post", return_value=_mock_ollama_post({"scenes": fake_scenes})):
        stage = Stage2Expand(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"parsed_prompt": _parsed_prompt()})
    assert len(result["scenes"]) == 8


def test_stage2_real_routes_via_hf_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "hf")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    scenes = [
        {"id": i, "description": f"desc {i}", "narration_text": f"narr {i}",
         "mood": "neutral", "duration_estimate_s": 15.0}
        for i in range(4)
    ]
    # InferenceClient is imported lazily inside hf_generate(), so patch at the source module
    with patch("huggingface_hub.InferenceClient") as mock_cls:
        mock_cls.return_value.text_generation.return_value = json.dumps({"scenes": scenes})
        stage = Stage2Expand(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"parsed_prompt": {
            "sentiment": "neutral",
            "style": "cinematic",
            "key_entities": ["knight"],
            "prompt": "A knight rides",
            "duration_target_s": 60,
        }})
    assert len(result["scenes"]) == 4
    mock_cls.return_value.text_generation.assert_called_once()
