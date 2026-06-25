import json
import pytest
import httpx
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.stages.stage1_parse import Stage1Parse


def _mock_ollama_post(response_dict: dict):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": json.dumps(response_dict)}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def test_stage1_real_returns_valid_schema(tmp_path):
    fake = {
        "sentiment": "sad",
        "duration_target_s": 30,
        "style": "cinematic",
        "key_entities": ["wolf", "moon", "mountain"],
        "prompt": "A wolf howls at the moon",
    }
    with patch("app.stages._ollama.httpx.post", return_value=_mock_ollama_post(fake)):
        stage = Stage1Parse(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "prompt": "A wolf howls at the moon",
            "duration": 30,
            "style": "cinematic",
            "voice": "female",
        })
    assert result["sentiment"] == "sad"
    assert result["duration_target_s"] == 30
    assert "wolf" in result["key_entities"]
    assert result["prompt"] == "A wolf howls at the moon"


def test_stage1_real_coerces_invalid_sentiment(tmp_path):
    fake = {
        "sentiment": "angry",
        "duration_target_s": 60,
        "style": "cinematic",
        "key_entities": ["cat"],
        "prompt": "A cat sits",
    }
    with patch("app.stages._ollama.httpx.post", return_value=_mock_ollama_post(fake)):
        stage = Stage1Parse(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"prompt": "A cat sits", "duration": 60, "style": "cinematic", "voice": "female"})
    assert result["sentiment"] == "neutral"


def test_stage1_real_raises_on_ollama_connection_failure(tmp_path):
    with patch("app.stages._ollama.httpx.post", side_effect=httpx.ConnectError("refused")):
        stage = Stage1Parse(cache_dir=tmp_path, stub_stages=set())
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            stage.run({"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})


def test_stage1_real_fills_entities_when_llm_returns_empty(tmp_path):
    fake = {
        "sentiment": "neutral",
        "duration_target_s": 60,
        "style": "cinematic",
        "key_entities": [],
        "prompt": "A wolf howls at the moon",
    }
    with patch("app.stages._ollama.httpx.post", return_value=_mock_ollama_post(fake)):
        stage = Stage1Parse(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"prompt": "A wolf howls at the moon", "duration": 60, "style": "cinematic", "voice": "female"})
    assert len(result["key_entities"]) >= 1
    assert all(isinstance(e, str) for e in result["key_entities"])
