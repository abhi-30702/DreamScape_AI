import json
import pytest
from unittest.mock import patch, MagicMock


def test_hf_generate_returns_parsed_json(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    mock_client = MagicMock()
    mock_client.text_generation.return_value = json.dumps({"sentiment": "neutral"})
    with patch("huggingface_hub.InferenceClient", return_value=mock_client):
        from app.stages._hf_inference import hf_generate
        result = hf_generate("meta-llama/Meta-Llama-3.1-8B-Instruct", "classify mood")
    assert result == {"sentiment": "neutral"}
    mock_client.text_generation.assert_called_once_with(
        "classify mood", max_new_tokens=1024, return_full_text=False
    )


def test_hf_generate_raises_on_missing_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from app.stages._hf_inference import hf_generate
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        hf_generate("meta-llama/Meta-Llama-3.1-8B-Instruct", "test")


def test_hf_generate_raises_on_401(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "bad_token")
    with patch("huggingface_hub.InferenceClient") as mock_cls:
        mock_cls.return_value.text_generation.side_effect = Exception("401 Unauthorized")
        from app.stages._hf_inference import hf_generate
        with pytest.raises(RuntimeError, match="auth failed"):
            hf_generate("meta-llama/Meta-Llama-3.1-8B-Instruct", "test")


def test_hf_generate_raises_on_non_json(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    with patch("huggingface_hub.InferenceClient") as mock_cls:
        mock_cls.return_value.text_generation.return_value = "not json at all {{{}}"
        from app.stages._hf_inference import hf_generate
        with pytest.raises(RuntimeError, match="non-JSON"):
            hf_generate("meta-llama/Meta-Llama-3.1-8B-Instruct", "test")
