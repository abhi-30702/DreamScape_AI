import pytest
from unittest.mock import patch

from app.stages._llm import llm_generate


def test_llm_generate_routes_to_ollama(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "ollama")
    with patch("app.stages._llm.ollama_generate", return_value={"k": "v"}) as mock_og:
        result = llm_generate("http://localhost:11434", "llama3.1:8b", "prompt")
    mock_og.assert_called_once_with("http://localhost:11434", "llama3.1:8b", "prompt", 120.0)
    assert result == {"k": "v"}


def test_llm_generate_routes_to_hf(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "hf")
    with patch("app.stages._llm.hf_generate", return_value={"k": "v"}) as mock_hf:
        result = llm_generate(
            "http://localhost:11434",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "prompt",
        )
    mock_hf.assert_called_once_with(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", "prompt", 120.0
    )
    assert result == {"k": "v"}


def test_llm_generate_default_backend_is_ollama(monkeypatch):
    monkeypatch.delenv("DREAMSCAPE_LLM_BACKEND", raising=False)
    with patch("app.stages._llm.ollama_generate", return_value={}) as mock_og:
        llm_generate("http://localhost:11434", "llama3.1:8b", "prompt")
    mock_og.assert_called_once_with("http://localhost:11434", "llama3.1:8b", "prompt", 120.0)


def test_llm_generate_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "groq")
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        llm_generate("http://localhost:11434", "llama3.1:8b", "prompt")
