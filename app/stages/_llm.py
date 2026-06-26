import os
from app.stages._ollama import ollama_generate
from app.stages._hf_inference import hf_generate


def llm_generate(base_url: str, model: str, prompt: str, timeout: float = 120.0) -> dict:
    backend = os.getenv("DREAMSCAPE_LLM_BACKEND", "ollama")
    if backend == "ollama":
        return ollama_generate(base_url, model, prompt, timeout)
    if backend == "hf":
        return hf_generate(model, prompt, timeout)
    raise ValueError(f"Unknown LLM backend: {backend!r}. Use 'ollama' or 'hf'.")
