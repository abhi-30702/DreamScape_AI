# HF Spaces Deployment Design

## Goal

Deploy DreamScapeAI as a public Hugging Face Space (Python SDK / Gradio runtime) with all seven pipeline stages running on real models, using the HF Inference API for Llama 3.1-8B (Stages 1–2) instead of a local Ollama server.

## Context

The pipeline is already implemented end-to-end. Stages 3–6 load GPU models as lazy module-level singletons (SDXL, XTTS-v2, Whisper, MusicGen). Stages 1–2 call `ollama_generate()` in `app/stages/_ollama.py` via `httpx`. The Gradio UI lives in `ui/gradio_app.py`. HF Spaces requires an `app.py` at the repo root that exposes a `demo` object.

---

## Architecture

The only structural change is a thin LLM backend dispatcher inserted between Stages 1–2 and the Ollama client. Everything else is unchanged.

```
Stage1 / Stage2
    │
    ▼
_llm.py  ──  DREAMSCAPE_LLM_BACKEND=ollama  ──▶  _ollama.py  (local dev)
         ──  DREAMSCAPE_LLM_BACKEND=hf      ──▶  _hf_inference.py  (HF Spaces)
```

---

## Files

### New files

**`app.py`** (repo root — HF Spaces entry point)
```python
from ui.gradio_app import build_ui
demo = build_ui()
if __name__ == "__main__":
    demo.launch()
```
HF Spaces auto-detects the `demo` object when `app_file: app.py` is set in the Space metadata.

---

**`app/stages/_hf_inference.py`**

Wraps `huggingface_hub.InferenceClient`. Public function signature matches `ollama_generate` exactly so `_llm.py` can call either interchangeably:

```python
def hf_generate(model: str, prompt: str, timeout: float = 120.0) -> dict:
```

- Reads `HF_TOKEN` from env at call time (not module load) so tests can patch it.
- Calls `InferenceClient(model=model, token=token).text_generation(prompt, max_new_tokens=1024, return_full_text=False)`.
- Parses JSON from the response string with the same guards as `_ollama.py`: raises `RuntimeError` on missing/invalid JSON, auth failure (401), or rate limit (429).

---

**`app/stages/_llm.py`**

Single public function:

```python
def llm_generate(base_url: str, model: str, prompt: str, timeout: float = 120.0) -> dict:
```

- Reads `DREAMSCAPE_LLM_BACKEND` (default `"ollama"`).
- Calls `ollama_generate(base_url, model, prompt, timeout)` or `hf_generate(model, prompt, timeout)`.
- `base_url` is ignored on the HF path (passed through but unused).
- Raises `ValueError` if backend is not `"ollama"` or `"hf"`.

---

**`packages.txt`** (repo root)
```
ffmpeg
```
HF Spaces installs these apt packages before building the Python environment.

---

### Modified files

**`app/stages/stage1_parse.py`** and **`app/stages/stage2_expand.py`**

Replace:
```python
from app.stages._ollama import ollama_generate
...
result = ollama_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text, timeout=180.0)
```
With:
```python
from app.stages._llm import llm_generate
...
result = llm_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text, timeout=180.0)
```
No other changes.

---

**`app/config.py`**

Add one line:
```python
LLM_BACKEND = os.getenv("DREAMSCAPE_LLM_BACKEND", "ollama")
```

---

**`.env.example`**

Add:
```
DREAMSCAPE_LLM_BACKEND=ollama
```

---

**`README.md`**

Prepend HF Spaces YAML metadata:
```yaml
---
title: DreamScapeAI
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.30.0"
app_file: app.py
pinned: false
---
```

---

## HF Space Secrets

Set these in the Space settings → Variables and secrets:

| Secret | Value |
|---|---|
| `HF_TOKEN` | HF access token with Meta Llama 3.1 license accepted |
| `DREAMSCAPE_LLM_BACKEND` | `hf` |
| `DREAMSCAPE_OLLAMA_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `DREAMSCAPE_STUB_STAGES` | _(empty — run all real)_ |

---

## Error Handling

- **Auth failure (401)**: `hf_generate` raises `RuntimeError("HF Inference API auth failed — is HF_TOKEN set and does it have Llama access?")`.
- **Rate limit (429)**: `RuntimeError("HF Inference API rate limit hit")`.
- **Non-JSON response**: `RuntimeError(f"HF returned non-JSON: {raw[:200]}")`.
- **Unknown backend**: `_llm.py` raises `ValueError(f"Unknown LLM backend: {backend!r}. Use 'ollama' or 'hf'.")`.

---

## Testing

No new integration tests required. Unit tests for `_hf_inference.py` and `_llm.py` mock `huggingface_hub.InferenceClient` and confirm:

- `llm_generate` routes to `ollama_generate` when `DREAMSCAPE_LLM_BACKEND=ollama`.
- `llm_generate` routes to `hf_generate` when `DREAMSCAPE_LLM_BACKEND=hf`.
- `hf_generate` raises `RuntimeError` on 401, 429, and non-JSON.
- Stage 1 and Stage 2 call `llm_generate` (not `ollama_generate` directly).

---

## What Is Not Changing

- `_ollama.py` — untouched.
- `ui/gradio_app.py` — untouched.
- Stages 3–7 — untouched.
- `orchestrator.py`, `cache.py`, `filters.py` — untouched.
- All existing tests continue to pass unchanged.
