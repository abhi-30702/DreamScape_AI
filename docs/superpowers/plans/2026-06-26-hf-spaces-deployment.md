# HF Spaces Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy DreamScapeAI as a public Hugging Face Space (Python SDK / Gradio runtime) with all seven pipeline stages running on real models, routing LLM calls for Stages 1–2 through the HF Inference API instead of a local Ollama server.

**Architecture:** A thin dispatcher `_llm.py` sits between Stages 1–2 and the network clients. An env var `DREAMSCAPE_LLM_BACKEND` selects between the existing `_ollama.py` (local dev) and a new `_hf_inference.py` (HF Spaces). Everything else — stages 3–7, orchestrator, Gradio UI — is untouched.

**Tech Stack:** `huggingface_hub.InferenceClient` (already installed transitively via `transformers`), `gradio` Python SDK runtime on HF Spaces, `packages.txt` for apt deps.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `app/stages/_hf_inference.py` | Create | HF Inference API client — wraps `InferenceClient.text_generation`, same return type as `ollama_generate` |
| `app/stages/_llm.py` | Create | Dispatcher — reads `DREAMSCAPE_LLM_BACKEND`, routes to `ollama_generate` or `hf_generate` |
| `app/stages/stage1_parse.py` | Modify line 3 + 34 | Swap `ollama_generate` import/call → `llm_generate` |
| `app/stages/stage2_expand.py` | Modify line 3 + 64 | Swap `ollama_generate` import/call → `llm_generate` |
| `app/config.py` | Modify | Add `LLM_BACKEND` env var |
| `.env.example` | Modify | Document `DREAMSCAPE_LLM_BACKEND` |
| `app.py` | Create (root) | HF Spaces entry point — exports `demo` object |
| `packages.txt` | Create (root) | Apt packages: `ffmpeg` |
| `README.md` | Modify | Prepend HF Spaces YAML metadata |
| `tests/test_hf_inference.py` | Create | Unit tests for `_hf_inference.py` |
| `tests/test_llm.py` | Create | Unit tests for `_llm.py` dispatcher |

---

## Task 1: HF Inference API client (`_hf_inference.py`)

**Files:**
- Create: `app/stages/_hf_inference.py`
- Create: `tests/test_hf_inference.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hf_inference.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_hf_inference.py -v
```

Expected: 4 failures with `ModuleNotFoundError` or `ImportError` (file doesn't exist yet).

- [ ] **Step 3: Implement `_hf_inference.py`**

Create `app/stages/_hf_inference.py`:

```python
import json
import os


def hf_generate(model: str, prompt: str, timeout: float = 120.0) -> dict:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN env var is not set — add it to .env or Space secrets"
        )
    from huggingface_hub import InferenceClient
    try:
        client = InferenceClient(model=model, token=token, timeout=timeout)
        raw = client.text_generation(prompt, max_new_tokens=1024, return_full_text=False)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "unauthorized" in msg.lower():
            raise RuntimeError(
                "HF Inference API auth failed — is HF_TOKEN set and does it have Llama access?"
            ) from e
        if "429" in msg or "rate limit" in msg.lower():
            raise RuntimeError("HF Inference API rate limit hit") from e
        raise RuntimeError(f"HF Inference API error: {e}") from e
    if not raw:
        raise RuntimeError("HF Inference API returned empty response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"HF returned non-JSON: {raw[:200]}") from e
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_hf_inference.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add app/stages/_hf_inference.py tests/test_hf_inference.py
git commit -m "feat: add HF Inference API client for LLM stages"
```

---

## Task 2: LLM dispatcher (`_llm.py`)

**Files:**
- Create: `app/stages/_llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm.py`:

```python
import pytest
from unittest.mock import patch


def test_llm_generate_routes_to_ollama(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "ollama")
    with patch("app.stages._llm.ollama_generate", return_value={"k": "v"}) as mock_og:
        from app.stages._llm import llm_generate
        result = llm_generate("http://localhost:11434", "llama3.1:8b", "prompt")
    mock_og.assert_called_once_with("http://localhost:11434", "llama3.1:8b", "prompt", 120.0)
    assert result == {"k": "v"}


def test_llm_generate_routes_to_hf(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "hf")
    with patch("app.stages._llm.hf_generate", return_value={"k": "v"}) as mock_hf:
        from app.stages._llm import llm_generate
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
        from app.stages._llm import llm_generate
        llm_generate("http://localhost:11434", "llama3.1:8b", "prompt")
    mock_og.assert_called_once()


def test_llm_generate_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "groq")
    from app.stages._llm import llm_generate
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        llm_generate("http://localhost:11434", "llama3.1:8b", "prompt")
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_llm.py -v
```

Expected: 4 failures with `ModuleNotFoundError` (file doesn't exist yet).

- [ ] **Step 3: Implement `_llm.py`**

Create `app/stages/_llm.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_llm.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add app/stages/_llm.py tests/test_llm.py
git commit -m "feat: add LLM backend dispatcher routing ollama/hf via env var"
```

---

## Task 3: Wire stages 1 and 2 through the dispatcher

**Files:**
- Modify: `app/stages/stage1_parse.py` (lines 3, 34)
- Modify: `app/stages/stage2_expand.py` (lines 3, 64)

Context: `stage1_parse.py` line 3 has `from app.stages._ollama import ollama_generate` and line 34 calls `ollama_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text)`. Same pattern in `stage2_expand.py` line 3 and line 64.

- [ ] **Step 1: Update `stage1_parse.py`**

In `app/stages/stage1_parse.py`, change line 3:
```python
# Before
from app.stages._ollama import ollama_generate

# After
from app.stages._llm import llm_generate
```

Change line 34:
```python
# Before
        result = ollama_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text)

# After
        result = llm_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text)
```

- [ ] **Step 2: Update `stage2_expand.py`**

In `app/stages/stage2_expand.py`, change line 3:
```python
# Before
from app.stages._ollama import ollama_generate

# After
from app.stages._llm import llm_generate
```

Change line 64:
```python
# Before
        result = ollama_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text, timeout=180.0)

# After
        result = llm_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text, timeout=180.0)
```

- [ ] **Step 3: Verify existing stage 1 and 2 tests still pass**

The existing tests mock `app.stages._ollama.httpx.post`. The call chain is now: Stage1 → `llm_generate` → `ollama_generate` → `httpx.post`. The mock target hasn't changed, so all existing tests should pass unchanged.

```
pytest tests/test_stage1_real.py tests/test_stage2_real.py -v
```

Expected: all pass.

- [ ] **Step 4: Add HF-path smoke tests for stages 1 and 2**

Add these tests to `tests/test_stage1_real.py` (append to the bottom of the file):

```python
import json as _json
from unittest.mock import patch as _patch
from app.stages.stage1_parse import Stage1Parse as _Stage1Parse


def test_stage1_real_routes_via_hf_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "hf")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    response = {
        "sentiment": "happy",
        "key_entities": ["warrior"],
        "style": "cinematic",
        "prompt": "A warrior stands tall",
        "duration_target_s": 60,
    }
    with _patch("huggingface_hub.InferenceClient") as mock_cls:
        mock_cls.return_value.text_generation.return_value = _json.dumps(response)
        stage = _Stage1Parse(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "prompt": "A warrior stands tall",
            "duration": 60,
            "style": "cinematic",
            "voice": "female",
        })
    assert result["parsed_prompt"]["sentiment"] == "happy"
    assert mock_cls.return_value.text_generation.called
```

Add to `tests/test_stage2_real.py` (append to the bottom):

```python
import json as _json
from unittest.mock import patch as _patch
from app.stages.stage2_expand import Stage2Expand as _Stage2Expand


def test_stage2_real_routes_via_hf_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("DREAMSCAPE_LLM_BACKEND", "hf")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    scenes = [
        {"id": i, "description": f"desc {i}", "narration_text": f"narr {i}",
         "mood": "neutral", "duration_estimate_s": 15.0}
        for i in range(4)
    ]
    with _patch("huggingface_hub.InferenceClient") as mock_cls:
        mock_cls.return_value.text_generation.return_value = _json.dumps({"scenes": scenes})
        stage = _Stage2Expand(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({"parsed_prompt": {
            "sentiment": "neutral",
            "style": "cinematic",
            "key_entities": ["knight"],
            "prompt": "A knight rides",
            "duration_target_s": 60,
        }})
    assert len(result["scenes"]) == 4
    assert mock_cls.return_value.text_generation.called
```

- [ ] **Step 5: Run new tests**

```
pytest tests/test_stage1_real.py tests/test_stage2_real.py -v
```

Expected: all pass (including the 2 new HF-path tests).

- [ ] **Step 6: Commit**

```
git add app/stages/stage1_parse.py app/stages/stage2_expand.py tests/test_stage1_real.py tests/test_stage2_real.py
git commit -m "feat: wire stages 1-2 through LLM dispatcher; add HF-path smoke tests"
```

---

## Task 4: Config and env example

**Files:**
- Modify: `app/config.py` (after line 13)
- Modify: `.env.example` (append)

- [ ] **Step 1: Add `LLM_BACKEND` to `app/config.py`**

After the line `OLLAMA_MODEL = os.getenv("DREAMSCAPE_OLLAMA_MODEL", "llama3.1:8b")`, add:

```python
LLM_BACKEND = os.getenv("DREAMSCAPE_LLM_BACKEND", "ollama")
```

The full updated block (lines 12–14) should read:
```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("DREAMSCAPE_OLLAMA_MODEL", "llama3.1:8b")
LLM_BACKEND = os.getenv("DREAMSCAPE_LLM_BACKEND", "ollama")
```

- [ ] **Step 2: Add env var to `.env.example`**

Append to `.env.example` (after `DREAMSCAPE_OLLAMA_MODEL=llama3.1:8b`):

```
DREAMSCAPE_LLM_BACKEND=ollama
```

The updated file should be:
```
DREAMSCAPE_STUB_STAGES=3,4,6
DREAMSCAPE_CACHE_DIR=cache
DREAMSCAPE_CACHE_TTL_HOURS=24
OLLAMA_BASE_URL=http://localhost:11434
DREAMSCAPE_OLLAMA_MODEL=llama3.1:8b
DREAMSCAPE_LLM_BACKEND=ollama
DREAMSCAPE_DEFAULT_VOICE=female
DREAMSCAPE_DEFAULT_DURATION=60
DREAMSCAPE_WHISPER_MODEL=base
DREAMSCAPE_MUSICGEN_MODEL=facebook/musicgen-medium
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```
pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```
git add app/config.py .env.example
git commit -m "feat: add DREAMSCAPE_LLM_BACKEND env var to config and env example"
```

---

## Task 5: HF Spaces entry point (`app.py` + `packages.txt`)

**Files:**
- Create: `app.py` (repo root)
- Create: `packages.txt` (repo root)

- [ ] **Step 1: Create `app.py`**

Create `app.py` at the repo root (same level as `requirements.txt`):

```python
from ui.gradio_app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
```

HF Spaces scans for a module-level `demo` variable of type `gr.Blocks`. `build_ui()` is called at import time here, which constructs the Gradio layout and the Orchestrator (no models are loaded at this point — they load lazily on first pipeline run).

- [ ] **Step 2: Create `packages.txt`**

Create `packages.txt` at the repo root:

```
ffmpeg
```

HF Spaces runs `apt-get install` on each line of this file before setting up the Python environment. FFmpeg is required by MoviePy (Stage 7) for video assembly.

- [ ] **Step 3: Verify `app.py` can be imported**

```
python -c "import app; import gradio as gr; assert isinstance(app.demo, gr.Blocks); print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```
git add app.py packages.txt
git commit -m "feat: add HF Spaces entry point (app.py) and packages.txt with ffmpeg"
```

---

## Task 6: README.md HF Spaces metadata

**Files:**
- Modify: `README.md` (prepend YAML header)

HF Spaces reads the YAML frontmatter from a repo's `README.md` to configure the Space. It must appear at the very top of the file, before any other content.

- [ ] **Step 1: Prepend the YAML header to `README.md`**

The current `README.md` starts with `# DreamScapeAI`. Replace the start of the file so it begins with the YAML block:

```
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

# DreamScapeAI
```

The rest of the file is unchanged.

- [ ] **Step 2: Verify the YAML is valid**

```
python -c "
import re, sys
text = open('README.md').read()
assert text.startswith('---'), 'Missing YAML front matter'
end = text.index('---', 3)
block = text[3:end]
assert 'sdk: gradio' in block
assert 'app_file: app.py' in block
print('README YAML looks good')
"
```

Expected output: `README YAML looks good`

- [ ] **Step 3: Run full test suite one final time**

```
pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit and push**

```
git add README.md
git commit -m "docs: add HF Spaces YAML metadata to README"
git push origin main
```

---

## HF Space Setup (manual, post-implementation)

After pushing, configure the Space on huggingface.co:

1. Create a new Space → choose **Gradio** SDK → link to the `abhi-30702/DreamScape_AI` GitHub repo (or push directly).
2. In Space **Settings → Variables and secrets**, add:

| Name | Value |
|---|---|
| `HF_TOKEN` | Your HF access token (must have Meta Llama 3.1 license accepted at hf.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| `DREAMSCAPE_LLM_BACKEND` | `hf` |
| `DREAMSCAPE_OLLAMA_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `DREAMSCAPE_STUB_STAGES` | _(leave empty — run all real stages)_ |

3. Set hardware to **T4 Small** (free GPU tier) under Space hardware settings.
4. The Space will build and deploy automatically.
