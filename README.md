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

Transform a single text prompt into a cinematic short video (30–90s MP4) with AI-generated visuals, narration, subtitles, and emotion-aware background music — running entirely on local models.

---

## Architecture

Seven sequential stages, each independently stubable for local development:

```
Prompt
  │
  ▼
Stage 1 ── Prompt Parsing      Llama-3.1-8B (Ollama)   → sentiment, entities, style
  │
  ▼
Stage 2 ── Scene Expansion     Llama-3.1-8B (Ollama)   → 4–8 scene descriptions
  │
  ▼
Stage 3 ── Image Generation    SDXL (diffusers)         → 1024×576 PNG per scene
  │
  ▼
Stage 4 ── Narration           XTTS-v2 (Coqui TTS)     → WAV per scene
  │
  ▼
Stage 5 ── Subtitles           Whisper (OpenAI)         → SRT file
  │
  ▼
Stage 6 ── Music               MusicGen-medium (Meta)   → background WAV
  │
  ▼
Stage 7 ── Video Assembly      MoviePy + FFmpeg         → MP4 with burned subtitles
```

**Key design decisions:**
- TTS-first timing — narration audio duration drives image display duration
- Emotion propagation — prompt sentiment flows into SDXL negative prompt, XTTS speaker, and MusicGen conditioning
- Sequential pipeline to fit <8GB VRAM locally (T4 16GB on deployment)
- 24-hour SQLite cache — identical prompts reuse completed runs

---

## Quick Start

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) on PATH
- [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled

```bash
# Install Ollama model
ollama pull llama3.1:8b
```

### Install

```bash
git clone https://github.com/abhi-30702/DreamScape_AI.git
cd DreamScape_AI
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env as needed (defaults work for local development)
```

### Run (stub mode — no GPU required)

GPU-heavy stages (3, 4, 6) are stubbed by default — colored placeholder images and silence WAVs replace real model output. The full pipeline runs end-to-end on CPU:

```bash
# Gradio UI
python ui/gradio_app.py

# FastAPI backend only
uvicorn app.main:app --reload
```

### Run (real models)

Requires NVIDIA GPU. Models download automatically on first use (~10 GB total):

```bash
# Run all 6 real models (Stages 1-6)
DREAMSCAPE_STUB_STAGES= python ui/gradio_app.py

# Keep SDXL stubbed, run everything else real
DREAMSCAPE_STUB_STAGES=3 python ui/gradio_app.py
```

---

## Configuration

All settings are read from `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DREAMSCAPE_STUB_STAGES` | `3,4,6` | Comma-separated stage numbers to run as stubs |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DREAMSCAPE_OLLAMA_MODEL` | `llama3.1:8b` | Ollama model for Stages 1–2 |
| `DREAMSCAPE_WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `DREAMSCAPE_MUSICGEN_MODEL` | `facebook/musicgen-medium` | MusicGen HuggingFace model ID |
| `DREAMSCAPE_CACHE_DIR` | `cache` | Directory for SQLite DB and asset files |
| `DREAMSCAPE_CACHE_TTL_HOURS` | `24` | How long completed runs are cached |
| `DREAMSCAPE_LLM_BACKEND` | `ollama` | LLM routing: `ollama` for local dev, `hf` for HF Spaces |
| `HF_TOKEN` | _(none)_ | HF access token — required when `DREAMSCAPE_LLM_BACKEND=hf` |
| `DREAMSCAPE_DEFAULT_VOICE` | `female` | Default TTS voice (`female`, `male`) |

---

## Running Tests

```bash
# Full suite (67 tests, no model downloads required — all real-mode tests use mocks)
pytest tests/ -q

# Skip integration tests that require real models
pytest tests/ -q -m "not integration"
```

---

## Project Structure

```
DreamScapeAI/
├── app/
│   ├── main.py               # FastAPI entry point (REST API)
│   ├── orchestrator.py       # Pipeline coordinator
│   ├── cache.py              # SQLite + filesystem cache
│   ├── config.py             # Environment-variable config
│   ├── filters.py            # Content filter / blocklist
│   ├── models/
│   │   └── schemas.py        # Pydantic I/O schemas for all stages
│   └── stages/
│       ├── base.py           # BaseStage ABC (stub/real dispatcher)
│       ├── _ollama.py        # Shared Ollama HTTP client
│       ├── stage1_parse.py   # Prompt parsing
│       ├── stage2_expand.py  # Scene expansion
│       ├── stage3_visual.py  # Image generation
│       ├── stage4_narrate.py # TTS narration
│       ├── stage5_subtitle.py# Subtitle generation
│       ├── stage6_music.py   # Music generation
│       └── stage7_assemble.py# Video assembly
├── ui/
│   └── gradio_app.py         # Gradio web interface
├── tests/                    # 67 unit + integration tests
├── docs/                     # Design specs and implementation plans
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Research Contributions

This project is developed as a final-year research dissertation. The four contributions are:

1. **Multimodal Synchronization Engine** — a sequential pipeline that coordinates five independently-trained AI modalities (LLM, image gen, TTS, ASR, music) into a single coherent video artifact

2. **Emotion-Aware Storytelling** — cross-modal sentiment propagation: a single sentiment signal derived at Stage 1 modulates SDXL negative prompts, XTTS speaker selection, and MusicGen conditioning text

3. **Adaptive Scene Planning via LLM** — LLM-driven scene count and pacing (4–8 scenes, TTS-first timing correction) that adapts to both prompt complexity and target duration

4. **Human-Centered Evaluation Framework** — 20+ rater study with 5-point Likert scores across 6 dimensions, Fleiss' kappa inter-rater reliability ≥0.60, and automated metrics (CLIPScore, sync error <200ms, WER)

---

## Rater Study (Evaluation Framework Phase 2)

The deployed Gradio app exposes a second tab, **"Rate videos"**, used to collect
human ratings for the thesis evaluation. Each rater watches up to 20 short
videos and rates each on six 5-point Likert dimensions; responses are written
to a private HF Dataset repo as one JSON file per submission.

### One-time setup

1. Create a private dataset repo on https://huggingface.co/new-dataset
   (e.g. `your-username/dreamscape-rater-study`).
2. In the Space's **Settings → Variables and secrets**, add:
   - `DREAMSCAPE_RATER_DATASET` = `your-username/dreamscape-rater-study`
   - `HF_TOKEN` = a token with **write** scope on that dataset repo.
3. Generate the 20 study videos locally on a GPU runner and commit the MP4s
   into `study_videos/` via Git LFS:
   ```
   git lfs install
   git add study_videos/v01.mp4 ... study_videos/v20.mp4
   git commit -m "chore(study): add 20 rater-study videos"
   git push
   ```

### Data collection layout

- Each rater submission: `responses/<rater_id>/<video_id>.json`
- Each rater's final overall comment: `responses/<rater_id>/_overall.json`

Pull the dataset locally for analysis in Phase 3:

```
huggingface-cli download --repo-type dataset your-username/dreamscape-rater-study --local-dir responses_dump
```

---

## Contributors

| Name | GitHub | Role |
|---|---|---|
| Abhishek | [@abhi-30702](https://github.com/abhi-30702) | Author & Researcher |

---

## License

MIT
