# DreamScapeAI — Implementation Design Spec

**Date:** 2026-06-25
**Status:** Approved
**PRD Reference:** PRD_dreamscapeAI_v1.1.md
**Approach:** Thin skeleton first — stubs for GPU-heavy stages, swap in real models incrementally

---

## 1. Development Context

- **Dev machine:** Windows 11, NVIDIA GPU <8GB VRAM
- **Deployment target:** Hugging Face Spaces (T4, 16GB VRAM)
- **Strategy:** All stages implement a common `BaseStage` interface. GPU-heavy stages (3, 4, 6) have stub modes toggled via `DREAMSCAPE_STUB_STAGES` env var. Skeleton runs end-to-end locally on CPU; full models run on HF Spaces.

---

## 2. Project Structure

```
DreamScapeAI/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── orchestrator.py          # Central Orchestrator class
│   ├── cache.py                 # SQLite + filesystem cache (24hr TTL)
│   ├── models/
│   │   └── schemas.py           # Pydantic data models for all stage I/O
│   └── stages/
│       ├── base.py              # BaseStage ABC
│       ├── stage1_parse.py      # Prompt parsing (Llama-3.1-8B via Ollama)
│       ├── stage2_expand.py     # Scene expansion (Llama-3.1-8B via Ollama)
│       ├── stage3_visual.py     # Image generation (SDXL + CPU offload)
│       ├── stage4_narrate.py    # TTS narration (XTTS-v2)
│       ├── stage5_subtitle.py   # Subtitle generation (Whisper)
│       ├── stage6_music.py      # Music generation (MusicGen-medium)
│       └── stage7_assemble.py   # Video assembly (MoviePy + FFmpeg)
├── ui/
│   └── gradio_app.py            # Gradio UI
├── tests/
│   └── stages/                  # Per-stage unit tests with mock inputs
├── cache/                       # Runtime cache (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 3. Stage Interface Contract

All stages extend `BaseStage` in `app/stages/base.py`:

```python
from abc import ABC, abstractmethod

class BaseStage(ABC):
    stage_num: int
    stub_mode: bool  # set from env DREAMSCAPE_STUB_STAGES

    @abstractmethod
    def run(self, input: dict, cache_key: str) -> dict:
        """Execute stage. Returns output dict matching stage's output schema."""
        ...

    def is_cached(self, cache_key: str) -> bool: ...
    def load_cache(self, cache_key: str) -> dict: ...
    def save_cache(self, cache_key: str, output: dict): ...
```

**Rule:** Stages are stateless. The orchestrator owns all cross-stage state (sentiment, timing corrections, run metadata). Stages receive what they need in `input` and return a complete output dict.

**Stub contract:** When `stub_mode=True`, each stage returns a valid output dict of the correct schema using minimal compute (no GPU). Stubs are toggled by listing stage numbers in `DREAMSCAPE_STUB_STAGES=3,4,6`.

---

## 4. Data Models (`app/models/schemas.py`)

```python
# Stage 1 output
class ParsedPrompt(BaseModel):
    prompt: str
    sentiment: Literal["happy", "neutral", "sad"]
    duration_target_s: int          # 30, 60, or 90
    style: str                      # cinematic, documentary, anime, noir, horror
    key_entities: list[str]

# Stage 2 output
class SceneData(BaseModel):
    id: int
    description: str                # Visual description (100-150 tokens)
    narration_text: str             # Narration script (30-100 tokens)
    mood: Literal["happy", "neutral", "sad"]
    duration_estimate_s: float

class ScenePlan(BaseModel):
    scenes: list[SceneData]         # 4-8 scenes

# Stage 3 output
class ImageAsset(BaseModel):
    scene_id: int
    path: str                       # Absolute path to 1024x576 PNG
    width: int
    height: int

class VisualOutput(BaseModel):
    images: list[ImageAsset]

# Injected by orchestrator into Stage 4 input (derived from Stage 1 sentiment)
class SpeakerSettings(BaseModel):
    speaker_id: str             # e.g. "female_en_1", "male_en_1"
    pitch_semitones: float      # happy: +2, neutral: 0, sad: -1
    speed: float                # happy: 0.95, neutral: 1.0, sad: 1.05

# Stage 4 output
class AudioAsset(BaseModel):
    scene_id: int
    path: str                       # Absolute path to WAV
    duration_s: float               # Measured from actual TTS output

class NarrationOutput(BaseModel):
    audio: list[AudioAsset]
    total_duration_s: float

# Stage 5 output
class SubtitleEntry(BaseModel):
    index: int
    start_s: float
    end_s: float
    text: str

class SubtitleOutput(BaseModel):
    srt_path: str
    entries: list[SubtitleEntry]

# Stage 6 output
class MusicOutput(BaseModel):
    path: str
    duration_s: float

# Stage 7 output
class VideoOutput(BaseModel):
    path: str
    duration_s: float
    file_size_bytes: int

# Orchestrator run state
class PipelineRun(BaseModel):
    run_id: str
    prompt_hash: str
    parsed_prompt: Optional[ParsedPrompt]
    scene_plan: Optional[ScenePlan]
    visual_output: Optional[VisualOutput]
    narration_output: Optional[NarrationOutput]
    subtitle_output: Optional[SubtitleOutput]
    music_output: Optional[MusicOutput]
    video_output: Optional[VideoOutput]
    created_at: datetime
    status: Literal["pending", "running", "complete", "failed"]
```

---

## 5. Orchestrator (`app/orchestrator.py`)

```python
class Orchestrator:
    stages: list[BaseStage]         # [stage1, ..., stage7]
    cache: Cache

    def run_pipeline(self, prompt: str, duration: int, style: str, voice: str) -> PipelineRun:
        run = self.cache.get_or_create_run(prompt, duration, style, voice)
        for stage in self.stages:
            if self.cache.stage_complete(run.run_id, stage.stage_num):
                continue
            input = self._build_input(run, stage.stage_num)
            output = stage.run(input, cache_key=f"{run.run_id}/stage_{stage.stage_num}")
            self._apply_output(run, stage.stage_num, output)
        return run

    def run_stage(self, run_id: str, stage_num: int) -> PipelineRun:
        """Selective regeneration. Clears stage_num and all downstream cache entries,
        then resumes run_pipeline() — already-cached stages before stage_num are skipped."""
        self.cache.invalidate_from(run_id, stage_num)
        run = self.cache.load_run(run_id)
        return self.run_pipeline(run.parsed_prompt.prompt, ...restored from run...)
```

**Emotion propagation (in `_build_input`):**

| Stage | Injected field from Stage 1 sentiment |
|---|---|
| 3 (SDXL) | `negative_prompt` (derived from sentiment mapping) |
| 4 (XTTS) | `speaker_settings` (pitch, speed, speaker_id) |
| 6 (MusicGen) | `music_condition` (text description of mood) |

**TTS-first timing correction (after Stage 4):**
- Orchestrator measures `sum(audio.duration_s for audio in narration_output.audio)`
- If `abs(total - duration_target_s) > 2`: redistribute display durations proportionally
- Updates `scene.duration_estimate_s` in the run state before Stage 7

---

## 6. Caching (`app/cache.py`)

**SQLite schema:**
```sql
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    prompt_hash TEXT,
    params_json TEXT,
    status TEXT,
    created_at TIMESTAMP
);

CREATE TABLE stage_outputs (
    run_id TEXT,
    stage_num INTEGER,
    output_json TEXT,
    completed_at TIMESTAMP,
    PRIMARY KEY (run_id, stage_num)
);
```

**Asset storage:** `cache/{prompt_hash}/{run_id}/stage_{N}/`

**24hr TTL:** On `Cache.__init__()`, delete all `pipeline_runs` rows where `created_at < now - 24h`, cascade-delete their `stage_outputs` rows, and `rmtree` their asset directories.

**Invalidation:** `cache.invalidate_from(run_id, stage_num)` deletes `stage_outputs` rows where `stage_num >= N` and removes their asset directories.

---

## 7. Stage Implementation Details

### Stage 1 — Prompt Parsing
- **Input:** `{prompt, duration, style, voice}`
- **Output:** `ParsedPrompt`
- **Real:** Ollama HTTP API (`POST /api/generate`) with Llama-3.1-8B; structured JSON output via `format=json`
- **Stub:** Rule-based sentiment (keyword match: sad/happy/neutral); fixed duration; entities extracted by splitting nouns

### Stage 2 — Scene Expansion
- **Input:** `ParsedPrompt`
- **Output:** `ScenePlan` (4–8 scenes)
- **Real:** Ollama; prompt template asks for JSON array of scenes with descriptions, narration, mood, duration
- **Stub:** Returns 4 hardcoded scenes with placeholder text derived from key_entities

### Stage 3 — Visual Generation
- **Input:** `{scenes: list[SceneData], sentiment, style}`
- **Output:** `VisualOutput`
- **Real (local):** SDXL with `enable_model_cpu_offload()` (moves model to CPU between steps; fits <8GB); fp16 weights; 30 steps; 1024×576
- **Real (HF Spaces):** Full SDXL fp16 on T4; 50 steps; no CPU offload needed
- **Stub:** Generates a solid-color PIL image (color = mood: blue=sad, yellow=happy, grey=neutral) saved as PNG
- **NSFW handling:** Retry up to 3× with simplified prompt; fallback to placeholder after 3 failures

### Stage 4 — Narration
- **Input:** `{scenes: list[SceneData], speaker_settings}`
- **Output:** `NarrationOutput` (with measured `duration_s` per scene)
- **Real:** XTTS-v2 via `TTS` class; `tts.tts_to_file(text, speaker, language, file_path)`
- **Stub:** Uses `pyttsx3` (CPU TTS, Windows-compatible) or generates silence WAV of estimated duration

### Stage 5 — Subtitle Generation
- **Input:** `{audio: list[AudioAsset], scenes: list[SceneData]}`
- **Output:** `SubtitleOutput`
- **Real:** `whisper.transcribe(audio_path, word_timestamps=True)` per scene; concatenate SRT with cumulative timestamps
- **Stub:** Generates SRT directly from `narration_text` fields with estimated timing (no real ASR)

### Stage 6 — Music Generation
- **Input:** `{mood, total_duration_s, music_condition}`
- **Output:** `MusicOutput`
- **Real:** `MusicGen` via `audiocraft`; `model.generate_with_chroma()` or `model.generate()`; conditioned on text
- **Stub:** Returns pre-baked 90s silence WAV (trimmed to `total_duration_s`)

### Stage 7 — Video Assembly
- **Input:** `{images, narration, music, subtitles, scenes}`
- **Output:** `VideoOutput`
- **Real + Stub (same):** MoviePy — no GPU needed. Sequence images by narration duration, overlay audio, duck music (−6 dB under narration), burn SRT subtitles via FFmpeg, encode H.264 1080p 30fps
- **Always runs real** — no stub needed (CPU-only)

---

## 8. FastAPI Backend (`app/main.py`)

```
POST /generate              # Start pipeline run; returns run_id
GET  /run/{run_id}          # Poll run status + stage outputs
POST /run/{run_id}/regenerate/{stage_num}  # Selective regeneration
GET  /run/{run_id}/download # Stream final MP4
```

Single worker (HF Spaces constraint). No queue in v1.

---

## 9. Gradio UI (`ui/gradio_app.py`)

Five panels rendered sequentially as pipeline progresses:

1. **Input** — Textbox (500 char), duration radio (30s/60s/90s), style dropdown, voice radio
2. **Scene Plan** — After Stage 2: editable Dataframe of scenes; "Generate Video" button to confirm
3. **Progress** — Stage-by-stage status table with elapsed time per stage
4. **Inspection** — Image gallery (per scene), audio playback (per scene), music playback
5. **Output** — Inline `gr.Video`, download button

Regeneration: per-scene buttons in Inspection panel; "Regenerate Music" and "Regenerate All" in Output panel.

---

## 10. Content Filtering

**Prompt filter (Stage 1, pre-run):**
- Blocklist: ~40 keywords (violence, explicit content, copyrighted characters)
- Implementation: simple `any(kw in prompt.lower() for kw in BLOCKLIST)`
- Error: `ValueError("Your prompt contains restricted content. Please revise and try again.")`

**Image filter (Stage 3):**
- diffusers' built-in safety checker (enabled by default)
- Retry logic: up to 3× with simplified prompt (strip adjectives via regex)
- After 3 failures: return placeholder image + log

---

## 11. Environment Variables (`.env.example`)

```
DREAMSCAPE_STUB_STAGES=3,4,6    # Comma-separated; empty = all real
DREAMSCAPE_CACHE_DIR=cache/
DREAMSCAPE_CACHE_TTL_HOURS=24
OLLAMA_BASE_URL=http://localhost:11434
DREAMSCAPE_DEFAULT_VOICE=female
DREAMSCAPE_DEFAULT_DURATION=60
```

---

## 12. Local Setup Requirements

```
Python 3.10+
CUDA 12.x
Ollama (Windows installer) with llama3.1:8b pulled
FFmpeg on PATH
requirements.txt: fastapi, uvicorn, gradio, diffusers, transformers,
                  accelerate, TTS, whisper, audiocraft, moviepy,
                  pyttsx3, pydantic, python-dotenv, aiofiles
```

SDXL weights: auto-downloaded by diffusers on first Stage 3 real run.
XTTS-v2 weights: auto-downloaded by Coqui TTS on first Stage 4 real run.
MusicGen-medium: auto-downloaded by audiocraft on first Stage 6 real run.

---

## 13. Build Order

1. Project scaffold (directories, `.gitignore`, `requirements.txt`, `.env.example`)
2. Pydantic schemas (`app/models/schemas.py`)
3. Cache layer (`app/cache.py`) + SQLite init
4. `BaseStage` ABC (`app/stages/base.py`)
5. Stage stubs (all 7, stub mode only)
6. Orchestrator (`app/orchestrator.py`) — wires stubs end-to-end
7. FastAPI backend (`app/main.py`) — thin wrapper around orchestrator
8. Gradio UI (`ui/gradio_app.py`) — full end-to-end skeleton working
9. Replace stubs with real models: Stage 1→2 (Ollama), Stage 5 (Whisper), Stage 7 (MoviePy/FFmpeg), Stage 4 (XTTS), Stage 6 (MusicGen), Stage 3 (SDXL — last, heaviest)
10. Evaluation harness (CLIPScore, sync error measurement, duration accuracy logging)
    — Note: evaluation harness is deferred to its own spec (separate from this build).
      Build order item 10 means: add logging hooks in stages to capture timing/metric data.
      Full eval framework (rater study, automated benchmarks) is a later phase.
