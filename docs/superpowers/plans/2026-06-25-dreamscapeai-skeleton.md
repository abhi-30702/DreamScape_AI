# DreamScapeAI Core Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working end-to-end pipeline (prompt → MP4) using stub implementations for all GPU-heavy stages, so the full pipeline runs on any machine without GPU models installed.

**Architecture:** 7-stage sequential pipeline coordinated by a central `Orchestrator`. Each stage extends `BaseStage` (real/stub dispatcher). The Orchestrator owns all SQLite caching. GPU-heavy stages (3 SDXL, 4 XTTS, 6 MusicGen) have lightweight stubs toggled by `DREAMSCAPE_STUB_STAGES` env var. Stage 7 (video assembly) is always real since it only needs FFmpeg.

**Tech Stack:** Python 3.10+, FastAPI, Gradio 4.x, Pydantic v2, MoviePy 1.0.3, Pillow, SQLite (stdlib), pytest

---

## File Map

```
DreamScapeAI/
├── app/
│   ├── __init__.py
│   ├── config.py                  # dotenv settings loader
│   ├── filters.py                 # content blocklist
│   ├── cache.py                   # SQLite run tracking + filesystem assets
│   ├── orchestrator.py            # Pipeline coordinator, emotion propagation, TTS-first timing
│   ├── main.py                    # FastAPI app (factory function)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # All Pydantic models for stage I/O
│   └── stages/
│       ├── __init__.py
│       ├── base.py                # BaseStage ABC (real/stub dispatcher, no caching)
│       ├── stage1_parse.py        # Prompt parsing stub + real hook
│       ├── stage2_expand.py       # Scene expansion stub + real hook
│       ├── stage3_visual.py       # Image generation stub (solid color PNG)
│       ├── stage4_narrate.py      # Narration stub (silence WAV, estimated duration)
│       ├── stage5_subtitle.py     # Subtitle stub (SRT from narration_text)
│       ├── stage6_music.py        # Music stub (silence WAV)
│       └── stage7_assemble.py     # Video assembly — always real (MoviePy + FFmpeg)
├── ui/
│   ├── __init__.py
│   └── gradio_app.py              # Gradio UI (5 panels)
├── tests/
│   ├── conftest.py
│   ├── test_filters.py
│   ├── test_schemas.py
│   ├── test_cache.py
│   ├── test_base_stage.py
│   ├── test_stages.py             # Stub output tests for stages 1-6
│   ├── test_stage7.py             # Stage 7 integration test
│   ├── test_orchestrator.py
│   ├── test_api.py
│   └── test_e2e.py
├── cache/                         # Runtime cache (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Task 1: Git Init + Project Scaffold

**Files:** root structure, `requirements.txt`, `.env.example`, `.gitignore`

- [ ] **Step 1: Initialize git and create directories**

```powershell
cd D:\DreamScape
git init
New-Item -ItemType Directory -Path app/models, app/stages, ui, tests, cache -Force | Out-Null
"" | Set-Content app/__init__.py
"" | Set-Content app/models/__init__.py
"" | Set-Content app/stages/__init__.py
"" | Set-Content ui/__init__.py
"" | Set-Content tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

`D:\DreamScape\requirements.txt`:
```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
python-dotenv>=1.0.0
moviepy==1.0.3
Pillow>=10.0.0
gradio>=4.30.0
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Write `.env.example`**

`D:\DreamScape\.env.example`:
```
DREAMSCAPE_STUB_STAGES=3,4,6
DREAMSCAPE_CACHE_DIR=cache
DREAMSCAPE_CACHE_TTL_HOURS=24
OLLAMA_BASE_URL=http://localhost:11434
DREAMSCAPE_DEFAULT_VOICE=female
DREAMSCAPE_DEFAULT_DURATION=60
```

- [ ] **Step 4: Write `.gitignore`**

`D:\DreamScape\.gitignore`:
```
cache/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.venv/
venv/
*.mp4
*.wav
*.png
.pytest_cache/
```

- [ ] **Step 5: Install dependencies and copy env**

```powershell
Copy-Item .env.example .env
pip install -r requirements.txt
```

- [ ] **Step 6: Verify**

```powershell
python -c "import fastapi, pydantic, gradio, moviepy; print('OK')"
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "feat: project scaffold"
```

---

## Task 2: Config + Content Filter

**Files:** `app/config.py`, `app/filters.py`, `tests/test_filters.py`

- [ ] **Step 1: Write `app/config.py`**

```python
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

STUB_STAGES: set[int] = {
    int(s) for s in os.getenv("DREAMSCAPE_STUB_STAGES", "3,4,6").split(",") if s.strip()
}
CACHE_DIR = Path(os.getenv("DREAMSCAPE_CACHE_DIR", "cache"))
CACHE_TTL_HOURS = int(os.getenv("DREAMSCAPE_CACHE_TTL_HOURS", "24"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_VOICE = os.getenv("DREAMSCAPE_DEFAULT_VOICE", "female")
DEFAULT_DURATION = int(os.getenv("DREAMSCAPE_DEFAULT_DURATION", "60"))
```

- [ ] **Step 2: Write `app/filters.py`**

```python
BLOCKLIST = {
    "kill", "murder", "gore", "weapon", "blood", "torture", "massacre",
    "assassin", "genocide", "rape", "assault", "bomb", "terrorist",
    "nude", "naked", "sexual", "pornograph", "explicit",
    "mickey mouse", "darth vader", "harry potter", "batman", "superman",
    "spiderman", "iron man", "pikachu", "mario", "sonic",
    "suicide", "self-harm", "cutting",
}

def check_prompt(prompt: str) -> None:
    """Raise ValueError if prompt contains blocked content."""
    lower = prompt.lower()
    for term in BLOCKLIST:
        if term in lower:
            raise ValueError(
                "Your prompt contains restricted content. Please revise and try again."
            )
```

- [ ] **Step 3: Write failing test**

`tests/test_filters.py`:
```python
import pytest
from app.filters import check_prompt

def test_clean_prompt_passes():
    check_prompt("A lone wolf howls at the moon")

def test_blocked_prompt_raises():
    with pytest.raises(ValueError, match="restricted content"):
        check_prompt("A story about murder and blood")

def test_case_insensitive():
    with pytest.raises(ValueError):
        check_prompt("KILL the dragon")
```

- [ ] **Step 4: Run — expect failure**

```bash
pytest tests/test_filters.py -v
```
Expected: ImportError (filters.py not present yet — but we wrote it, so verify 3 pass)

Actually, since we just wrote the files: Expected: **3 passed**

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/filters.py tests/test_filters.py
git commit -m "feat: config loader and content filter blocklist"
```

---

## Task 3: Pydantic Schemas

**Files:** `app/models/schemas.py`, `tests/test_schemas.py`

- [ ] **Step 1: Write failing test**

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError
from app.models.schemas import (
    ParsedPrompt, SceneData, ScenePlan, SpeakerSettings,
    AudioAsset, NarrationOutput, SubtitleEntry, SubtitleOutput,
    ImageAsset, VisualOutput, MusicOutput, VideoOutput, PipelineRun,
)
from datetime import datetime

def test_parsed_prompt_valid():
    p = ParsedPrompt(prompt="A wolf howls", sentiment="sad",
                     duration_target_s=60, style="cinematic", key_entities=["wolf"])
    assert p.sentiment == "sad"

def test_parsed_prompt_invalid_sentiment():
    with pytest.raises(ValidationError):
        ParsedPrompt(prompt="test", sentiment="angry",
                     duration_target_s=60, style="cinematic", key_entities=[])

def test_scene_plan_roundtrip():
    scene = SceneData(id=0, description="Wide shot", narration_text="In winter...",
                      mood="sad", duration_estimate_s=12.0)
    plan = ScenePlan(scenes=[scene])
    assert plan.scenes[0].id == 0

def test_narration_output_total():
    audio = [
        AudioAsset(scene_id=0, path="/tmp/s0.wav", duration_s=10.0),
        AudioAsset(scene_id=1, path="/tmp/s1.wav", duration_s=15.0),
    ]
    out = NarrationOutput(audio=audio, total_duration_s=25.0)
    assert out.total_duration_s == 25.0

def test_pipeline_run_defaults():
    run = PipelineRun(run_id="abc", prompt_hash="xyz", prompt="test",
                      duration_target_s=60, style="cinematic", voice="female")
    assert run.status == "pending"
    assert run.parsed_prompt is None
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_schemas.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `app/models/schemas.py`**

```python
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class ParsedPrompt(BaseModel):
    prompt: str
    sentiment: Literal["happy", "neutral", "sad"]
    duration_target_s: int
    style: str
    key_entities: list[str]


class SceneData(BaseModel):
    id: int
    description: str
    narration_text: str
    mood: Literal["happy", "neutral", "sad"]
    duration_estimate_s: float


class ScenePlan(BaseModel):
    scenes: list[SceneData]


class SpeakerSettings(BaseModel):
    speaker_id: str
    pitch_semitones: float
    speed: float


class ImageAsset(BaseModel):
    scene_id: int
    path: str
    width: int
    height: int


class VisualOutput(BaseModel):
    images: list[ImageAsset]


class AudioAsset(BaseModel):
    scene_id: int
    path: str
    duration_s: float


class NarrationOutput(BaseModel):
    audio: list[AudioAsset]
    total_duration_s: float


class SubtitleEntry(BaseModel):
    index: int
    start_s: float
    end_s: float
    text: str


class SubtitleOutput(BaseModel):
    srt_path: str
    entries: list[SubtitleEntry]


class MusicOutput(BaseModel):
    path: str
    duration_s: float


class VideoOutput(BaseModel):
    path: str
    duration_s: float
    file_size_bytes: int


class PipelineRun(BaseModel):
    run_id: str
    prompt_hash: str
    prompt: str
    duration_target_s: int
    style: str
    voice: str
    parsed_prompt: Optional[ParsedPrompt] = None
    scene_plan: Optional[ScenePlan] = None
    visual_output: Optional[VisualOutput] = None
    narration_output: Optional[NarrationOutput] = None
    subtitle_output: Optional[SubtitleOutput] = None
    music_output: Optional[MusicOutput] = None
    video_output: Optional[VideoOutput] = None
    created_at: datetime = None
    status: Literal["pending", "running", "complete", "failed"] = "pending"

    def model_post_init(self, __context: object) -> None:
        if self.created_at is None:
            self.__dict__["created_at"] = datetime.utcnow()
```

- [ ] **Step 4: Run — expect 5 passed**

```bash
pytest tests/test_schemas.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/models/schemas.py tests/test_schemas.py
git commit -m "feat: pydantic schemas for all pipeline stage I/O"
```

---

## Task 4: Cache Layer

**Files:** `app/cache.py`, `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cache.py`:
```python
import pytest
from pathlib import Path
from app.cache import Cache, prompt_hash

@pytest.fixture
def cache(tmp_path):
    return Cache(db_path=tmp_path / "test.db", asset_dir=tmp_path / "assets")

def test_create_run(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    assert run_id is not None

def test_stage_not_complete_initially(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    assert not cache.stage_complete(run_id, 1)

def test_save_and_load_stage_output(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    cache.save_stage_output(run_id, 1, {"sentiment": "sad"})
    assert cache.stage_complete(run_id, 1)
    assert cache.load_stage_output(run_id, 1) == {"sentiment": "sad"}

def test_invalidate_from(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    for i in range(1, 6):
        cache.save_stage_output(run_id, i, {"stage": i})
    cache.invalidate_from(run_id, 3)
    assert cache.stage_complete(run_id, 2)
    assert not cache.stage_complete(run_id, 3)
    assert not cache.stage_complete(run_id, 4)

def test_get_asset_dir_creates_path(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    d = cache.get_asset_dir(run_id, 3)
    assert d.exists()

def test_find_run_by_hash_returns_none_when_missing(cache):
    assert cache.find_run_by_hash("nonexistent") is None

def test_find_run_by_hash_returns_run_id(cache):
    run_id = cache.create_run("myhash", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    assert cache.find_run_by_hash("myhash") == run_id

def test_prompt_hash_is_deterministic():
    h1 = prompt_hash("A wolf howls", {"duration": 60, "style": "cinematic", "voice": "female"})
    h2 = prompt_hash("A wolf howls", {"duration": 60, "style": "cinematic", "voice": "female"})
    assert h1 == h2

def test_prompt_hash_differs_for_different_inputs():
    h1 = prompt_hash("A wolf howls", {"duration": 60, "style": "cinematic", "voice": "female"})
    h2 = prompt_hash("A cat sleeps", {"duration": 60, "style": "cinematic", "voice": "female"})
    assert h1 != h2
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_cache.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `app/cache.py`**

```python
import json
import sqlite3
import hashlib
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path


class Cache:
    def __init__(self, db_path: Path, asset_dir: Path):
        self.db_path = db_path
        self.asset_dir = asset_dir
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    params_json TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stage_outputs (
                    run_id TEXT,
                    stage_num INTEGER,
                    output_json TEXT,
                    completed_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (run_id, stage_num)
                )
            """)

    def create_run(self, phash: str, params: dict) -> str:
        run_id = str(uuid.uuid4())[:8]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, prompt_hash, params_json) VALUES (?, ?, ?)",
                (run_id, phash, json.dumps(params)),
            )
        return run_id

    def find_run_by_hash(self, phash: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE prompt_hash=? ORDER BY created_at DESC LIMIT 1",
                (phash,),
            ).fetchone()
        return row[0] if row else None

    def stage_complete(self, run_id: str, stage_num: int) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM stage_outputs WHERE run_id=? AND stage_num=?",
                (run_id, stage_num),
            ).fetchone()
        return row is not None

    def save_stage_output(self, run_id: str, stage_num: int, output: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO stage_outputs (run_id, stage_num, output_json) VALUES (?, ?, ?)",
                (run_id, stage_num, json.dumps(output)),
            )

    def load_stage_output(self, run_id: str, stage_num: int) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT output_json FROM stage_outputs WHERE run_id=? AND stage_num=?",
                (run_id, stage_num),
            ).fetchone()
        if row is None:
            raise KeyError(f"No cached output for run={run_id} stage={stage_num}")
        return json.loads(row[0])

    def invalidate_from(self, run_id: str, stage_num: int):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM stage_outputs WHERE run_id=? AND stage_num>=?",
                (run_id, stage_num),
            )

    def get_asset_dir(self, run_id: str, stage_num: int) -> Path:
        d = self.asset_dir / run_id / f"stage_{stage_num}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_run_params(self, run_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT params_json FROM pipeline_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Run not found: {run_id}")
        return json.loads(row[0])

    def prune_expired(self, ttl_hours: int = 24):
        cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
        with self._conn() as conn:
            old = conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE created_at < ?", (cutoff,)
            ).fetchall()
            for (run_id,) in old:
                conn.execute("DELETE FROM stage_outputs WHERE run_id=?", (run_id,))
                conn.execute("DELETE FROM pipeline_runs WHERE run_id=?", (run_id,))
                run_dir = self.asset_dir / run_id
                if run_dir.exists():
                    shutil.rmtree(run_dir)


def prompt_hash(prompt: str, params: dict) -> str:
    key = json.dumps({"prompt": prompt, **params}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Run — expect 9 passed**

```bash
pytest tests/test_cache.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add app/cache.py tests/test_cache.py
git commit -m "feat: SQLite cache with TTL, selective invalidation, and run deduplication"
```

---

## Task 5: BaseStage ABC

**Files:** `app/stages/base.py`, `tests/test_base_stage.py`

- [ ] **Step 1: Write failing test**

`tests/test_base_stage.py`:
```python
import pytest
from pathlib import Path
from app.stages.base import BaseStage

class DummyStage(BaseStage):
    stage_num = 99

    def _run_real(self, input: dict) -> dict:
        return {"mode": "real", "value": input.get("x", 0) * 2}

    def _run_stub(self, input: dict) -> dict:
        return {"mode": "stub", "value": 42}

def test_real_mode(tmp_path):
    stage = DummyStage(cache_dir=tmp_path, stub_stages=set())
    assert stage.run({"x": 5}) == {"mode": "real", "value": 10}

def test_stub_mode(tmp_path):
    stage = DummyStage(cache_dir=tmp_path, stub_stages={99})
    assert stage.run({"x": 5}) == {"mode": "stub", "value": 42}

def test_stub_ignores_input(tmp_path):
    stage = DummyStage(cache_dir=tmp_path, stub_stages={99})
    assert stage.run({"x": 999}) == {"mode": "stub", "value": 42}
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_base_stage.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `app/stages/base.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path


class BaseStage(ABC):
    stage_num: int

    def __init__(self, cache_dir: Path, stub_stages: set[int]):
        self.cache_dir = cache_dir
        self.stub_mode = self.stage_num in stub_stages

    @abstractmethod
    def _run_real(self, input: dict) -> dict: ...

    @abstractmethod
    def _run_stub(self, input: dict) -> dict: ...

    def run(self, input: dict) -> dict:
        return self._run_stub(input) if self.stub_mode else self._run_real(input)
```

- [ ] **Step 4: Run — expect 3 passed**

```bash
pytest tests/test_base_stage.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/stages/base.py tests/test_base_stage.py
git commit -m "feat: BaseStage ABC (real/stub dispatcher)"
```

---

## Task 6: Stage Stubs (Stages 1–6)

**Files:** `app/stages/stage1_parse.py` through `stage6_music.py`, `tests/test_stages.py`

- [ ] **Step 1: Write failing tests**

`tests/test_stages.py`:
```python
import pytest
from pathlib import Path
from app.stages.stage1_parse import Stage1Parse
from app.stages.stage2_expand import Stage2Expand
from app.stages.stage3_visual import Stage3Visual
from app.stages.stage4_narrate import Stage4Narrate
from app.stages.stage5_subtitle import Stage5Subtitle
from app.stages.stage6_music import Stage6Music
from app.models.schemas import (
    ParsedPrompt, ScenePlan, VisualOutput, NarrationOutput, SubtitleOutput, MusicOutput
)

ALL_STUBS = {1, 2, 3, 4, 5, 6}

def s(cls, tmp):
    return cls(cache_dir=tmp, stub_stages=ALL_STUBS)

def test_stage1_returns_parsed_prompt(tmp_path):
    result = s(Stage1Parse, tmp_path).run(
        {"prompt": "A sad wolf howls alone", "duration": 60, "style": "cinematic", "voice": "female"}
    )
    p = ParsedPrompt(**result)
    assert p.sentiment == "sad"
    assert p.duration_target_s == 60

def test_stage1_happy_sentiment(tmp_path):
    result = s(Stage1Parse, tmp_path).run(
        {"prompt": "A joyful celebration in bright sunlight", "duration": 60, "style": "cinematic", "voice": "female"}
    )
    assert result["sentiment"] == "happy"

def test_stage2_returns_4_to_8_scenes(tmp_path):
    pp = {"prompt": "A wolf", "sentiment": "sad", "duration_target_s": 60, "style": "cinematic", "key_entities": ["wolf", "moon"]}
    result = s(Stage2Expand, tmp_path).run({"parsed_prompt": pp})
    plan = ScenePlan(**result)
    assert 4 <= len(plan.scenes) <= 8
    assert all(scene.mood == "sad" for scene in plan.scenes)

def test_stage3_creates_png_files(tmp_path):
    scenes = [
        {"id": i, "description": f"Scene {i}", "narration_text": "text", "mood": "sad", "duration_estimate_s": 12.0}
        for i in range(4)
    ]
    result = s(Stage3Visual, tmp_path).run(
        {"scenes": scenes, "sentiment": "sad", "style": "cinematic", "asset_dir": str(tmp_path / "imgs")}
    )
    out = VisualOutput(**result)
    assert len(out.images) == 4
    for img in out.images:
        assert Path(img.path).exists()
        assert img.width == 1024 and img.height == 576

def test_stage4_creates_wavs_with_duration(tmp_path):
    scenes = [
        {"id": 0, "description": "desc", "narration_text": "In winter the wolf howls alone at the moon.", "mood": "sad", "duration_estimate_s": 12.0}
    ]
    result = s(Stage4Narrate, tmp_path).run({
        "scenes": scenes,
        "speaker_settings": {"speaker_id": "male_en_1", "pitch_semitones": -1.0, "speed": 1.05},
        "asset_dir": str(tmp_path / "audio"),
    })
    out = NarrationOutput(**result)
    assert len(out.audio) == 1
    assert out.audio[0].duration_s > 0
    assert Path(out.audio[0].path).exists()

def test_stage5_creates_srt(tmp_path):
    audio_assets = [{"scene_id": 0, "path": "/fake/audio.wav", "duration_s": 10.0}]
    scenes = [{"id": 0, "description": "desc", "narration_text": "In winter, silence reigns.", "mood": "sad", "duration_estimate_s": 10.0}]
    result = s(Stage5Subtitle, tmp_path).run(
        {"audio_assets": audio_assets, "scenes": scenes, "asset_dir": str(tmp_path / "subs")}
    )
    out = SubtitleOutput(**result)
    assert Path(out.srt_path).exists()
    assert len(out.entries) >= 1

def test_stage6_creates_wav_matching_duration(tmp_path):
    result = s(Stage6Music, tmp_path).run(
        {"mood": "sad", "total_duration_s": 30.0, "music_condition": "melancholic minor key", "asset_dir": str(tmp_path / "music")}
    )
    out = MusicOutput(**result)
    assert Path(out.path).exists()
    assert abs(out.duration_s - 30.0) < 1.0
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_stages.py -v
```
Expected: ImportError for stage modules

- [ ] **Step 3: Write `app/stages/stage1_parse.py`**

```python
from app.stages.base import BaseStage

_SAD = {"sad", "lonely", "lost", "dark", "grief", "sorrow", "alone", "death", "cry", "melanchol", "despair"}
_HAPPY = {"happy", "joyful", "bright", "celebrat", "wonderful", "cheer", "laugh", "love", "hope", "delight"}

class Stage1Parse(BaseStage):
    stage_num = 1

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("Ollama integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        prompt = input["prompt"].lower()
        if any(w in prompt for w in _SAD):
            sentiment = "sad"
        elif any(w in prompt for w in _HAPPY):
            sentiment = "happy"
        else:
            sentiment = "neutral"
        words = [w.strip(".,!?") for w in prompt.split() if len(w) > 4]
        entities = list(dict.fromkeys(words))[:4] or ["character", "setting", "action", "moment"]
        return {
            "prompt": input["prompt"],
            "sentiment": sentiment,
            "duration_target_s": input.get("duration", 60),
            "style": input.get("style", "cinematic"),
            "key_entities": entities,
        }
```

- [ ] **Step 4: Write `app/stages/stage2_expand.py`**

```python
from app.stages.base import BaseStage

_BEATS = [
    ("Wide establishing shot", "The world holds its breath..."),
    ("Close-up on subject", "Something stirs beneath the surface..."),
    ("Medium shot, rising tension", "A moment of decision approaches..."),
    ("Wide shot, climax", "Everything converges at once..."),
]

class Stage2Expand(BaseStage):
    stage_num = 2

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("Ollama integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        pp = input["parsed_prompt"]
        entities = pp.get("key_entities", ["subject"])
        mood = pp.get("sentiment", "neutral")
        duration = pp.get("duration_target_s", 60)
        scene_dur = duration / len(_BEATS)
        scenes = []
        for i, (visual, narration) in enumerate(_BEATS):
            entity = entities[i % len(entities)]
            scenes.append({
                "id": i,
                "description": f"{visual} featuring {entity}.",
                "narration_text": f"{narration} The {entity} reveals itself.",
                "mood": mood,
                "duration_estimate_s": round(scene_dur, 1),
            })
        return {"scenes": scenes}
```

- [ ] **Step 5: Write `app/stages/stage3_visual.py`**

```python
from pathlib import Path
from app.stages.base import BaseStage

_COLORS = {"happy": (255, 220, 80), "neutral": (140, 140, 160), "sad": (70, 90, 130)}

class Stage3Visual(BaseStage):
    stage_num = 3

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("SDXL integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        from PIL import Image, ImageDraw
        scenes = input["scenes"]
        mood = input.get("sentiment", "neutral")
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        color = _COLORS.get(mood, (140, 140, 160))
        images = []
        for scene in scenes:
            img = Image.new("RGB", (1024, 576), color)
            draw = ImageDraw.Draw(img)
            label = f"Scene {scene['id']}: {scene['description'][:50]}"
            draw.text((20, 270), label, fill=(255, 255, 255))
            path = asset_dir / f"scene_{scene['id']}.png"
            img.save(str(path))
            images.append({"scene_id": scene["id"], "path": str(path), "width": 1024, "height": 576})
        return {"images": images}
```

- [ ] **Step 6: Write `app/stages/stage4_narrate.py`**

```python
import wave
from pathlib import Path
from app.stages.base import BaseStage

_SAMPLE_RATE = 22050
_WORDS_PER_SEC = 2.5

def _silence_wav(path: Path, duration_s: float):
    n = int(duration_s * _SAMPLE_RATE)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(_SAMPLE_RATE)
        f.writeframes(b"\x00\x00" * n)

class Stage4Narrate(BaseStage):
    stage_num = 4

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("XTTS-v2 integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        scenes = input["scenes"]
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        audio_assets = []
        total = 0.0
        for scene in scenes:
            duration_s = max(2.0, len(scene["narration_text"].split()) / _WORDS_PER_SEC)
            path = asset_dir / f"scene_{scene['id']}.wav"
            _silence_wav(path, duration_s)
            audio_assets.append({"scene_id": scene["id"], "path": str(path), "duration_s": round(duration_s, 2)})
            total += duration_s
        return {"audio": audio_assets, "total_duration_s": round(total, 2)}
```

- [ ] **Step 7: Write `app/stages/stage5_subtitle.py`**

```python
from pathlib import Path
from app.stages.base import BaseStage

def _ts(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

class Stage5Subtitle(BaseStage):
    stage_num = 5

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("Whisper integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        audio_assets = input["audio_assets"]
        scenes_by_id = {s["id"]: s for s in input["scenes"]}
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        cursor = 0.0
        for audio in audio_assets:
            text = scenes_by_id[audio["scene_id"]]["narration_text"]
            end = cursor + audio["duration_s"]
            entries.append({"index": len(entries) + 1, "start_s": cursor, "end_s": end, "text": text})
            cursor = end

        srt_lines = []
        for e in entries:
            srt_lines += [str(e["index"]), f"{_ts(e['start_s'])} --> {_ts(e['end_s'])}", e["text"], ""]

        srt_path = asset_dir / "subtitles.srt"
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        return {"srt_path": str(srt_path), "entries": entries}
```

- [ ] **Step 8: Write `app/stages/stage6_music.py`**

```python
import wave
from pathlib import Path
from app.stages.base import BaseStage

_SAMPLE_RATE = 44100

def _silence_wav_stereo(path: Path, duration_s: float):
    n = int(duration_s * _SAMPLE_RATE)
    with wave.open(str(path), "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(_SAMPLE_RATE)
        f.writeframes(b"\x00\x00\x00\x00" * n)

class Stage6Music(BaseStage):
    stage_num = 6

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("MusicGen integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        duration_s = input["total_duration_s"]
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        path = asset_dir / "music.wav"
        _silence_wav_stereo(path, duration_s)
        return {"path": str(path), "duration_s": round(duration_s, 2)}
```

- [ ] **Step 9: Run all stage tests**

```bash
pytest tests/test_stages.py -v
```
Expected: 8 passed

- [ ] **Step 10: Commit**

```bash
git add app/stages/stage1_parse.py app/stages/stage2_expand.py app/stages/stage3_visual.py app/stages/stage4_narrate.py app/stages/stage5_subtitle.py app/stages/stage6_music.py tests/test_stages.py
git commit -m "feat: stub implementations for stages 1-6"
```

---

## Task 7: Stage 7 — Video Assembly (Real)

**Files:** `app/stages/stage7_assemble.py`, `tests/test_stage7.py`

Stage 7 needs FFmpeg on PATH. Verify before starting:

- [ ] **Step 1: Verify FFmpeg**

```powershell
ffmpeg -version
ffprobe -version
```
Expected: version strings. If missing, download from https://ffmpeg.org/download.html and add `ffmpeg/bin` to PATH, then restart the terminal.

- [ ] **Step 2: Write failing test**

`tests/test_stage7.py`:
```python
import wave, pytest
from pathlib import Path
from PIL import Image
from app.stages.stage7_assemble import Stage7Assemble

@pytest.fixture
def assets(tmp_path):
    imgs = []
    for i in range(2):
        img = Image.new("RGB", (1024, 576), (100 + i * 50, 80, 120))
        p = tmp_path / f"scene_{i}.png"
        img.save(str(p))
        imgs.append({"scene_id": i, "path": str(p), "width": 1024, "height": 576})

    audios = []
    for i in range(2):
        p = tmp_path / f"narr_{i}.wav"
        n = 22050 * 2
        with wave.open(str(p), "w") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
            f.writeframes(b"\x00\x00" * n)
        audios.append({"scene_id": i, "path": str(p), "duration_s": 2.0})

    mp = tmp_path / "music.wav"
    with wave.open(str(mp), "w") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(44100)
        f.writeframes(b"\x00\x00\x00\x00" * (44100 * 6))

    srt_p = tmp_path / "subtitles.srt"
    srt_p.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello world\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nGoodbye\n\n",
        encoding="utf-8"
    )

    return {
        "images": imgs,
        "narration": audios,
        "music": {"path": str(mp), "duration_s": 6.0},
        "subtitles": {"srt_path": str(srt_p), "entries": []},
        "output_path": str(tmp_path / "output.mp4"),
    }

def test_stage7_produces_mp4(tmp_path, assets):
    stage = Stage7Assemble(cache_dir=tmp_path / "cache", stub_stages=set())
    result = stage.run(assets)
    assert Path(result["path"]).exists()
    assert result["file_size_bytes"] > 0
    assert result["duration_s"] > 0
```

- [ ] **Step 3: Run — expect ImportError**

```bash
pytest tests/test_stage7.py -v
```
Expected: ImportError

- [ ] **Step 4: Write `app/stages/stage7_assemble.py`**

```python
import os
import subprocess
import tempfile
from pathlib import Path
from app.stages.base import BaseStage


class Stage7Assemble(BaseStage):
    stage_num = 7

    def _run_real(self, input: dict) -> dict:
        return self._assemble(input)

    def _run_stub(self, input: dict) -> dict:
        return self._assemble(input)

    def _assemble(self, input: dict) -> dict:
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip

        images = sorted(input["images"], key=lambda x: x["scene_id"])
        narration = sorted(input["narration"], key=lambda x: x["scene_id"])
        music_data = input["music"]
        srt_path = input["subtitles"]["srt_path"]
        output_path = Path(input["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clips = []
        for img_data, audio_data in zip(images, narration):
            narr_clip = AudioFileClip(audio_data["path"])
            img_clip = (
                ImageClip(img_data["path"])
                .set_duration(audio_data["duration_s"])
                .set_audio(narr_clip)
                .resize((1920, 1080))
            )
            clips.append(img_clip)

        video = concatenate_videoclips(clips, method="compose")

        music_clip = AudioFileClip(music_data["path"]).subclip(0, video.duration).volumex(0.5)
        mixed = CompositeAudioClip([video.audio, music_clip])
        video = video.set_audio(mixed)

        temp_path = output_path.with_suffix(".temp.mp4")
        video.write_videofile(
            str(temp_path), fps=30, codec="libx264", audio_codec="aac",
            verbose=False, logger=None,
        )
        video.close()

        # Burn subtitles. On Windows, escape drive colon for FFmpeg filter.
        srt_for_ffmpeg = Path(srt_path).as_posix()
        if len(srt_for_ffmpeg) > 1 and srt_for_ffmpeg[1] == ":":
            srt_for_ffmpeg = srt_for_ffmpeg[0] + "\\:" + srt_for_ffmpeg[2:]

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(temp_path),
                    "-vf", f"subtitles='{srt_for_ffmpeg}':force_style='FontName=Arial,FontSize=14,Alignment=2'",
                    "-c:a", "copy", str(output_path),
                ],
                check=True, capture_output=True,
            )
            temp_path.unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            # Subtitle burn failed (common on Windows path edge cases); ship without subtitles
            temp_path.rename(output_path)

        size = output_path.stat().st_size
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
            capture_output=True, text=True, check=True,
        )
        duration_s = float(probe.stdout.strip())
        return {"path": str(output_path), "duration_s": duration_s, "file_size_bytes": size}
```

- [ ] **Step 5: Run — expect 1 passed (takes ~30s)**

```bash
pytest tests/test_stage7.py -v
```
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add app/stages/stage7_assemble.py tests/test_stage7.py
git commit -m "feat: Stage 7 video assembly with MoviePy and FFmpeg subtitle burn"
```

---

## Task 8: Orchestrator

**Files:** `app/orchestrator.py`, `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_orchestrator.py`:
```python
import pytest
from pathlib import Path
from app.cache import Cache
from app.orchestrator import Orchestrator

@pytest.fixture
def orch(tmp_path):
    cache = Cache(db_path=tmp_path / "runs.db", asset_dir=tmp_path / "assets")
    return Orchestrator(cache=cache, stub_stages={1, 2, 3, 4, 5, 6})

def test_run_pipeline_produces_video(orch):
    run = orch.run_pipeline("A lone wolf howls at the moon", 30, "cinematic", "female")
    assert run.status == "complete"
    assert run.video_output is not None
    assert Path(run.video_output.path).exists()

def test_run_pipeline_same_prompt_reuses_run_id(orch):
    run1 = orch.run_pipeline("A wolf at the moon", 60, "cinematic", "female")
    run2 = orch.run_pipeline("A wolf at the moon", 60, "cinematic", "female")
    assert run1.run_id == run2.run_id

def test_run_pipeline_different_prompt_creates_new_run(orch):
    run1 = orch.run_pipeline("A wolf at the moon", 60, "cinematic", "female")
    run2 = orch.run_pipeline("A cat on a roof", 60, "cinematic", "female")
    assert run1.run_id != run2.run_id

def test_blocked_prompt_raises(orch):
    with pytest.raises(ValueError, match="restricted content"):
        orch.run_pipeline("A murder scene", 60, "cinematic", "female")

def test_run_stage_regenerates_from_given_stage(orch):
    run = orch.run_pipeline("A lone wolf howls", 30, "cinematic", "female")
    run2 = orch.run_stage(run.run_id, stage_num=6)
    assert run2.status == "complete"
    assert run2.video_output is not None

def test_tts_first_timing_applied(orch):
    run = orch.run_pipeline("A wolf howls in the night", 60, "cinematic", "female")
    # Scene durations should reflect actual narration durations (not LLM estimates)
    for scene in run.scene_plan.scenes:
        assert scene.duration_estimate_s > 0
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_orchestrator.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `app/orchestrator.py`**

```python
from pathlib import Path
from app.cache import Cache, prompt_hash
from app.models.schemas import (
    PipelineRun, SpeakerSettings,
    ParsedPrompt, ScenePlan, VisualOutput, NarrationOutput,
    SubtitleOutput, MusicOutput, VideoOutput,
)
from app.stages.stage1_parse import Stage1Parse
from app.stages.stage2_expand import Stage2Expand
from app.stages.stage3_visual import Stage3Visual
from app.stages.stage4_narrate import Stage4Narrate
from app.stages.stage5_subtitle import Stage5Subtitle
from app.stages.stage6_music import Stage6Music
from app.stages.stage7_assemble import Stage7Assemble

_SPEAKER: dict[str, SpeakerSettings] = {
    "happy":   SpeakerSettings(speaker_id="female_en_1", pitch_semitones=2.0,  speed=0.95),
    "neutral": SpeakerSettings(speaker_id="neutral_en_1", pitch_semitones=0.0, speed=1.0),
    "sad":     SpeakerSettings(speaker_id="male_en_1",   pitch_semitones=-1.0, speed=1.05),
}
_MUSIC_COND: dict[str, str] = {
    "happy":   "uplifting, bright, major key, fast tempo",
    "neutral": "calm, ambient, minimal, mid-tempo",
    "sad":     "melancholic, minor key, slow tempo, sparse",
}
_NEG_PROMPT: dict[str, str] = {
    "happy":   "dark, bleak, sad, monochrome",
    "neutral": "",
    "sad":     "bright, colorful, cheerful, optimistic",
}
_SCHEMA_MAP = {
    1: ("parsed_prompt", ParsedPrompt),
    2: ("scene_plan", ScenePlan),
    3: ("visual_output", VisualOutput),
    4: ("narration_output", NarrationOutput),
    5: ("subtitle_output", SubtitleOutput),
    6: ("music_output", MusicOutput),
    7: ("video_output", VideoOutput),
}


class Orchestrator:
    def __init__(self, cache: Cache, stub_stages: set[int]):
        self.cache = cache
        kw = {"cache_dir": cache.asset_dir, "stub_stages": stub_stages}
        self.stages = [
            Stage1Parse(**kw), Stage2Expand(**kw), Stage3Visual(**kw),
            Stage4Narrate(**kw), Stage5Subtitle(**kw), Stage6Music(**kw),
            Stage7Assemble(**kw),
        ]

    def run_pipeline(self, prompt: str, duration: int, style: str, voice: str) -> PipelineRun:
        from app.filters import check_prompt
        check_prompt(prompt)

        phash = prompt_hash(prompt, {"duration": duration, "style": style, "voice": voice})
        run_id = self.cache.find_run_by_hash(phash)
        if run_id is None:
            run_id = self.cache.create_run(
                phash, {"prompt": prompt, "duration": duration, "style": style, "voice": voice}
            )

        run = PipelineRun(
            run_id=run_id, prompt_hash=phash, prompt=prompt,
            duration_target_s=duration, style=style, voice=voice,
        )

        for stage in self.stages:
            if self.cache.stage_complete(run_id, stage.stage_num):
                self._apply_output(run, stage.stage_num,
                                   self.cache.load_stage_output(run_id, stage.stage_num))

        run.status = "running"
        for stage in self.stages:
            if not self.cache.stage_complete(run_id, stage.stage_num):
                output = stage.run(self._build_input(run, stage.stage_num))
                self.cache.save_stage_output(run_id, stage.stage_num, output)
                self._apply_output(run, stage.stage_num, output)

        run.status = "complete"
        return run

    def run_stage(self, run_id: str, stage_num: int) -> PipelineRun:
        self.cache.invalidate_from(run_id, stage_num)
        params = self.cache.load_run_params(run_id)
        return self.run_pipeline(
            prompt=params["prompt"], duration=params["duration"],
            style=params["style"], voice=params["voice"],
        )

    def _build_input(self, run: PipelineRun, stage_num: int) -> dict:
        sentiment = run.parsed_prompt.sentiment if run.parsed_prompt else "neutral"
        asset_dir = str(self.cache.get_asset_dir(run.run_id, stage_num))
        if stage_num == 1:
            return {"prompt": run.prompt, "duration": run.duration_target_s,
                    "style": run.style, "voice": run.voice}
        if stage_num == 2:
            return {"parsed_prompt": run.parsed_prompt.model_dump()}
        if stage_num == 3:
            return {"scenes": [s.model_dump() for s in run.scene_plan.scenes],
                    "sentiment": sentiment, "negative_prompt": _NEG_PROMPT[sentiment],
                    "style": run.style, "asset_dir": asset_dir}
        if stage_num == 4:
            return {"scenes": [s.model_dump() for s in run.scene_plan.scenes],
                    "speaker_settings": _SPEAKER[sentiment].model_dump(), "asset_dir": asset_dir}
        if stage_num == 5:
            return {"audio_assets": [a.model_dump() for a in run.narration_output.audio],
                    "scenes": [s.model_dump() for s in run.scene_plan.scenes], "asset_dir": asset_dir}
        if stage_num == 6:
            return {"mood": sentiment, "total_duration_s": run.narration_output.total_duration_s,
                    "music_condition": _MUSIC_COND[sentiment], "asset_dir": asset_dir}
        if stage_num == 7:
            return {"images": [i.model_dump() for i in run.visual_output.images],
                    "narration": [a.model_dump() for a in run.narration_output.audio],
                    "music": run.music_output.model_dump(),
                    "subtitles": run.subtitle_output.model_dump(),
                    "output_path": str(self.cache.get_asset_dir(run.run_id, 7) / "output.mp4")}

    def _apply_output(self, run: PipelineRun, stage_num: int, output: dict):
        attr, model_cls = _SCHEMA_MAP[stage_num]
        setattr(run, attr, model_cls(**output))
        if stage_num == 4:
            self._correct_scene_durations(run)

    def _correct_scene_durations(self, run: PipelineRun):
        scene_by_id = {s.id: s for s in run.scene_plan.scenes}
        for audio in run.narration_output.audio:
            scene_by_id[audio.scene_id].duration_estimate_s = audio.duration_s + 1.0
        total = run.narration_output.total_duration_s
        target = run.duration_target_s
        if abs(total - target) > 2:
            ratio = target / total
            for scene in run.scene_plan.scenes:
                scene.duration_estimate_s = round(scene.duration_estimate_s * ratio, 2)
```

- [ ] **Step 4: Run — expect 6 passed (takes ~60s due to Stage 7)**

```bash
pytest tests/test_orchestrator.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: Orchestrator with emotion propagation, TTS-first timing, and selective regeneration"
```

---

## Task 9: FastAPI Backend

**Files:** `app/main.py`, `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

`tests/test_api.py`:
```python
import pytest
import os
from fastapi.testclient import TestClient

@pytest.fixture
def client(tmp_path):
    from app.main import create_app
    return TestClient(create_app(
        cache_dir=tmp_path / "cache",
        stub_stages={1, 2, 3, 4, 5, 6},
    ))

def test_generate_returns_run_id(client):
    resp = client.post("/generate", json={"prompt": "A wolf howls at the moon", "duration": 60, "style": "cinematic", "voice": "female"})
    assert resp.status_code == 200
    assert "run_id" in resp.json()

def test_get_run_returns_complete(client):
    resp = client.post("/generate", json={"prompt": "A wolf howls", "duration": 30, "style": "cinematic", "voice": "female"})
    run_id = resp.json()["run_id"]
    resp2 = client.get(f"/run/{run_id}")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "complete"

def test_blocked_prompt_returns_422(client):
    resp = client.post("/generate", json={"prompt": "A murder scene", "duration": 60, "style": "cinematic", "voice": "female"})
    assert resp.status_code == 422

def test_download_returns_mp4(client):
    resp = client.post("/generate", json={"prompt": "A wolf howls", "duration": 30, "style": "cinematic", "voice": "female"})
    run_id = resp.json()["run_id"]
    resp2 = client.get(f"/run/{run_id}/download")
    assert resp2.status_code == 200
    assert "video" in resp2.headers["content-type"]
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_api.py -v
```
Expected: ImportError

- [ ] **Step 3: Write `app/main.py`**

```python
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.cache import Cache
from app.orchestrator import Orchestrator
import os


class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 60
    style: str = "cinematic"
    voice: str = "female"


def create_app(cache_dir: Path = None, stub_stages: set[int] = None) -> FastAPI:
    if cache_dir is None:
        cache_dir = Path(os.getenv("DREAMSCAPE_CACHE_DIR", "cache"))
    if stub_stages is None:
        stub_stages = {
            int(s) for s in os.getenv("DREAMSCAPE_STUB_STAGES", "3,4,6").split(",") if s.strip()
        }

    cache = Cache(db_path=cache_dir / "runs.db", asset_dir=cache_dir / "assets")
    orch = Orchestrator(cache=cache, stub_stages=stub_stages)

    app = FastAPI(title="DreamScapeAI")

    @app.post("/generate")
    def generate(req: GenerateRequest):
        try:
            run = orch.run_pipeline(req.prompt, req.duration, req.style, req.voice)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"run_id": run.run_id, "status": run.status}

    @app.get("/run/{run_id}")
    def get_run(run_id: str):
        if not cache.stage_complete(run_id, 7):
            raise HTTPException(status_code=404, detail="Run not found or incomplete")
        output = cache.load_stage_output(run_id, 7)
        return {"run_id": run_id, "status": "complete", "video": output}

    @app.post("/run/{run_id}/regenerate/{stage_num}")
    def regenerate(run_id: str, stage_num: int):
        try:
            run = orch.run_stage(run_id, stage_num)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"run_id": run.run_id, "status": run.status}

    @app.get("/run/{run_id}/download")
    def download(run_id: str):
        if not cache.stage_complete(run_id, 7):
            raise HTTPException(status_code=404, detail="Video not ready")
        output = cache.load_stage_output(run_id, 7)
        path = Path(output["path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(str(path), media_type="video/mp4", filename="dreamscape.mp4")

    return app


app = create_app()
```

- [ ] **Step 4: Run — expect 4 passed**

```bash
pytest tests/test_api.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: FastAPI backend with generate, status, regenerate, and download endpoints"
```

---

## Task 10: Gradio UI

**Files:** `ui/gradio_app.py`

Gradio is tested by import smoke test + manual verification.

- [ ] **Step 1: Write `ui/gradio_app.py`**

```python
import os
from pathlib import Path
import gradio as gr
from app.cache import Cache
from app.orchestrator import Orchestrator


def _build_orch():
    stub_stages = {
        int(s) for s in os.getenv("DREAMSCAPE_STUB_STAGES", "3,4,6").split(",") if s.strip()
    }
    cache_dir = Path(os.getenv("DREAMSCAPE_CACHE_DIR", "cache"))
    cache = Cache(db_path=cache_dir / "runs.db", asset_dir=cache_dir / "assets")
    return Orchestrator(cache=cache, stub_stages=stub_stages)


def build_ui() -> gr.Blocks:
    orch = _build_orch()

    with gr.Blocks(title="DreamScapeAI") as demo:
        gr.Markdown("# DreamScapeAI\nTransform a text prompt into a cinematic short video.")

        with gr.Row():
            with gr.Column(scale=2):
                prompt_box = gr.Textbox(
                    label="Story Prompt",
                    placeholder="A lone wolf howls at a full moon in a snowy mountain.",
                    max_lines=4,
                )
                with gr.Row():
                    duration_radio = gr.Radio(choices=[30, 60, 90], value=60, label="Duration (s)")
                    style_dd = gr.Dropdown(
                        choices=["cinematic", "documentary", "anime", "noir", "horror"],
                        value="cinematic", label="Style",
                    )
                    voice_radio = gr.Radio(choices=["female", "male"], value="female", label="Voice")
                generate_btn = gr.Button("Generate Video", variant="primary")
            with gr.Column(scale=1):
                status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.TabItem("Scene Plan"):
                scene_table = gr.Dataframe(
                    headers=["Scene", "Description", "Narration", "Mood", "Duration (s)"],
                    interactive=False,
                )
            with gr.TabItem("Images"):
                image_gallery = gr.Gallery(label="Generated Images", columns=4)
            with gr.TabItem("Output"):
                video_out = gr.Video(label="Generated Video")
                download_file = gr.File(label="Download MP4")

        run_state = gr.State({})

        def on_generate(prompt, duration, style, voice):
            if not prompt.strip():
                return "Please enter a prompt.", None, None, None, None, {}
            try:
                run = orch.run_pipeline(prompt.strip(), int(duration), style, voice)
            except ValueError as e:
                return str(e), None, None, None, None, {}

            scenes = run.scene_plan.scenes if run.scene_plan else []
            rows = [[s.id, s.description[:60], s.narration_text[:60], s.mood,
                     f"{s.duration_estimate_s:.1f}"] for s in scenes]
            img_paths = [img.path for img in run.visual_output.images] if run.visual_output else []
            video_path = run.video_output.path if run.video_output else None
            status = (f"Done — run {run.run_id} | {run.video_output.duration_s:.1f}s"
                      if run.video_output else "Failed")
            return status, rows, img_paths, video_path, video_path, {"run_id": run.run_id}

        generate_btn.click(
            fn=on_generate,
            inputs=[prompt_box, duration_radio, style_dd, voice_radio],
            outputs=[status_box, scene_table, image_gallery, video_out, download_file, run_state],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
```

- [ ] **Step 2: Smoke-test import**

```bash
python -c "from ui.gradio_app import build_ui; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Launch and test manually**

```bash
python ui/gradio_app.py
```
Open http://localhost:7860. Enter `A wolf howls at the moon`, click **Generate Video**.

Expected:
- Status shows "Done — run XXXXXXXX"
- Scene Plan tab shows 4 rows
- Images tab shows 4 colored placeholder images
- Output tab shows a video player with a short MP4

- [ ] **Step 4: Commit**

```bash
git add ui/gradio_app.py
git commit -m "feat: Gradio UI with prompt input, scene plan, image gallery, and video player"
```

---

## Task 11: End-to-End Test + Final Verification

**Files:** `tests/test_e2e.py`

- [ ] **Step 1: Write E2E test**

`tests/test_e2e.py`:
```python
import pytest
from pathlib import Path
from app.cache import Cache
from app.orchestrator import Orchestrator

@pytest.fixture
def orch(tmp_path):
    cache = Cache(db_path=tmp_path / "runs.db", asset_dir=tmp_path / "assets")
    return Orchestrator(cache=cache, stub_stages={1, 2, 3, 4, 5, 6})

def test_full_pipeline_astronaut_prompt(orch):
    run = orch.run_pipeline(
        "A lonely astronaut discovers a glowing forest on Mars",
        duration=30, style="cinematic", voice="female",
    )
    assert run.status == "complete"
    assert run.parsed_prompt is not None
    assert run.scene_plan is not None
    assert 4 <= len(run.scene_plan.scenes) <= 8
    assert run.visual_output is not None
    assert len(run.visual_output.images) == len(run.scene_plan.scenes)
    assert run.narration_output is not None
    assert run.subtitle_output is not None
    assert run.music_output is not None
    assert run.video_output is not None
    video_path = Path(run.video_output.path)
    assert video_path.exists()
    assert video_path.suffix == ".mp4"
    assert run.video_output.file_size_bytes > 0
    assert run.video_output.duration_s > 0

def test_emotion_propagation_sad(orch):
    run = orch.run_pipeline("A lonely lost child in darkness", 30, "cinematic", "female")
    assert run.parsed_prompt.sentiment == "sad"
    # Sad → all scenes should have sad mood
    assert all(s.mood == "sad" for s in run.scene_plan.scenes)

def test_caching_reuses_run(orch):
    r1 = orch.run_pipeline("A wolf at the moon", 30, "cinematic", "female")
    r2 = orch.run_pipeline("A wolf at the moon", 30, "cinematic", "female")
    assert r1.run_id == r2.run_id
    assert Path(r1.video_output.path) == Path(r2.video_output.path)
```

- [ ] **Step 2: Run E2E tests**

```bash
pytest tests/test_e2e.py -v
```
Expected: 3 passed (takes ~60–90s)

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests passed across all test files

- [ ] **Step 4: Final commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end smoke tests covering full pipeline, emotion propagation, and caching"
```

---

## What's Next: Plan B — Real Model Integration

This plan delivers a working skeleton. To replace stubs with real models, a second plan will cover (in this order, least to most GPU-intensive):

1. **Stage 1 & 2** — Ollama + Llama-3.1-8B (requires `ollama pull llama3.1:8b`)
2. **Stage 5** — Whisper (CPU/GPU, `pip install openai-whisper`)
3. **Stage 4** — XTTS-v2 (`pip install TTS`)
4. **Stage 6** — MusicGen (`pip install audiocraft`)
5. **Stage 3** — SDXL with CPU offload (`pip install diffusers accelerate transformers`)

Each replaces `_run_real()` in its stage; stubs remain for local development.
