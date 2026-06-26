# Automated Metrics Runner — Design Spec

**Date:** 2026-06-26
**Subsystem:** Evaluation Framework — Phase 1 (Automated Metrics)

---

## Goal

A CLI script that loads any completed DreamScapeAI pipeline run from the existing SQLite cache, computes three automated quality metrics, and writes a JSON result file. Designed for local thesis data collection — run once per generated video, batch over all completed runs for the full study dataset.

---

## Architecture

Three new files. No existing files modified.

```
eval/
├── __init__.py          # empty package marker
└── metrics.py           # all metric logic + CLI entry point

eval_results/            # created at runtime, gitignored
└── <run_id>.json        # one result file per evaluated run

tests/
└── test_metrics.py      # 8 unit tests (mocked models, no downloads)
```

**New dependencies** added to `requirements.txt`:
- `open-clip-torch>=2.20.0` — CLIP-ViT-B/32 embeddings for CLIPScore
- `jiwer>=3.0.0` — WER computation

Whisper and MoviePy are already in `requirements.txt` (used by Stages 5 and 7).

---

## CLI Interface

```bash
# Single run
python eval/metrics.py --run-id <run_id>

# Batch over all complete runs in cache
python eval/metrics.py --batch

# Custom output path
python eval/metrics.py --run-id <run_id> --out results/my_run.json
```

Default output: `eval_results/<run_id>.json` (directory created if absent).

---

## Metric Definitions

### 1. CLIPScore

**What it measures:** Image-text alignment — does each generated image match its narration text?

**Computation:**
1. Load CLIP-ViT-B/32 via `open_clip` (downloaded once, cached by `open_clip`)
2. For each scene `i`:
   - Encode `visual_output.images[i].path` as CLIP image embedding
   - Encode `scene_plan.scenes[i].narration_text` as CLIP text embedding
   - Compute cosine similarity → `per_scene[i]`
3. Average across all scenes → `mean`

**Target:** mean ≥ 0.70

**Inputs:** `visual_output.images[].path`, `scene_plan.scenes[].narration_text`

---

### 2. WER (Word Error Rate)

**What it measures:** Subtitle accuracy — how well does Whisper's transcription match the original narration script?

**Computation:**
1. Compute scene time windows: scene `i` starts at `T_i = sum(narration_output.audio[0..i-1].duration_s)` and ends at `T_i + audio[i].duration_s`
2. For each scene `i`, collect subtitle entries whose `start_s` falls within `[T_i, T_i + audio[i].duration_s)`
3. Join entry texts → hypothesis string
4. Use `scene_plan.scenes[i].narration_text` → reference string
5. `jiwer.wer(reference, hypothesis)` → `per_scene[i]`
6. Average across scenes → `mean`

**Target:** as low as possible (no hard threshold; reported for thesis)

**Inputs:** `subtitle_output.entries`, `scene_plan.scenes[].narration_text`, `narration_output.audio[].duration_s` (for scene time windows)

---

### 3. Sync Error

**What it measures:** Audio-visual timing drift — do subtitle timestamps in the assembled video match the actual narration audio?

**Computation:**
1. Extract audio from `video_output.path` using MoviePy → temp WAV
2. Re-run Whisper on the extracted audio → re-transcription with word timestamps
3. Align SRT entries to re-transcription entries by position index (entry `i` maps to re-transcription word group `i`); unmatched tail entries are skipped
4. `error_ms = |re_transcription_start_s - srt_entry.start_s| * 1000`
5. Report `per_entry` list, `mean`, `max`, and `pass` (True if all entries < 200ms)

**Target:** all entries < 200ms (`pass: true`)

**Inputs:** `video_output.path`, `subtitle_output.entries`

---

## Data Flow

```
run_id
  │
  ▼
cache.get_run(run_id)               → PipelineRun
  │
  ├─► images[], narration_texts[]   → compute_clip_score()   → {per_scene, mean}
  ├─► srt_entries, scene_texts[]    → compute_wer()          → {per_scene, mean}
  └─► video_path, srt_entries[]     → compute_sync_error()   → {per_entry, mean, max, pass}
  │
  ▼
eval_results/<run_id>.json
```

---

## Output Format

```json
{
  "run_id": "abc123",
  "computed_at": "2026-06-26T12:00:00Z",
  "prompt": "A warrior stands tall at dawn",
  "duration_target_s": 60,
  "clip_score": {
    "per_scene": [0.82, 0.79, 0.85, 0.78],
    "mean": 0.81
  },
  "wer": {
    "per_scene": [0.05, 0.08, 0.03, 0.06],
    "mean": 0.055
  },
  "sync_error_ms": {
    "per_entry": [45, 88, 120, 55, 98],
    "mean": 81.2,
    "max": 120.0,
    "pass": true
  }
}
```

If a metric cannot be computed (e.g. `video_output` is `None` — stub run), that metric's value is `null` in the JSON. The script never raises on incomplete runs.

---

## Module Interface

Each metric function takes plain types — no `PipelineRun` dependency, fully unit-testable:

```python
def compute_clip_score(image_paths: list[str], texts: list[str]) -> dict:
    """Returns {"per_scene": [...], "mean": float}"""

def compute_wer(hypotheses: list[str], references: list[str]) -> dict:
    """Returns {"per_scene": [...], "mean": float}"""

def compute_sync_error(video_path: str, srt_entries: list[SubtitleEntry]) -> dict:
    """Returns {"per_entry": [...], "mean": float, "max": float, "pass": bool}"""
```

---

## Testing

`tests/test_metrics.py` — 8 unit tests, all models mocked:

| Test | Verifies |
|---|---|
| `test_clip_score_returns_per_scene_and_mean` | mocked CLIP embeddings → cosine sim computed correctly |
| `test_clip_score_single_scene` | edge case: one scene |
| `test_wer_perfect_match` | hypothesis == reference → 0.0 |
| `test_wer_partial_match` | known substitution → expected WER value |
| `test_sync_error_within_threshold` | mocked Whisper timestamps within 200ms → `pass: true` |
| `test_sync_error_exceeds_threshold` | mocked Whisper timestamp >200ms → `pass: false` |
| `test_batch_mode_skips_incomplete_runs` | `video_output=None` → sync error is `null`, no crash |
| `test_json_output_written_to_eval_results` | full run → JSON written to correct path |

---

## Non-Goals

- No rater study UI (Phase 2)
- No statistical analysis or Fleiss' kappa (Phase 3)
- No HF Spaces integration — runs locally only
- No per-word CLIPScore (scene-level only)
- No real-time metrics during pipeline execution
