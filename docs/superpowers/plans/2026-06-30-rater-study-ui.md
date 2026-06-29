# Rater Study UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a "Rate videos" tab in the existing DreamScapeAI Gradio app where raters watch 20 pre-generated videos and rate each on six 5-point Likert dimensions, with responses pushed to a private HF Dataset repo as one JSON file per submission.

**Architecture:** New module `ui/rater_study.py` exports `build_rater_tab()` which constructs three logical screens (welcome → rating loop → thank-you) inside a single `gr.Blocks` group. A thin `ui/rater_storage.py` wraps `huggingface_hub.HfApi` for `list_completed` / `save_response` / `has_completed_overall`. `ui/gradio_app.py` is restructured to wrap its existing content in an outer `gr.Tabs` with two top-level tabs ("Generate" and "Rate videos"). Pure helper functions (`validate_start`, `compute_start_state`, `build_submission_payload`) are extracted at module level so the business logic is testable without a live Gradio server.

**Tech Stack:** Python 3.11, Gradio 4.30+, `huggingface_hub` 0.22+, pytest + `unittest.mock`, Git LFS for the bundled MP4s.

**Spec:** `docs/superpowers/specs/2026-06-30-rater-study-ui-design.md`

**Branch:** `feat/rater-study-ui` (already created from `main`).

---

## File Structure

```
ui/
├── gradio_app.py          # Modified (Task 6): wrap existing UI in outer Tabs
├── rater_study.py         # NEW (Tasks 3, 4, 5): helpers + handlers + build_rater_tab
└── rater_storage.py       # NEW (Task 2): HfApi wrapper

study_videos/
├── manifest.json          # NEW (Task 1): 20 video records
└── README.md              # NEW (Task 1): note that MP4s are added separately via Git LFS

tests/
├── test_manifest.py       # NEW (Task 1)
├── test_rater_storage.py  # NEW (Task 2)
└── test_rater_study.py    # NEW (Tasks 3, 4)

.gitattributes             # NEW (Task 1): Git LFS for study_videos/*.mp4
.env.example               # Modified (Task 7): add DREAMSCAPE_RATER_DATASET
README.md                  # Modified (Task 7): add Rater Study setup section
```

---

## Task 1: Manifest, Git LFS, and `test_manifest.py`

Sets up the source-of-truth manifest and the Git LFS rule for future MP4s.

**Files:**
- Create: `.gitattributes`
- Create: `study_videos/manifest.json`
- Create: `study_videos/README.md`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Create the failing test file**

Create `tests/test_manifest.py`:

```python
import json
import re
from pathlib import Path

import pytest

MANIFEST_PATH = Path("study_videos/manifest.json")
EXPECTED_VIDEO_COUNT = 20
ID_PATTERN = re.compile(r"^v\d{2}$")
REQUIRED_KEYS = {"id", "filename", "prompt", "sentiment", "style"}
VALID_SENTIMENTS = {"happy", "neutral", "sad"}
VALID_STYLES = {"cinematic", "documentary", "anime", "noir", "horror"}


def _load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_file_exists():
    assert MANIFEST_PATH.is_file(), f"{MANIFEST_PATH} must exist"


def test_manifest_parses_as_list():
    data = _load_manifest()
    assert isinstance(data, list)


def test_manifest_has_expected_count():
    data = _load_manifest()
    assert len(data) == EXPECTED_VIDEO_COUNT


def test_manifest_entries_have_required_keys():
    data = _load_manifest()
    for entry in data:
        missing = REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"Entry {entry.get('id')!r} missing keys: {missing}"


def test_manifest_ids_are_unique_and_match_pattern():
    data = _load_manifest()
    ids = [entry["id"] for entry in data]
    assert len(set(ids)) == len(ids), "manifest ids must be unique"
    for entry_id in ids:
        assert ID_PATTERN.match(entry_id), f"id {entry_id!r} does not match ^v\\d{{2}}$"


def test_manifest_sentiments_and_styles_are_valid():
    data = _load_manifest()
    for entry in data:
        assert entry["sentiment"] in VALID_SENTIMENTS, (
            f"{entry['id']}: sentiment {entry['sentiment']!r} not in {VALID_SENTIMENTS}"
        )
        assert entry["style"] in VALID_STYLES, (
            f"{entry['id']}: style {entry['style']!r} not in {VALID_STYLES}"
        )


def test_manifest_filenames_match_id():
    data = _load_manifest()
    for entry in data:
        assert entry["filename"] == f"{entry['id']}.mp4", (
            f"{entry['id']}: filename {entry['filename']!r} expected {entry['id']}.mp4"
        )


def test_manifest_referenced_files_exist_or_skip():
    data = _load_manifest()
    study_dir = MANIFEST_PATH.parent
    missing = [e["filename"] for e in data if not (study_dir / e["filename"]).is_file()]
    if missing:
        pytest.skip(f"{len(missing)} MP4(s) not yet generated: {missing[:3]}...")
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL with `FileNotFoundError` / `AssertionError` because `study_videos/manifest.json` does not exist yet.

- [ ] **Step 3: Create the manifest file**

Create `study_videos/manifest.json`:

```json
[
  {"id": "v01", "filename": "v01.mp4", "prompt": "A lone wolf howls at the moon over a snowy mountain peak.", "sentiment": "sad", "style": "cinematic", "run_id": null},
  {"id": "v02", "filename": "v02.mp4", "prompt": "Children fly colorful kites across a sunlit summer meadow.", "sentiment": "happy", "style": "cinematic", "run_id": null},
  {"id": "v03", "filename": "v03.mp4", "prompt": "A wide river flows through a dense forest at dawn.", "sentiment": "neutral", "style": "cinematic", "run_id": null},
  {"id": "v04", "filename": "v04.mp4", "prompt": "Empty city streets glisten after a heavy autumn rain.", "sentiment": "sad", "style": "documentary", "run_id": null},
  {"id": "v05", "filename": "v05.mp4", "prompt": "A baker shapes fresh dough as morning light fills the bakery.", "sentiment": "happy", "style": "documentary", "run_id": null},
  {"id": "v06", "filename": "v06.mp4", "prompt": "Time-lapse of clouds crossing a wide wheat field.", "sentiment": "neutral", "style": "documentary", "run_id": null},
  {"id": "v07", "filename": "v07.mp4", "prompt": "A girl waits alone at a train station as cherry blossoms fall.", "sentiment": "sad", "style": "anime", "run_id": null},
  {"id": "v08", "filename": "v08.mp4", "prompt": "Friends share ramen under bright festival lanterns at night.", "sentiment": "happy", "style": "anime", "run_id": null},
  {"id": "v09", "filename": "v09.mp4", "prompt": "A small robot tends a rooftop garden in a quiet city.", "sentiment": "neutral", "style": "anime", "run_id": null},
  {"id": "v10", "filename": "v10.mp4", "prompt": "A detective walks alone through a neon-lit alley in the rain.", "sentiment": "sad", "style": "noir", "run_id": null},
  {"id": "v11", "filename": "v11.mp4", "prompt": "Jazz musicians celebrate in a smoky club after a sold-out show.", "sentiment": "happy", "style": "noir", "run_id": null},
  {"id": "v12", "filename": "v12.mp4", "prompt": "A taxi drifts through empty downtown streets at three in the morning.", "sentiment": "neutral", "style": "noir", "run_id": null},
  {"id": "v13", "filename": "v13.mp4", "prompt": "An abandoned house stands silent in a moonlit graveyard.", "sentiment": "sad", "style": "horror", "run_id": null},
  {"id": "v14", "filename": "v14.mp4", "prompt": "A flickering candle in a vast empty hall casts long shadows.", "sentiment": "neutral", "style": "horror", "run_id": null},
  {"id": "v15", "filename": "v15.mp4", "prompt": "Fog rolls slowly across a still lake at midnight.", "sentiment": "neutral", "style": "horror", "run_id": null},
  {"id": "v16", "filename": "v16.mp4", "prompt": "A father reads his daughter's last letter under a streetlamp.", "sentiment": "sad", "style": "cinematic", "run_id": null},
  {"id": "v17", "filename": "v17.mp4", "prompt": "A couple dances on a sunlit beach at sunset.", "sentiment": "happy", "style": "cinematic", "run_id": null},
  {"id": "v18", "filename": "v18.mp4", "prompt": "A glacier calves into the arctic sea under a pale sky.", "sentiment": "neutral", "style": "documentary", "run_id": null},
  {"id": "v19", "filename": "v19.mp4", "prompt": "A small spirit fades into the rain after saying goodbye.", "sentiment": "sad", "style": "anime", "run_id": null},
  {"id": "v20", "filename": "v20.mp4", "prompt": "A trumpet player wins the city championship under the spotlight.", "sentiment": "happy", "style": "noir", "run_id": null}
]
```

- [ ] **Step 4: Create the LFS rule**

Create `.gitattributes`:

```
study_videos/*.mp4 filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 5: Create the study_videos README**

Create `study_videos/README.md`:

```markdown
# Study videos

This directory holds the 20 MP4s rated in the human evaluation study (Phase 2 of the Evaluation Framework).

- `manifest.json` is the source of truth for the videos in the study.
- MP4 files are tracked with Git LFS (see `.gitattributes`).
- MP4s are generated separately on a GPU runner and committed via `git lfs push`. Until that happens, the file-existence test in `tests/test_manifest.py` will skip.
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `pytest tests/test_manifest.py -v`
Expected: 7 passed, 1 skipped (the file-existence test skips because no MP4s exist yet).

- [ ] **Step 7: Commit**

```bash
git add .gitattributes study_videos/manifest.json study_videos/README.md tests/test_manifest.py
git commit -m "feat(eval): add rater study manifest, LFS rule, and manifest tests"
```

---

## Task 2: `ui/rater_storage.py` — HF Dataset I/O

Wraps `huggingface_hub.HfApi` so the UI layer has a clean three-method surface: `list_completed`, `save_response`, `has_completed_overall`.

**Files:**
- Create: `ui/rater_storage.py`
- Test: `tests/test_rater_storage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rater_storage.py`:

```python
import json
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import HfHubHTTPError

from ui import rater_storage


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_RATER_DATASET", "test-user/test-dataset")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")


def test_list_completed_returns_video_ids_for_rater():
    files = [
        "responses/reviewer_a/v01.json",
        "responses/reviewer_a/v02.json",
        "responses/reviewer_a/v07.json",
        "responses/reviewer_b/v01.json",
        "README.md",
    ]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        result = rater_storage.list_completed("reviewer_a")
    assert result == {"v01", "v02", "v07"}


def test_list_completed_excludes_overall_marker():
    files = [
        "responses/reviewer_a/v01.json",
        "responses/reviewer_a/_overall.json",
    ]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        result = rater_storage.list_completed("reviewer_a")
    assert result == {"v01"}
    assert "_overall" not in result


def test_list_completed_empty_for_new_rater():
    files = ["responses/reviewer_a/v01.json"]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        result = rater_storage.list_completed("brand_new_rater")
    assert result == set()


def test_has_completed_overall_true_when_marker_present():
    files = ["responses/reviewer_a/_overall.json"]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        assert rater_storage.has_completed_overall("reviewer_a") is True


def test_has_completed_overall_false_when_absent():
    files = ["responses/reviewer_a/v01.json"]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        assert rater_storage.has_completed_overall("reviewer_a") is False


def test_save_response_uploads_to_expected_path():
    payload = {"schema_version": 1, "rater_id": "reviewer_a", "video_id": "v07"}
    with patch("ui.rater_storage.HfApi") as MockApi:
        instance = MockApi.return_value
        rater_storage.save_response("reviewer_a", "v07", payload)
    instance.upload_file.assert_called_once()
    kwargs = instance.upload_file.call_args.kwargs
    assert kwargs["path_in_repo"] == "responses/reviewer_a/v07.json"
    assert kwargs["repo_id"] == "test-user/test-dataset"
    assert kwargs["repo_type"] == "dataset"
    uploaded_bytes = kwargs["path_or_fileobj"]
    assert json.loads(uploaded_bytes.decode("utf-8")) == payload


def test_save_response_works_for_overall_marker():
    payload = {"schema_version": 1, "rater_id": "reviewer_a", "overall_comment": "Nice."}
    with patch("ui.rater_storage.HfApi") as MockApi:
        instance = MockApi.return_value
        rater_storage.save_response("reviewer_a", "_overall", payload)
    kwargs = instance.upload_file.call_args.kwargs
    assert kwargs["path_in_repo"] == "responses/reviewer_a/_overall.json"


def test_save_response_raises_when_dataset_env_missing(monkeypatch):
    monkeypatch.delenv("DREAMSCAPE_RATER_DATASET", raising=False)
    with pytest.raises(RuntimeError, match="Rater study not configured"):
        rater_storage.save_response("reviewer_a", "v07", {})


def test_save_response_raises_when_token_env_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Rater study not configured"):
        rater_storage.save_response("reviewer_a", "v07", {})


def test_save_response_propagates_hfhub_http_error():
    with patch("ui.rater_storage.HfApi") as MockApi:
        response = MagicMock(status_code=500)
        MockApi.return_value.upload_file.side_effect = HfHubHTTPError(
            "boom", response=response
        )
        with pytest.raises(HfHubHTTPError):
            rater_storage.save_response("reviewer_a", "v07", {"x": 1})
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_rater_storage.py -v`
Expected: All tests fail with `ModuleNotFoundError: No module named 'ui.rater_storage'`.

- [ ] **Step 3: Implement the module**

Create `ui/rater_storage.py`:

```python
import json
import os

from huggingface_hub import HfApi


def _get_config() -> tuple[str, str]:
    repo_id = os.getenv("DREAMSCAPE_RATER_DATASET")
    token = os.getenv("HF_TOKEN")
    if not repo_id or not token:
        raise RuntimeError(
            "Rater study not configured: set DREAMSCAPE_RATER_DATASET and HF_TOKEN."
        )
    return repo_id, token


def _list_rater_files(rater_id: str) -> list[str]:
    repo_id, token = _get_config()
    api = HfApi(token=token)
    prefix = f"responses/{rater_id}/"
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [f for f in all_files if f.startswith(prefix)]


def list_completed(rater_id: str) -> set[str]:
    files = _list_rater_files(rater_id)
    completed: set[str] = set()
    for path in files:
        name = path.rsplit("/", 1)[-1]
        if not name.endswith(".json"):
            continue
        stem = name[: -len(".json")]
        if stem == "_overall":
            continue
        completed.add(stem)
    return completed


def has_completed_overall(rater_id: str) -> bool:
    files = _list_rater_files(rater_id)
    return any(f.endswith(f"/{rater_id}/_overall.json") for f in files)


def save_response(rater_id: str, video_id: str, payload: dict) -> None:
    repo_id, token = _get_config()
    api = HfApi(token=token)
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    api.upload_file(
        path_or_fileobj=body,
        path_in_repo=f"responses/{rater_id}/{video_id}.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"rater {rater_id} submission {video_id}",
    )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_rater_storage.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add ui/rater_storage.py tests/test_rater_storage.py
git commit -m "feat(eval): add rater_storage HfApi wrapper with list_completed and save_response"
```

---

## Task 3: `ui/rater_study.py` — Pure Helpers

Extracts the testable business logic (manifest loading, ID validation, resume index, payload builders) into module-level pure functions, no Gradio yet.

**Files:**
- Create: `ui/rater_study.py`
- Test: `tests/test_rater_study.py` (helpers section)

- [ ] **Step 1: Write the failing helper tests**

Create `tests/test_rater_study.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ui import rater_study


# ----- load_manifest -----

def test_load_manifest_parses_real_file():
    entries = rater_study.load_manifest(Path("study_videos/manifest.json"))
    assert len(entries) == 20
    assert entries[0]["id"] == "v01"


def test_load_manifest_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        rater_study.load_manifest(tmp_path / "nope.json")


# ----- validate_start -----

def test_validate_start_accepts_valid_input():
    result = rater_study.validate_start("reviewer_a", consent=True)
    assert result == {"ok": True, "error": None}


def test_validate_start_rejects_no_consent():
    result = rater_study.validate_start("reviewer_a", consent=False)
    assert result["ok"] is False
    assert "consent" in result["error"].lower()


def test_validate_start_rejects_short_id():
    result = rater_study.validate_start("ab", consent=True)
    assert result["ok"] is False
    assert "ID must be" in result["error"]


def test_validate_start_rejects_id_with_spaces():
    result = rater_study.validate_start("reviewer a", consent=True)
    assert result["ok"] is False


def test_validate_start_rejects_id_with_special_chars():
    result = rater_study.validate_start("rev!ewer", consent=True)
    assert result["ok"] is False


def test_validate_start_accepts_id_with_underscores_and_digits():
    result = rater_study.validate_start("reviewer_42", consent=True)
    assert result["ok"] is True


# ----- compute_start_state -----

def _fake_manifest(n=20):
    return [{"id": f"v{i:02d}", "filename": f"v{i:02d}.mp4", "prompt": "p", "sentiment": "neutral", "style": "cinematic", "run_id": None} for i in range(1, n + 1)]


def test_compute_start_state_new_rater_starts_at_zero():
    manifest = _fake_manifest()
    state = rater_study.compute_start_state("reviewer_a", set(), False, manifest)
    assert state["status"] == "rating"
    assert state["current_index"] == 0
    assert state["rater_id"] == "reviewer_a"
    assert state["total"] == 20


def test_compute_start_state_resumes_at_first_unrated():
    manifest = _fake_manifest()
    completed = {"v01", "v02", "v03"}
    state = rater_study.compute_start_state("reviewer_a", completed, False, manifest)
    assert state["status"] == "rating"
    assert state["current_index"] == 3  # next unrated, manifest is fixed order


def test_compute_start_state_skips_to_overall_when_all_videos_rated():
    manifest = _fake_manifest()
    completed = {f"v{i:02d}" for i in range(1, 21)}
    state = rater_study.compute_start_state("reviewer_a", completed, False, manifest)
    assert state["status"] == "overall_pending"
    assert state["current_index"] == 20


def test_compute_start_state_marks_all_done_when_overall_submitted():
    manifest = _fake_manifest()
    completed = {f"v{i:02d}" for i in range(1, 21)}
    state = rater_study.compute_start_state("reviewer_a", completed, True, manifest)
    assert state["status"] == "all_done"


def test_compute_start_state_resume_after_gap_finds_lowest_unrated():
    # Defensive: if files exist out of order (e.g. v01, v03 done but not v02),
    # resume at v02. Manifest order is the spine.
    manifest = _fake_manifest()
    completed = {"v01", "v03"}
    state = rater_study.compute_start_state("reviewer_a", completed, False, manifest)
    assert state["current_index"] == 1  # v02 is index 1


# ----- build_submission_payload -----

def test_build_submission_payload_shape():
    entry = {
        "id": "v07",
        "filename": "v07.mp4",
        "prompt": "A wolf howls.",
        "sentiment": "sad",
        "style": "cinematic",
        "run_id": "run_abc",
    }
    ratings = {
        "visual_quality": 4,
        "narration_clarity": 5,
        "music_mood_fit": 3,
        "av_sync": 4,
        "narrative_coherence": 4,
        "overall_quality": 4,
    }
    payload = rater_study.build_submission_payload(
        rater_id="reviewer_a",
        manifest_entry=entry,
        order_index=7,
        ratings=ratings,
        comment="Loud music",
    )
    assert payload["schema_version"] == 1
    assert payload["rater_id"] == "reviewer_a"
    assert payload["video_id"] == "v07"
    assert payload["video_filename"] == "v07.mp4"
    assert payload["video_prompt"] == "A wolf howls."
    assert payload["video_sentiment"] == "sad"
    assert payload["video_style"] == "cinematic"
    assert payload["video_run_id"] == "run_abc"
    assert payload["ratings"] == ratings
    assert payload["comment"] == "Loud music"
    assert payload["video_order_index"] == 7
    assert payload["app_version"] == rater_study.APP_VERSION
    # submitted_at_utc is ISO-8601 with Z suffix
    datetime.fromisoformat(payload["submitted_at_utc"].replace("Z", "+00:00"))


def test_build_overall_payload_shape():
    payload = rater_study.build_overall_payload(rater_id="reviewer_a", comment="Solid run.")
    assert payload["schema_version"] == 1
    assert payload["rater_id"] == "reviewer_a"
    assert payload["overall_comment"] == "Solid run."
    assert payload["app_version"] == rater_study.APP_VERSION
    datetime.fromisoformat(payload["submitted_at_utc"].replace("Z", "+00:00"))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_rater_study.py -v`
Expected: All fail with `ModuleNotFoundError: No module named 'ui.rater_study'`.

- [ ] **Step 3: Implement the helpers**

Create `ui/rater_study.py`:

```python
import json
import re
from datetime import datetime, timezone
from pathlib import Path

APP_VERSION = "0.2.0"

DIMENSIONS = [
    ("visual_quality", "Visual quality", "How sharp / coherent the imagery is"),
    ("narration_clarity", "Narration clarity", "How clearly the voice can be understood"),
    ("music_mood_fit", "Music–mood fit", "How well the music matches the story's tone"),
    ("av_sync", "A/V sync", "How well visuals, narration, and subtitles line up"),
    ("narrative_coherence", "Narrative coherence", "Whether the story makes sense from start to finish"),
    ("overall_quality", "Overall quality", "Your holistic impression of the video"),
]

_ID_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_start(rater_id: str, consent: bool) -> dict:
    if not consent:
        return {"ok": False, "error": "You must give consent to start."}
    if not _ID_RE.match(rater_id or ""):
        return {
            "ok": False,
            "error": "ID must be 3–32 characters, letters/numbers/underscore only.",
        }
    return {"ok": True, "error": None}


def compute_start_state(
    rater_id: str,
    completed_ids: set[str],
    has_overall: bool,
    manifest: list[dict],
) -> dict:
    total = len(manifest)
    if has_overall:
        return {
            "rater_id": rater_id,
            "current_index": total,
            "total": total,
            "status": "all_done",
        }
    # First index in manifest order whose id is not in completed_ids.
    next_index = total
    for i, entry in enumerate(manifest):
        if entry["id"] not in completed_ids:
            next_index = i
            break
    status = "overall_pending" if next_index == total else "rating"
    return {
        "rater_id": rater_id,
        "current_index": next_index,
        "total": total,
        "status": status,
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_submission_payload(
    rater_id: str,
    manifest_entry: dict,
    order_index: int,
    ratings: dict,
    comment: str,
) -> dict:
    return {
        "schema_version": 1,
        "rater_id": rater_id,
        "video_id": manifest_entry["id"],
        "video_filename": manifest_entry["filename"],
        "video_prompt": manifest_entry["prompt"],
        "video_sentiment": manifest_entry["sentiment"],
        "video_style": manifest_entry["style"],
        "video_run_id": manifest_entry.get("run_id"),
        "ratings": dict(ratings),
        "comment": comment or "",
        "video_order_index": order_index,
        "submitted_at_utc": _now_utc_iso(),
        "app_version": APP_VERSION,
    }


def build_overall_payload(rater_id: str, comment: str) -> dict:
    return {
        "schema_version": 1,
        "rater_id": rater_id,
        "overall_comment": comment or "",
        "submitted_at_utc": _now_utc_iso(),
        "app_version": APP_VERSION,
    }
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_rater_study.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add ui/rater_study.py tests/test_rater_study.py
git commit -m "feat(eval): add rater_study pure helpers (validation, state, payload)"
```

---

## Task 4: `ui/rater_study.py` — Module-Level Handlers

Adds three orchestrating handlers — `on_start`, `on_submit`, `on_overall_submit` — that call `rater_storage` and produce return tuples. Defined at module level so tests can call them directly without a live Gradio server.

**Files:**
- Modify: `ui/rater_study.py` (append handlers)
- Test: `tests/test_rater_study.py` (append handler tests)

- [ ] **Step 1: Append failing handler tests**

Append to `tests/test_rater_study.py`:

```python
# ----- on_start -----

def _fake_manifest_20():
    return _fake_manifest(20)


def test_on_start_with_valid_inputs_returns_rating_state(monkeypatch):
    monkeypatch.setattr(
        "ui.rater_storage.list_completed", lambda rid: set()
    )
    monkeypatch.setattr(
        "ui.rater_storage.has_completed_overall", lambda rid: False
    )
    result = rater_study.on_start("reviewer_a", True, _fake_manifest_20())
    assert result["ok"] is True
    assert result["state"]["status"] == "rating"
    assert result["state"]["current_index"] == 0
    assert result["error"] is None


def test_on_start_resumes_at_first_unrated(monkeypatch):
    monkeypatch.setattr(
        "ui.rater_storage.list_completed", lambda rid: {"v01", "v02"}
    )
    monkeypatch.setattr(
        "ui.rater_storage.has_completed_overall", lambda rid: False
    )
    result = rater_study.on_start("reviewer_a", True, _fake_manifest_20())
    assert result["state"]["current_index"] == 2
    assert result["state"]["status"] == "rating"


def test_on_start_with_already_completed_rater_jumps_to_done(monkeypatch):
    monkeypatch.setattr(
        "ui.rater_storage.list_completed",
        lambda rid: {f"v{i:02d}" for i in range(1, 21)},
    )
    monkeypatch.setattr(
        "ui.rater_storage.has_completed_overall", lambda rid: True
    )
    result = rater_study.on_start("reviewer_a", True, _fake_manifest_20())
    assert result["state"]["status"] == "all_done"


def test_on_start_with_invalid_id_returns_error(monkeypatch):
    # Even if storage would say "new rater", we never call it for invalid input.
    called = {"n": 0}
    def _list(rid):
        called["n"] += 1
        return set()
    monkeypatch.setattr("ui.rater_storage.list_completed", _list)
    result = rater_study.on_start("a!", True, _fake_manifest_20())
    assert result["ok"] is False
    assert result["error"] is not None
    assert result["state"] is None
    assert called["n"] == 0


def test_on_start_with_no_consent_returns_error(monkeypatch):
    called = {"n": 0}
    def _list(rid):
        called["n"] += 1
        return set()
    monkeypatch.setattr("ui.rater_storage.list_completed", _list)
    result = rater_study.on_start("reviewer_a", False, _fake_manifest_20())
    assert result["ok"] is False
    assert "consent" in result["error"].lower()
    assert called["n"] == 0


# ----- on_submit -----

def test_on_submit_saves_and_advances_state(monkeypatch):
    saved = []
    monkeypatch.setattr(
        "ui.rater_storage.save_response",
        lambda rid, vid, payload: saved.append((rid, vid, payload)),
    )
    manifest = _fake_manifest_20()
    state = {"rater_id": "reviewer_a", "current_index": 0, "total": 20, "status": "rating"}
    ratings = {k: 4 for k, _, _ in rater_study.DIMENSIONS}
    result = rater_study.on_submit(state, ratings, "Nice", manifest)
    assert result["ok"] is True
    assert result["state"]["current_index"] == 1
    assert result["state"]["status"] == "rating"
    assert len(saved) == 1
    rid, vid, payload = saved[0]
    assert rid == "reviewer_a"
    assert vid == "v01"
    assert payload["ratings"] == ratings
    assert payload["comment"] == "Nice"


def test_on_submit_on_last_video_transitions_to_overall_pending(monkeypatch):
    monkeypatch.setattr("ui.rater_storage.save_response", lambda *a, **k: None)
    manifest = _fake_manifest_20()
    state = {"rater_id": "reviewer_a", "current_index": 19, "total": 20, "status": "rating"}
    ratings = {k: 3 for k, _, _ in rater_study.DIMENSIONS}
    result = rater_study.on_submit(state, ratings, "", manifest)
    assert result["state"]["status"] == "overall_pending"
    assert result["state"]["current_index"] == 20


def test_on_submit_when_storage_raises_does_not_advance(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("ui.rater_storage.save_response", _raise)
    manifest = _fake_manifest_20()
    state = {"rater_id": "reviewer_a", "current_index": 5, "total": 20, "status": "rating"}
    ratings = {k: 4 for k, _, _ in rater_study.DIMENSIONS}
    result = rater_study.on_submit(state, ratings, "", manifest)
    assert result["ok"] is False
    assert result["state"]["current_index"] == 5  # unchanged
    assert "could not save" in result["error"].lower()


def test_on_submit_validates_all_ratings_present(monkeypatch):
    called = {"n": 0}
    def _save(*a, **k):
        called["n"] += 1
    monkeypatch.setattr("ui.rater_storage.save_response", _save)
    manifest = _fake_manifest_20()
    state = {"rater_id": "reviewer_a", "current_index": 0, "total": 20, "status": "rating"}
    ratings = {k: 4 for k, _, _ in rater_study.DIMENSIONS}
    ratings["visual_quality"] = None  # one missing
    result = rater_study.on_submit(state, ratings, "", manifest)
    assert result["ok"] is False
    assert called["n"] == 0


# ----- on_overall_submit -----

def test_on_overall_submit_saves_and_marks_done(monkeypatch):
    saved = []
    monkeypatch.setattr(
        "ui.rater_storage.save_response",
        lambda rid, vid, payload: saved.append((rid, vid, payload)),
    )
    state = {"rater_id": "reviewer_a", "current_index": 20, "total": 20, "status": "overall_pending"}
    result = rater_study.on_overall_submit(state, "Great experience.")
    assert result["ok"] is True
    assert result["state"]["status"] == "all_done"
    assert len(saved) == 1
    rid, vid, payload = saved[0]
    assert rid == "reviewer_a"
    assert vid == "_overall"
    assert payload["overall_comment"] == "Great experience."


def test_on_overall_submit_when_storage_raises_keeps_pending(monkeypatch):
    monkeypatch.setattr(
        "ui.rater_storage.save_response",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    state = {"rater_id": "reviewer_a", "current_index": 20, "total": 20, "status": "overall_pending"}
    result = rater_study.on_overall_submit(state, "")
    assert result["ok"] is False
    assert result["state"]["status"] == "overall_pending"
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `pytest tests/test_rater_study.py -v`
Expected: 15 prior tests still pass; new handler tests fail with `AttributeError: module 'ui.rater_study' has no attribute 'on_start'`.

- [ ] **Step 3: Append handler implementations**

Append to `ui/rater_study.py`:

```python
from ui import rater_storage


def on_start(rater_id: str, consent: bool, manifest: list[dict]) -> dict:
    """Handle the Start click. Returns {ok, error, state}."""
    check = validate_start(rater_id, consent)
    if not check["ok"]:
        return {"ok": False, "error": check["error"], "state": None}
    try:
        completed = rater_storage.list_completed(rater_id)
        has_overall = rater_storage.has_completed_overall(rater_id)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not load your progress: {exc}",
            "state": None,
        }
    state = compute_start_state(rater_id, completed, has_overall, manifest)
    return {"ok": True, "error": None, "state": state}


def _all_ratings_present(ratings: dict) -> bool:
    keys = {k for k, _, _ in DIMENSIONS}
    return all(ratings.get(k) is not None for k in keys)


def on_submit(state: dict, ratings: dict, comment: str, manifest: list[dict]) -> dict:
    """Handle the Submit & next click. Returns {ok, error, state}."""
    if not _all_ratings_present(ratings):
        return {
            "ok": False,
            "error": "Please rate every dimension before submitting.",
            "state": state,
        }
    idx = state["current_index"]
    entry = manifest[idx]
    payload = build_submission_payload(
        rater_id=state["rater_id"],
        manifest_entry=entry,
        order_index=idx,
        ratings=ratings,
        comment=comment,
    )
    try:
        rater_storage.save_response(state["rater_id"], entry["id"], payload)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not save your rating — please try again. ({exc})",
            "state": state,  # unchanged
        }
    next_idx = idx + 1
    next_status = "overall_pending" if next_idx >= state["total"] else "rating"
    new_state = {**state, "current_index": next_idx, "status": next_status}
    return {"ok": True, "error": None, "state": new_state}


def on_overall_submit(state: dict, overall_comment: str) -> dict:
    """Handle the final overall-comment submit. Returns {ok, error, state}."""
    payload = build_overall_payload(state["rater_id"], overall_comment)
    try:
        rater_storage.save_response(state["rater_id"], "_overall", payload)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not save your final comment — please try again. ({exc})",
            "state": state,
        }
    new_state = {**state, "status": "all_done"}
    return {"ok": True, "error": None, "state": new_state}
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `pytest tests/test_rater_study.py -v`
Expected: 25 passed (15 helpers + 10 handlers).

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `pytest -q`
Expected: All prior tests still pass; new tests pass.

- [ ] **Step 6: Commit**

```bash
git add ui/rater_study.py tests/test_rater_study.py
git commit -m "feat(eval): add rater_study orchestrating handlers (on_start, on_submit, on_overall_submit)"
```

---

## Task 5: `build_rater_tab()` — Gradio UI

Wires the helpers and handlers into a single `gr.Blocks` group with three screens (welcome / rating loop / thank-you). No new unit tests — manual smoke verification only.

**Files:**
- Modify: `ui/rater_study.py` (append `build_rater_tab`)

- [ ] **Step 1: Append `build_rater_tab` to `ui/rater_study.py`**

Append:

```python
from pathlib import Path

import gradio as gr

_INFO_SHEET = """\
### About this study

You will watch up to 20 short AI-generated videos (each 30–90 seconds) and rate
each on six dimensions. The full session takes about 20 minutes. Your ratings
are stored in a private dataset for academic analysis. You may stop at any time
by simply closing the browser tab — any ratings you already submitted are kept.

By starting, you confirm you are at least 18 years old and consent to participate.

Questions or concerns: contact the researcher.
"""

_STUDY_VIDEOS_DIR = Path("study_videos")


def build_rater_tab() -> gr.Group:
    """Construct the Rate-videos tab content. Mount inside a `gr.TabItem`."""
    manifest = load_manifest(_STUDY_VIDEOS_DIR / "manifest.json")

    with gr.Group() as tab_root:
        # ---- Screen A: welcome / consent ----
        with gr.Group(visible=True) as screen_welcome:
            gr.Markdown(_INFO_SHEET)
            consent_chk = gr.Checkbox(label="I am 18+ and consent to participate", value=False)
            rater_id_tb = gr.Textbox(
                label="Choose an anonymous ID",
                placeholder="letters, numbers, underscore (e.g. reviewer_a)",
                max_lines=1,
            )
            welcome_error = gr.Markdown(visible=False)
            start_btn = gr.Button("Start", variant="primary", interactive=False)

        # ---- Screen B: rating loop ----
        with gr.Group(visible=False) as screen_rating:
            progress_md = gr.Markdown("Video 1 of 20")
            video_player = gr.Video(autoplay=False, show_label=False)
            gr.Markdown("Rate this video on each dimension (1 = Poor, 5 = Excellent):")
            radios: list[gr.Radio] = []
            for key, label, hint in DIMENSIONS:
                r = gr.Radio(choices=[1, 2, 3, 4, 5], label=label, info=hint)
                radios.append(r)
            comment_tb = gr.Textbox(
                label="Optional: anything specific you noticed?",
                lines=2,
            )
            rating_error = gr.Markdown(visible=False)
            submit_btn = gr.Button("Submit & next", variant="primary", interactive=False)

        # ---- Screen C: thank-you / overall ----
        with gr.Group(visible=False) as screen_thanks:
            thanks_md = gr.Markdown("### Thank you! Your ratings have been saved.")
            overall_tb = gr.Textbox(label="Any overall feedback?", lines=4)
            overall_error = gr.Markdown(visible=False)
            overall_btn = gr.Button("Submit final comment", variant="primary")
            done_md = gr.Markdown("Done — you can close this tab.", visible=False)

        state = gr.State({"status": "welcome"})

        # ---- Start-button enable logic ----
        def _toggle_start(consent, rid):
            ok = bool(consent) and bool(_ID_RE.match(rid or ""))
            return gr.update(interactive=ok)

        consent_chk.change(_toggle_start, [consent_chk, rater_id_tb], start_btn)
        rater_id_tb.change(_toggle_start, [consent_chk, rater_id_tb], start_btn)

        # ---- Submit-button enable logic ----
        def _toggle_submit(*values):
            ok = all(v is not None for v in values)
            return gr.update(interactive=ok)

        for r in radios:
            r.change(_toggle_submit, radios, submit_btn)

        # ---- Start click ----
        def _start_click(rater_id, consent):
            result = on_start(rater_id, consent, manifest)
            if not result["ok"]:
                return (
                    gr.update(visible=True, value=f"**{result['error']}**"),  # welcome_error
                    gr.update(),  # screen_welcome
                    gr.update(),  # screen_rating
                    gr.update(),  # screen_thanks
                    gr.update(),  # video_player
                    gr.update(),  # progress_md
                    result.get("state") or {"status": "welcome"},  # state
                )
            new_state = result["state"]
            if new_state["status"] == "all_done":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(),
                    new_state,
                )
            if new_state["status"] == "overall_pending":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(),
                    new_state,
                )
            # status == "rating"
            idx = new_state["current_index"]
            entry = manifest[idx]
            video_path = str(_STUDY_VIDEOS_DIR / entry["filename"])
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=video_path),
                gr.update(value=f"Video {idx + 1} of {new_state['total']}"),
                new_state,
            )

        start_btn.click(
            _start_click,
            [rater_id_tb, consent_chk],
            [welcome_error, screen_welcome, screen_rating, screen_thanks,
             video_player, progress_md, state],
        )

        # ---- Submit click ----
        def _submit_click(state_val, comment_val, *rating_values):
            ratings = {k: v for (k, _, _), v in zip(DIMENSIONS, rating_values)}
            result = on_submit(state_val, ratings, comment_val, manifest)
            if not result["ok"]:
                return (
                    gr.update(visible=True, value=f"**{result['error']}**"),  # rating_error
                    gr.update(),  # screen_rating
                    gr.update(),  # screen_thanks
                    gr.update(),  # video_player
                    gr.update(),  # progress_md
                    gr.update(),  # comment_tb
                    *[gr.update() for _ in DIMENSIONS],
                    gr.update(interactive=False),  # submit_btn
                    result["state"],
                )
            new_state = result["state"]
            if new_state["status"] == "overall_pending":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(),
                    gr.update(value=""),
                    *[gr.update(value=None) for _ in DIMENSIONS],
                    gr.update(interactive=False),
                    new_state,
                )
            # next video
            idx = new_state["current_index"]
            entry = manifest[idx]
            video_path = str(_STUDY_VIDEOS_DIR / entry["filename"])
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=video_path),
                gr.update(value=f"Video {idx + 1} of {new_state['total']}"),
                gr.update(value=""),
                *[gr.update(value=None) for _ in DIMENSIONS],
                gr.update(interactive=False),
                new_state,
            )

        submit_btn.click(
            _submit_click,
            [state, comment_tb, *radios],
            [rating_error, screen_rating, screen_thanks, video_player, progress_md,
             comment_tb, *radios, submit_btn, state],
        )

        # ---- Overall submit click ----
        def _overall_click(state_val, overall_val):
            result = on_overall_submit(state_val, overall_val)
            if not result["ok"]:
                return (
                    gr.update(visible=True, value=f"**{result['error']}**"),  # overall_error
                    gr.update(),  # done_md
                    gr.update(),  # overall_btn
                    result["state"],
                )
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(interactive=False),
                result["state"],
            )

        overall_btn.click(
            _overall_click,
            [state, overall_tb],
            [overall_error, done_md, overall_btn, state],
        )

    return tab_root
```

- [ ] **Step 2: Manual smoke check — module imports cleanly**

Run: `python -c "from ui.rater_study import build_rater_tab; print('ok')"`
Expected: prints `ok` with no traceback.

- [ ] **Step 3: Run the full test suite to ensure no regressions**

Run: `pytest -q`
Expected: all previously passing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add ui/rater_study.py
git commit -m "feat(eval): build_rater_tab Gradio UI with welcome/rating/thanks screens"
```

---

## Task 6: Mount the Rater Tab in `gradio_app.py`

Wraps the existing demo content in an outer `gr.Tabs` with two top-level tabs: "Generate" (current UI) and "Rate videos" (new tab).

**Files:**
- Modify: `ui/gradio_app.py`

- [ ] **Step 1: Read the current `build_ui()` body so the rewrite is exact**

Already shown in plan header. The existing function defines a `gr.Blocks` with a `gr.Markdown` header, a `gr.Row` containing prompt+status, an inner `gr.Tabs` (Scene Plan / Images / Output), a state, and a click handler.

- [ ] **Step 2: Replace `build_ui()` with the wrapped version**

Replace the body of `ui/gradio_app.py` (lines 17–78) so the function reads:

```python
def build_ui() -> gr.Blocks:
    from ui.rater_study import build_rater_tab

    orch = _build_orch()

    with gr.Blocks(title="DreamScapeAI") as demo:
        gr.Markdown("# DreamScapeAI\nTransform a text prompt into a cinematic short video.")

        with gr.Tabs():
            with gr.TabItem("Generate"):
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

            with gr.TabItem("Rate videos"):
                build_rater_tab()

    return demo
```

The `from ui.rater_study import build_rater_tab` is intentionally inside `build_ui()` to keep `gradio_app.py` importable in environments where the manifest file is missing (it's imported lazily at UI build time, not at module import time).

- [ ] **Step 3: Manual smoke check — module imports cleanly**

Run: `python -c "from ui.gradio_app import build_ui; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Manual smoke check — UI launches**

Run: `python -m ui.gradio_app` (Ctrl-C to stop after confirming the local URL is served and both tabs render).
Expected: console shows `Running on local URL: http://127.0.0.1:7860`. Browser shows the DreamScapeAI header and two top tabs ("Generate", "Rate videos"). The Generate tab still renders the original layout. The Rate videos tab shows the info sheet, consent checkbox, ID textbox, and a disabled Start button.

- [ ] **Step 5: Run the full test suite to ensure no regressions**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add ui/gradio_app.py
git commit -m "feat(eval): wrap demo in outer Tabs and mount rater study tab"
```

---

## Task 7: `.env.example` and `README.md` Updates

Documents the new env var and the one-time HF Dataset setup so a future operator (or supervisor reviewer) can stand the study up from scratch.

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Add the env var to `.env.example`**

Append to `.env.example`:

```
# Rater study (Evaluation Framework Phase 2)
# Private HF Dataset repo where per-submission JSON files are written.
# HF_TOKEN above must have WRITE scope on this repo.
DREAMSCAPE_RATER_DATASET=your-username/dreamscape-rater-study
```

- [ ] **Step 2: Add a Rater Study section to `README.md`**

Append the following section to `README.md`:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add .env.example README.md
git commit -m "docs(eval): document rater study env var and one-time HF Dataset setup"
```

---

## Final Checks

- [ ] **Run full test suite:** `pytest -q` — all tests pass (including 7 manifest tests with 1 skipped, 10 storage tests, 25 study tests, plus all prior tests).
- [ ] **Smoke-launch the app:** `python -m ui.gradio_app` — both tabs render; the rater tab shows the info sheet.
- [ ] **Inspect the diff:** `git log --oneline main..HEAD` — should show 7 focused commits, one per task.

Once green, hand off to `superpowers:finishing-a-development-branch`.

---

## Notes for the Implementer

- The branch `feat/rater-study-ui` is already created. Do **not** commit on `main`.
- Do not generate any MP4s — those come from a separate GPU-run task. Until then, `study_videos/` will only contain `manifest.json` and `README.md`, and the file-existence manifest test will skip.
- Treat `ui/rater_study.py` as a single growing file across Tasks 3 → 4 → 5. The split into tasks is for review granularity; the final file has helpers + handlers + `build_rater_tab` together.
- `validate_start`, `compute_start_state`, `build_submission_payload`, and `build_overall_payload` are intentionally module-level pure functions. Do not collapse them into closures inside `build_rater_tab` — that breaks the unit tests.
- All tests use `unittest.mock.patch` and `monkeypatch.setattr` to mock the storage layer. No live HF API calls in tests.
- **Deferred hardening (not in this plan):** the spec mentions a "3 consecutive failures → `/tmp/rater_fallback.jsonl`" best-effort backup path. This is intentionally not implemented here; the single-attempt failure path (don't advance, keep ratings, show warning) already satisfies the primary error-handling requirement. Add the retry + fallback in a separate small PR if a live study run shows it's needed.
