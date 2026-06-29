# Rater Study UI — Design Spec

**Date:** 2026-06-30
**Phase:** Evaluation Framework Phase 2 (human study)
**Status:** Approved by user, ready for plan

## Goal

Add a "Rate videos" tab to the existing DreamScapeAI Gradio app so 20+ raters can each watch 20 pre-generated videos and rate each on six 5-point Likert dimensions. Responses are persisted to a private Hugging Face Dataset repo so the data survives Space restarts and can be analyzed locally in Phase 3.

This is the data-collection arm of the thesis's 4th research contribution ("Human-Centered Evaluation Framework"). Phase 3 will compute MOS, Fleiss' kappa, and produce thesis-ready tables from the data this UI collects.

## Constraints

- Same Hugging Face Space as the demo. No separate deployment.
- Free-tier HF Spaces — filesystem is ephemeral, so all durable data must live in a Dataset repo.
- N = 20 videos × 20 raters = 400 rows per dimension. Tiny scale; correctness matters more than throughput.
- Final-year dissertation — needs at least a minimal info-sheet + consent step for academic ethics.

## Architecture & Integration

The existing `ui/gradio_app.py` `build_ui()` is wrapped in a `gr.Tabs` container with two tabs:
1. **Generate** — the existing demo, unchanged.
2. **Rate videos** — the new study tab, mounted via `build_rater_tab()` from a new `ui/rater_study.py`.

The rater tab is a single `gr.Blocks` with three logical screens (Welcome / Rating loop / Thank-you), swapped by toggling `visible` on `gr.Group`s.

Per-session state is held in a `gr.State` dict: `{rater_id, current_index, completed_ids, manifest}`. State is reconstructed on resume by listing existing response files for that rater_id.

### Module boundaries

```
ui/
├── gradio_app.py          # Modified — wraps existing UI in Tabs, mounts rater tab
├── rater_study.py         # NEW — build_rater_tab() + handler functions
└── rater_storage.py       # NEW — HfApi wrapper: list_completed(), save_response()

study_videos/
├── manifest.json          # NEW — 20 video records
└── v01.mp4 ... v20.mp4    # NEW — via Git LFS

tests/
├── test_manifest.py       # NEW
├── test_rater_storage.py  # NEW
└── test_rater_study.py    # NEW

.env.example               # Modified — DREAMSCAPE_RATER_DATASET, note HF_TOKEN write scope
README.md                  # Modified — Rater Study section
.gitattributes             # Modified — Git LFS for study_videos/*.mp4
```

- `rater_storage.py` is the only module that talks to HF Hub. Trivial to mock; trivial to swap for local-disk during dev (env var unset → write to `./local_responses/`).
- `rater_study.py` knows Gradio + business logic only. Imports `rater_storage` and nothing else from the project.
- `study_videos/manifest.json` is the single source of truth for "what the 20 videos are."

## User Flow

### Screen A — Welcome / consent
- Title + 2–3 paragraph info sheet (purpose, ~20 min, data storage, right to stop, contact email).
- `gr.Checkbox("I am 18+ and consent to participate")`.
- `gr.Textbox("Choose an anonymous ID (letters/numbers/underscore, e.g. 'reviewer_a')")`.
- `gr.Button("Start")` — disabled until checkbox is ticked **and** ID matches `^[A-Za-z0-9_]{3,32}$`.

On click: validate, call `list_completed(rater_id)`, compute resume point, swap to Screen B. If all 20 + `_overall` already done → skip to a "you've already finished" branch of Screen C.

### Screen B — Rating loop
For each of 20 videos in fixed manifest order:
- Progress label: `"Video 7 of 20"`
- `gr.Video(value=<path>, autoplay=False, show_label=False)`
- 6 `gr.Radio(choices=[1,2,3,4,5], label="<Dimension>", info="<hint>")` rows.
- `gr.Textbox(label="Optional: anything specific you noticed?", lines=2)`.
- `gr.Button("Submit & next")` — `interactive=False` until all 6 radios have a value (recomputed via `.change()` on each radio).

On submit: assemble payload, call `save_response(rater_id, video_id, payload)`, advance `current_index`, reset radios + comment to defaults via `gr.update`, render next video. On the 20th submit, hide Screen B, show Screen C.

### Screen C — Thank-you
- "Thank you! Your ratings have been saved."
- `gr.Textbox("Any overall feedback?", lines=4)`.
- `gr.Button("Submit final comment")` → `save_response(rater_id, "_overall", {...})`, then shows "Done — you can close this tab."

## Rating Screen Layout

```
Video 7 of 20
[ gr.Video player ]

Rate this video on each dimension (1 = Poor, 5 = Excellent):

Visual quality          (1)  (2)  (3)  (4)  (5)
 How sharp / coherent the imagery is

Narration clarity       (1)  (2)  (3)  (4)  (5)
 How clearly the voice can be understood

Music–mood fit          (1)  (2)  (3)  (4)  (5)
 How well the music matches the story's tone

A/V sync                (1)  (2)  (3)  (4)  (5)
 How well visuals, narration, and subtitles line up

Narrative coherence     (1)  (2)  (3)  (4)  (5)
 Whether the story makes sense from start to finish

Overall quality         (1)  (2)  (3)  (4)  (5)
 Your holistic impression of the video

Optional: anything specific you noticed?
[ textarea ]

                                       [ Submit & next ]
```

The 6 dimension labels and hints used for `gr.Radio(label=..., info=...)`:

| Dimension key (JSON) | Label | Hint |
|---|---|---|
| `visual_quality` | Visual quality | How sharp / coherent the imagery is |
| `narration_clarity` | Narration clarity | How clearly the voice can be understood |
| `music_mood_fit` | Music–mood fit | How well the music matches the story's tone |
| `av_sync` | A/V sync | How well visuals, narration, and subtitles line up |
| `narrative_coherence` | Narrative coherence | Whether the story makes sense from start to finish |
| `overall_quality` | Overall quality | Your holistic impression of the video |

## Data Model

### Per-submission JSON (`responses/<rater_id>/<video_id>.json`)

```json
{
  "schema_version": 1,
  "rater_id": "reviewer_a",
  "video_id": "v07",
  "video_filename": "v07.mp4",
  "video_prompt": "A lonely wolf howls at the moon...",
  "video_sentiment": "sad",
  "video_style": "cinematic",
  "video_run_id": "run_abc123",
  "ratings": {
    "visual_quality": 4,
    "narration_clarity": 5,
    "music_mood_fit": 3,
    "av_sync": 4,
    "narrative_coherence": 4,
    "overall_quality": 4
  },
  "comment": "Music was a bit loud during the narration.",
  "video_order_index": 7,
  "submitted_at_utc": "2026-07-15T14:23:11Z",
  "app_version": "0.2.0"
}
```

### Overall final-comment JSON (`responses/<rater_id>/_overall.json`)

```json
{
  "schema_version": 1,
  "rater_id": "reviewer_a",
  "overall_comment": "Visuals stronger than narration overall.",
  "submitted_at_utc": "2026-07-15T14:45:02Z",
  "app_version": "0.2.0"
}
```

### Manifest (`study_videos/manifest.json`)

```json
[
  {
    "id": "v01",
    "filename": "v01.mp4",
    "prompt": "A wolf howls at the moon...",
    "sentiment": "sad",
    "style": "cinematic",
    "run_id": "run_abc123"
  }
]
```

Twenty entries total. `id` matches `^v\d{2}$`. `filename` is just the basename — joined with `study_videos/` at load time. `run_id` is optional and points back to the cache entry that produced the video.

### Design rationale

- **One JSON file per submission** — no read-modify-write, so concurrent raters cannot clobber each other. `list_completed(rater_id)` is just `HfApi.list_repo_files(...)` filtered by prefix. Phase 3 analysis globs all files into a pandas DataFrame in ~3 lines.
- **Denormalized prompt / sentiment / style into each row** — if the manifest evolves (typos fixed, videos re-rendered), each row still carries the exact state at rating time. No joins needed to reproduce thesis results.
- **`schema_version` and `app_version`** — future-proofing for a mid-study schema tweak; analysis script can branch on `schema_version`.

## Storage Layer (`ui/rater_storage.py`)

Public API:

```python
def list_completed(rater_id: str) -> set[str]:
    """Return the set of video_ids this rater has already submitted.

    Excludes the special '_overall' key. Returns empty set if rater is new
    or if no responses exist yet.
    """

def save_response(rater_id: str, video_id: str, payload: dict) -> None:
    """Upload payload as JSON to responses/<rater_id>/<video_id>.json.

    video_id may be a regular id like 'v07' or the special string '_overall'.
    Overwrites if the file already exists (intentional, see 'Concurrent raters'
    in Error Handling).
    """

def has_completed_overall(rater_id: str) -> bool:
    """True iff responses/<rater_id>/_overall.json exists."""
```

Reads `DREAMSCAPE_RATER_DATASET` (e.g. `"abhi-30702/dreamscape-rater-study"`) and `HF_TOKEN` from env at call time. Both required — raises `RuntimeError("Rater study not configured")` if either is missing.

Implementation uses `huggingface_hub.HfApi`:
- `list_repo_files(repo_id=..., repo_type="dataset")` for `list_completed` / `has_completed_overall`.
- `upload_file(path_or_fileobj=<bytes>, path_in_repo=..., repo_id=..., repo_type="dataset", commit_message=...)` for `save_response`.

`HfHubHTTPError` is re-raised so the UI layer can show a `gr.Warning`.

## Error Handling

| Scenario | Behavior |
|---|---|
| Invalid rater ID format | Inline error: "ID must be 3–32 characters, letters/numbers/underscore only." Start stays disabled. |
| Consent unchecked | Start stays disabled. No error needed; visual state is self-explanatory. |
| Already-completed rater enters their ID | `list_completed` returns 20 + `has_completed_overall` is True → skip to a "you've already finished — thank you" Screen C branch. No re-rating exposed. |
| Resume mid-study | `list_completed` returns e.g. {v01, v02, v03} → `current_index = 3` (next unrated, manifest is fixed order). Re-rating is not exposed. |
| Storage upload fails | Submit handler catches, does **not** advance index, shows `gr.Warning("Could not save your rating — please try again.")`. Radios stay populated. After 3 consecutive failures on the same submit click, also append a JSON line to `/tmp/rater_fallback.jsonl` inside the Space (best-effort). |
| Missing video file on disk | Caught by `test_manifest.py`. At runtime, Screen A shows "Study not ready — contact the researcher" and Start stays disabled. |
| Missing env vars (`HF_TOKEN` / `DREAMSCAPE_RATER_DATASET`) | Rater tab renders a hard error "Rater study not configured." Generate tab is unaffected. |
| Concurrent raters with same ID | Last write wins on the per-`<video_id>.json` file. Acceptable; info sheet says "don't share your ID." |
| Browser can't decode the video | `gr.Video` falls back to a download link automatically. No special handling. |

## Testing

### `tests/test_manifest.py` (3 tests)
- Manifest parses as JSON with expected 6 keys per entry.
- Exactly 20 entries with unique `id`s matching `^v\d{2}$`.
- Every `filename` references a file that exists in `study_videos/` *(skipped if directory is empty — videos are generated separately from app development)*.

### `tests/test_rater_storage.py` (6 tests, `HfApi` mocked)
- `list_completed` returns the set of `video_id`s for a given rater_id.
- `list_completed` excludes `_overall` from its result set.
- `save_response` calls `upload_file` with `path_in_repo="responses/<rater_id>/<video_id>.json"` and JSON-encoded bytes.
- `save_response` raises `RuntimeError("Rater study not configured")` when either env var is missing.
- `save_response` re-raises `HfHubHTTPError` for the UI layer to catch.
- `save_response` overwrites existing files (last-write-wins is intentional).

### `tests/test_rater_study.py` (8 tests, no live Gradio server — handlers called directly)
- Start handler with valid ID + consent → initial state with `current_index=0`.
- Start handler with already-completed rater → state points at the thank-you screen.
- Start handler with invalid ID format → validation error, no state change.
- Start handler with consent unchecked → validation error.
- Submit handler with all 6 ratings → calls `save_response` once, advances `current_index`, resets radios.
- Submit handler on the 20th video → transitions to Screen C.
- Submit handler when storage raises → does not advance index, surfaces warning.
- Overall-comment submit → calls `save_response(rater_id, "_overall", ...)` and shows done state.

### Not automated (manual smoke checks before launching study)
- Gradio rendering on the deployed Space (welcome → loop → finish on Chrome and Firefox).
- Real HF Hub upload round-trip with a throwaway rater ID.
- Video playback in at least 2 browsers.

## Environment Variables

- `HF_TOKEN` — must have **write** scope on the rater dataset repo (the existing token used for the LLM backend has read scope; this study needs an upgraded scope or a second token).
- `DREAMSCAPE_RATER_DATASET` — repo id of the private dataset, e.g. `"abhi-30702/dreamscape-rater-study"`.

Both documented in `.env.example` and the README's "Rater Study setup" section.

## Out of Scope

- Counterbalancing video order across raters (decided: not worth the complexity at N=20).
- Demographics collection (decided: keeps consent friction low).
- Re-rating already-rated videos (decided: walk-forward only, no edit).
- External survey handoff (decided: keeps raters in one place).
- Real-time aggregation dashboard (Phase 3 will analyze locally from the dumped JSON).
- Email reminders / invitations (out-of-band; will recruit via the supervisor's channels).

## Deferred Decisions for Plan / Implementation

- Exact wording of the info sheet — will be drafted during implementation and reviewed by the supervisor before launch.
- Exact contents of the 20-video manifest — generated separately (Phase 3 prerequisite: generate 20 videos on T4 covering the sentiment × style matrix).
- One-time HF Dataset repo setup (create private repo at huggingface.co, add HF_TOKEN with write scope to the Space secrets) — operational, not code.
