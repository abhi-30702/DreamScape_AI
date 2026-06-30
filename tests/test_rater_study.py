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
