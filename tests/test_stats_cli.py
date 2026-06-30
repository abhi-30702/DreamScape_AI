import json
import subprocess
import sys
from pathlib import Path

import pytest


DIMS = [
    "visual_quality", "narration_clarity", "music_mood_fit",
    "av_sync", "narrative_coherence", "overall_quality",
]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def study_dirs(tmp_path):
    responses = tmp_path / "responses_dump" / "responses"
    manifest = tmp_path / "manifest.json"
    eval_runs = tmp_path / "eval_results"
    out = tmp_path / "thesis_tables"

    manifest_data = [
        {"id": f"v{i:02d}", "filename": f"v{i:02d}.mp4", "prompt": "p",
         "sentiment": ["sad", "happy", "neutral"][i % 3],
         "style": ["cinematic", "anime", "noir"][i % 3],
         "run_id": f"run_{i:02d}"}
        for i in range(1, 6)
    ]
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    for r in ["ra", "rb", "rc"]:
        for entry in manifest_data:
            ratings = {dim: 4 for dim in DIMS}
            _write(responses / r / f"{entry['id']}.json", {
                "schema_version": 1,
                "rater_id": r,
                "video_id": entry["id"],
                "video_filename": entry["filename"],
                "video_prompt": entry["prompt"],
                "video_sentiment": entry["sentiment"],
                "video_style": entry["style"],
                "video_run_id": entry["run_id"],
                "ratings": ratings,
                "comment": "",
                "video_order_index": 0,
                "submitted_at_utc": "2026-07-01T00:00:00Z",
                "app_version": "0.2.0",
            })

    for entry in manifest_data:
        _write(eval_runs / f"{entry['run_id']}.json", {
            "run_id": entry["run_id"],
            "clip_score": {"per_scene": [0.3], "mean": 0.3 + 0.01 * int(entry["id"][1:])},
            "wer": {"per_scene": [0.1], "mean": 0.1},
            "sync_error_ms": {"per_entry": [80.0], "mean": 80.0, "max": 80.0, "pass": True},
        })

    return {
        "responses": tmp_path / "responses_dump",
        "manifest": manifest,
        "eval_runs": eval_runs,
        "out": out,
    }


def test_cli_creates_all_six_table_pairs(study_dirs):
    result = subprocess.run(
        [sys.executable, "-m", "eval.stats",
         "--responses", str(study_dirs["responses"]),
         "--manifest", str(study_dirs["manifest"]),
         "--metrics", str(study_dirs["eval_runs"]),
         "--out", str(study_dirs["out"])],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    expected_stems = [
        "summary", "mos_per_video", "mos_by_sentiment", "mos_by_style",
        "kappa_per_dimension", "auto_vs_human_correlations",
    ]
    for stem in expected_stems:
        assert (study_dirs["out"] / f"{stem}.csv").is_file(), f"missing {stem}.csv"
        assert (study_dirs["out"] / f"{stem}.md").is_file(), f"missing {stem}.md"
    assert (study_dirs["out"] / "provenance.txt").is_file()


def test_cli_skips_correlations_when_no_metrics(study_dirs):
    out_dir = study_dirs["out"]
    result = subprocess.run(
        [sys.executable, "-m", "eval.stats",
         "--responses", str(study_dirs["responses"]),
         "--manifest", str(study_dirs["manifest"]),
         "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "mos_per_video.csv").is_file()
    assert (out_dir / "kappa_per_dimension.csv").is_file()
    assert not (out_dir / "auto_vs_human_correlations.csv").is_file()
    assert "skipping correlations" in (result.stdout + result.stderr).lower()


def test_cli_fails_when_responses_dir_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "eval.stats",
         "--responses", str(tmp_path / "does_not_exist"),
         "--manifest", str(tmp_path / "manifest.json"),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
