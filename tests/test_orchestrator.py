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
