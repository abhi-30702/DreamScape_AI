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
