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
