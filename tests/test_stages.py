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
