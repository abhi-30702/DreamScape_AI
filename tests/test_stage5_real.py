import wave
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import app.stages.stage5_subtitle as _s5_mod
from app.stages.stage5_subtitle import Stage5Subtitle


@pytest.fixture
def silence_wav(tmp_path):
    p = tmp_path / "narr_0.wav"
    n = 22050 * 2
    with wave.open(str(p), "w") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
        f.writeframes(b"\x00\x00" * n)
    return str(p)


def test_stage5_real_transcribes_each_scene(tmp_path, silence_wav):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Hello world", "segments": []}
    _s5_mod._MODEL = None

    with patch("app.stages.stage5_subtitle._get_model", return_value=mock_model):
        stage = Stage5Subtitle(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "audio_assets": [{"scene_id": 0, "path": silence_wav, "duration_s": 2.0}],
            "scenes": [{"id": 0, "narration_text": "Hello world"}],
            "asset_dir": str(tmp_path / "out"),
        })

    assert Path(result["srt_path"]).exists()
    assert len(result["entries"]) == 1
    assert result["entries"][0]["text"] == "Hello world"
    assert result["entries"][0]["start_s"] == 0.0
    assert result["entries"][0]["end_s"] == 2.0


def test_stage5_real_cumulative_timestamps(tmp_path, silence_wav):
    wav2 = str(tmp_path / "narr_1.wav")
    n = 22050 * 3
    with wave.open(wav2, "w") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
        f.writeframes(b"\x00\x00" * n)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Scene narration.", "segments": []}
    _s5_mod._MODEL = None

    with patch("app.stages.stage5_subtitle._get_model", return_value=mock_model):
        stage = Stage5Subtitle(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "audio_assets": [
                {"scene_id": 0, "path": silence_wav, "duration_s": 2.0},
                {"scene_id": 1, "path": wav2, "duration_s": 3.0},
            ],
            "scenes": [],
            "asset_dir": str(tmp_path / "out"),
        })

    assert result["entries"][0]["start_s"] == 0.0
    assert result["entries"][0]["end_s"] == 2.0
    assert result["entries"][1]["start_s"] == 2.0
    assert result["entries"][1]["end_s"] == 5.0


def test_stage5_real_writes_valid_srt(tmp_path, silence_wav):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "The wolf howls.", "segments": []}
    _s5_mod._MODEL = None

    with patch("app.stages.stage5_subtitle._get_model", return_value=mock_model):
        stage = Stage5Subtitle(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "audio_assets": [{"scene_id": 0, "path": silence_wav, "duration_s": 2.0}],
            "scenes": [],
            "asset_dir": str(tmp_path / "out"),
        })

    srt_content = Path(result["srt_path"]).read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,000" in srt_content
    assert "The wolf howls." in srt_content
    assert srt_content.count(" --> ") == len(result["entries"])
