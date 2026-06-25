import wave
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import app.stages.stage6_music as _s6_mod
from app.stages.stage6_music import Stage6Music, _tile_to_duration


def _write_silence_wav(path: Path, duration_s: float, rate=32000, channels=1):
    n = int(duration_s * rate)
    with wave.open(str(path), "w") as f:
        f.setnchannels(channels); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(b"\x00\x00" * n * channels)


def test_stage6_real_creates_wav(tmp_path):
    _s6_mod._MUSIC_MODEL = None
    asset_dir = tmp_path / "music"
    asset_dir.mkdir()

    mock_model = MagicMock()
    mock_model.sample_rate = 32000

    def fake_generate_and_save(model, condition, duration_s, out_path):
        _write_silence_wav(out_path, min(duration_s, 30.0), rate=32000)

    with patch("app.stages.stage6_music._get_model", return_value=mock_model), \
         patch("app.stages.stage6_music._generate_and_save", side_effect=fake_generate_and_save):
        stage = Stage6Music(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "mood": "sad",
            "total_duration_s": 10.0,
            "music_condition": "melancholic minor key slow tempo sparse",
            "asset_dir": str(asset_dir),
        })

    assert Path(result["path"]).exists()
    assert result["duration_s"] == 10.0


def test_stage6_real_tiles_audio_for_long_videos(tmp_path):
    _s6_mod._MUSIC_MODEL = None
    asset_dir = tmp_path / "music"
    asset_dir.mkdir()

    mock_model = MagicMock()

    def fake_generate_and_save(model, condition, duration_s, out_path):
        # Write only 5 seconds — stage should tile to 20s
        _write_silence_wav(out_path, 5.0, rate=32000)

    with patch("app.stages.stage6_music._get_model", return_value=mock_model), \
         patch("app.stages.stage6_music._generate_and_save", side_effect=fake_generate_and_save):
        stage = Stage6Music(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "mood": "happy",
            "total_duration_s": 20.0,
            "music_condition": "uplifting major key fast tempo",
            "asset_dir": str(asset_dir),
        })

    assert result["duration_s"] == 20.0
    # Verify the WAV is actually ~20 seconds
    with wave.open(result["path"], "r") as f:
        actual_s = f.getnframes() / f.getframerate()
    assert abs(actual_s - 20.0) < 0.1


def test_tile_to_duration_extends_short_audio(tmp_path):
    p = tmp_path / "test.wav"
    _write_silence_wav(p, 3.0, rate=44100, channels=2)

    _tile_to_duration(p, 10.0)

    with wave.open(str(p), "r") as f:
        actual_s = f.getnframes() / f.getframerate()
    assert abs(actual_s - 10.0) < 0.05


def test_tile_to_duration_trims_long_audio(tmp_path):
    p = tmp_path / "test.wav"
    _write_silence_wav(p, 15.0, rate=44100, channels=1)

    _tile_to_duration(p, 10.0)

    with wave.open(str(p), "r") as f:
        actual_s = f.getnframes() / f.getframerate()
    assert abs(actual_s - 10.0) < 0.05
