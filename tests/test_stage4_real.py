import wave
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import app.stages.stage4_narrate as _s4_mod
from app.stages.stage4_narrate import Stage4Narrate


def _make_mock_tts():
    """Returns a mock TTS object whose tts_to_file writes a real WAV."""
    def tts_to_file(text, speaker, language, file_path):
        n = int(max(2.0, len(text.split()) / 2.5) * 22050)
        with wave.open(file_path, "w") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
            f.writeframes(b"\x00\x00" * n)

    mock_tts = MagicMock()
    mock_tts.tts_to_file.side_effect = tts_to_file
    return mock_tts


def test_stage4_real_creates_wav_per_scene(tmp_path):
    _s4_mod._TTS_MODEL = None
    mock_tts = _make_mock_tts()

    with patch("app.stages.stage4_narrate._get_tts", return_value=mock_tts):
        stage = Stage4Narrate(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "scenes": [
                {"id": 0, "narration_text": "A wolf howls at the winter moon above the frozen peaks."},
                {"id": 1, "narration_text": "Silence falls across the snow."},
            ],
            "speaker_settings": {"speaker_id": "female_en_1", "pitch_semitones": 0.0, "speed": 1.0},
            "asset_dir": str(tmp_path / "audio"),
        })

    assert len(result["audio"]) == 2
    assert result["total_duration_s"] > 0
    import pytest
    assert result["total_duration_s"] == pytest.approx(
        sum(audio["duration_s"] for audio in result["audio"]), abs=0.05
    )
    for audio in result["audio"]:
        assert Path(audio["path"]).exists()
        assert audio["duration_s"] > 0


def test_stage4_real_maps_speaker_id_to_xtts_name(tmp_path):
    _s4_mod._TTS_MODEL = None
    mock_tts = _make_mock_tts()

    with patch("app.stages.stage4_narrate._get_tts", return_value=mock_tts):
        stage = Stage4Narrate(cache_dir=tmp_path, stub_stages=set())
        stage.run({
            "scenes": [{"id": 0, "narration_text": "Hello world."}],
            "speaker_settings": {"speaker_id": "male_en_1", "pitch_semitones": -1.0, "speed": 1.05},
            "asset_dir": str(tmp_path / "audio"),
        })

    mock_tts.tts_to_file.assert_called_once_with(
        text="Hello world.",
        speaker="Viktor Eka",
        language="en",
        file_path=str(tmp_path / "audio" / "scene_0.wav"),
    )


def test_stage4_real_falls_back_speaker_on_unknown_id(tmp_path):
    _s4_mod._TTS_MODEL = None
    mock_tts = _make_mock_tts()

    with patch("app.stages.stage4_narrate._get_tts", return_value=mock_tts):
        stage = Stage4Narrate(cache_dir=tmp_path, stub_stages=set())
        stage.run({
            "scenes": [{"id": 0, "narration_text": "Test."}],
            "speaker_settings": {"speaker_id": "unknown_speaker", "pitch_semitones": 0.0, "speed": 1.0},
            "asset_dir": str(tmp_path / "audio"),
        })

    _, kwargs = mock_tts.tts_to_file.call_args
    assert kwargs["speaker"] == "Claribel Dervla"  # fallback


def test_stage4_real_measures_actual_duration(tmp_path):
    _s4_mod._TTS_MODEL = None

    def tts_to_file(text, speaker, language, file_path):
        # Write exactly 3 seconds of audio
        n = 22050 * 3
        with wave.open(file_path, "w") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
            f.writeframes(b"\x00\x00" * n)

    mock_tts = MagicMock()
    mock_tts.tts_to_file.side_effect = tts_to_file

    with patch("app.stages.stage4_narrate._get_tts", return_value=mock_tts):
        stage = Stage4Narrate(cache_dir=tmp_path, stub_stages=set())
        result = stage.run({
            "scenes": [{"id": 0, "narration_text": "A sentence."}],
            "speaker_settings": {"speaker_id": "female_en_1", "pitch_semitones": 0.0, "speed": 1.0},
            "asset_dir": str(tmp_path / "audio"),
        })

    assert abs(result["audio"][0]["duration_s"] - 3.0) < 0.05
    assert abs(result["total_duration_s"] - 3.0) < 0.05
