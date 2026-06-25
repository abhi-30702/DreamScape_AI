import threading
import wave
from pathlib import Path
from app.stages.base import BaseStage

_SAMPLE_RATE = 22050
_WORDS_PER_SEC = 2.5

# XTTS-v2 built-in speaker names mapped from SpeakerSettings.speaker_id
_SPEAKER_MAP = {
    "female_en_1": "Claribel Dervla",
    "neutral_en_1": "Gracie Wise",
    "male_en_1": "Viktor Eka",
}

_TTS_MODEL = None  # module-level lazy singleton
_TTS_LOCK = threading.Lock()


def _get_tts():
    global _TTS_MODEL
    if _TTS_MODEL is None:
        with _TTS_LOCK:
            if _TTS_MODEL is None:
                from TTS.api import TTS
                _TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _TTS_MODEL


def _silence_wav(path: Path, duration_s: float):
    n = int(duration_s * _SAMPLE_RATE)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(_SAMPLE_RATE)
        f.writeframes(b"\x00\x00" * n)


def _measure_wav_duration(path: Path) -> float:
    with wave.open(str(path), "r") as f:
        return f.getnframes() / f.getframerate()


class Stage4Narrate(BaseStage):
    stage_num = 4

    def _run_real(self, input: dict) -> dict:
        scenes = input["scenes"]
        speaker_settings = input["speaker_settings"]
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)

        tts = _get_tts()
        speaker_name = _SPEAKER_MAP.get(speaker_settings["speaker_id"], "Claribel Dervla")

        audio_assets = []
        total = 0.0
        for scene in scenes:
            path = asset_dir / f"scene_{scene['id']}.wav"
            tts.tts_to_file(
                text=scene["narration_text"],
                speaker=speaker_name,
                language="en",
                file_path=str(path),
            )
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"TTS produced no output for scene {scene['id']}")
            duration = _measure_wav_duration(path)
            rounded_dur = round(duration, 2)
            audio_assets.append({
                "scene_id": scene["id"],
                "path": str(path),
                "duration_s": rounded_dur,
            })
            total += rounded_dur

        return {"audio": audio_assets, "total_duration_s": round(total, 2)}

    def _run_stub(self, input: dict) -> dict:
        scenes = input["scenes"]
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        audio_assets = []
        total = 0.0
        for scene in scenes:
            duration_s = max(2.0, len(scene["narration_text"].split()) / _WORDS_PER_SEC)
            path = asset_dir / f"scene_{scene['id']}.wav"
            _silence_wav(path, duration_s)
            audio_assets.append({"scene_id": scene["id"], "path": str(path), "duration_s": round(duration_s, 2)})
            total += duration_s
        return {"audio": audio_assets, "total_duration_s": round(total, 2)}
