import wave
from pathlib import Path
from app.stages.base import BaseStage

_SAMPLE_RATE = 44100

def _silence_wav_stereo(path: Path, duration_s: float):
    n = int(duration_s * _SAMPLE_RATE)
    with wave.open(str(path), "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(_SAMPLE_RATE)
        f.writeframes(b"\x00\x00\x00\x00" * n)

class Stage6Music(BaseStage):
    stage_num = 6

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("MusicGen integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        duration_s = input["total_duration_s"]
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        path = asset_dir / "music.wav"
        _silence_wav_stereo(path, duration_s)
        return {"path": str(path), "duration_s": round(duration_s, 2)}
