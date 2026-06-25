import wave
from pathlib import Path
from app.stages.base import BaseStage

_SAMPLE_RATE = 22050
_WORDS_PER_SEC = 2.5

def _silence_wav(path: Path, duration_s: float):
    n = int(duration_s * _SAMPLE_RATE)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(_SAMPLE_RATE)
        f.writeframes(b"\x00\x00" * n)

class Stage4Narrate(BaseStage):
    stage_num = 4

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("XTTS-v2 integration — Plan B")

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
