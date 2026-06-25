import threading
import wave
import numpy as np
from pathlib import Path
from app.stages.base import BaseStage
from app.config import MUSICGEN_MODEL

_SAMPLE_RATE = 44100
_MUSIC_MODEL = None  # module-level lazy singleton
_MUSIC_LOCK = threading.Lock()


def _get_model():
    global _MUSIC_MODEL
    if _MUSIC_MODEL is None:
        with _MUSIC_LOCK:
            if _MUSIC_MODEL is None:
                from audiocraft.models import MusicGen
                _MUSIC_MODEL = MusicGen.get_pretrained(MUSICGEN_MODEL)
    return _MUSIC_MODEL


def _tile_to_duration(path: Path, target_s: float):
    """Tile or trim a WAV file to exactly target_s seconds in-place."""
    if target_s <= 0:
        raise ValueError(f"target_s must be > 0, got {target_s}")
    with wave.open(str(path), "r") as f:
        rate = f.getframerate()
        channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        frames = f.readframes(f.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    data = np.frombuffer(frames, dtype=dtype).copy().reshape(-1, channels)
    if len(data) == 0:
        raise RuntimeError(f"WAV file at {path} contains no audio frames")
    target_frames = int(target_s * rate)
    if len(data) < target_frames:
        reps = int(np.ceil(target_frames / max(len(data), 1)))
        data = np.tile(data, (reps, 1))
    data = data[:target_frames]
    with wave.open(str(path), "w") as f:
        f.setnchannels(channels)
        f.setsampwidth(sampwidth)
        f.setframerate(rate)
        f.writeframes(data.tobytes())


def _generate_and_save(model, music_condition: str, duration_s: float, out_path: Path):
    """Generate audio with MusicGen and save to out_path (with .wav suffix)."""
    from audiocraft.data.audio import audio_write
    gen_duration = min(duration_s, 30.0)
    model.set_generation_params(duration=gen_duration)
    wav = model.generate([music_condition])  # shape [batch, channels, time]
    out_stem = str(out_path.with_suffix(""))  # audio_write appends .wav
    audio_write(out_stem, wav[0].cpu(), model.sample_rate, strategy="loudness")


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
        duration_s = float(input["total_duration_s"])
        music_condition = input["music_condition"]
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        out_path = asset_dir / "music.wav"

        model = _get_model()
        _generate_and_save(model, music_condition, duration_s, out_path)

        if not out_path.exists():
            raise RuntimeError(f"MusicGen did not produce output at {out_path}")

        _tile_to_duration(out_path, duration_s)

        return {"path": str(out_path), "duration_s": round(duration_s, 2)}

    def _run_stub(self, input: dict) -> dict:
        duration_s = input["total_duration_s"]
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        path = asset_dir / "music.wav"
        _silence_wav_stereo(path, duration_s)
        return {"path": str(path), "duration_s": round(duration_s, 2)}
