import wave, pytest
from pathlib import Path
from PIL import Image
from app.stages.stage7_assemble import Stage7Assemble

@pytest.fixture
def assets(tmp_path):
    imgs = []
    for i in range(2):
        img = Image.new("RGB", (1024, 576), (100 + i * 50, 80, 120))
        p = tmp_path / f"scene_{i}.png"
        img.save(str(p))
        imgs.append({"scene_id": i, "path": str(p), "width": 1024, "height": 576})

    audios = []
    for i in range(2):
        p = tmp_path / f"narr_{i}.wav"
        n = 22050 * 2
        with wave.open(str(p), "w") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
            f.writeframes(b"\x00\x00" * n)
        audios.append({"scene_id": i, "path": str(p), "duration_s": 2.0})

    mp = tmp_path / "music.wav"
    with wave.open(str(mp), "w") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(44100)
        f.writeframes(b"\x00\x00\x00\x00" * (44100 * 6))

    srt_p = tmp_path / "subtitles.srt"
    srt_p.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello world\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nGoodbye\n\n",
        encoding="utf-8"
    )

    return {
        "images": imgs,
        "narration": audios,
        "music": {"path": str(mp), "duration_s": 6.0},
        "subtitles": {"srt_path": str(srt_p), "entries": []},
        "output_path": str(tmp_path / "output.mp4"),
    }

def test_stage7_produces_mp4(tmp_path, assets):
    stage = Stage7Assemble(cache_dir=tmp_path / "cache", stub_stages=set())
    result = stage.run(assets)
    assert Path(result["path"]).exists()
    assert result["file_size_bytes"] > 0
    assert result["duration_s"] > 0
