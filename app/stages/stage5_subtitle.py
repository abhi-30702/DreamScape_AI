from pathlib import Path
from app.stages.base import BaseStage

def _ts(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

class Stage5Subtitle(BaseStage):
    stage_num = 5

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("Whisper integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        audio_assets = input["audio_assets"]
        scenes_by_id = {s["id"]: s for s in input["scenes"]}
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        cursor = 0.0
        for audio in audio_assets:
            text = scenes_by_id[audio["scene_id"]]["narration_text"]
            end = cursor + audio["duration_s"]
            entries.append({"index": len(entries) + 1, "start_s": cursor, "end_s": end, "text": text})
            cursor = end

        srt_lines = []
        for e in entries:
            srt_lines += [str(e["index"]), f"{_ts(e['start_s'])} --> {_ts(e['end_s'])}", e["text"], ""]

        srt_path = asset_dir / "subtitles.srt"
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        return {"srt_path": str(srt_path), "entries": entries}
