import os
import subprocess
import tempfile
from pathlib import Path
from app.stages.base import BaseStage


class Stage7Assemble(BaseStage):
    stage_num = 7

    def _run_real(self, input: dict) -> dict:
        return self._assemble(input)

    def _run_stub(self, input: dict) -> dict:
        return self._assemble(input)

    def _assemble(self, input: dict) -> dict:
        import numpy as np
        from PIL import Image as PILImage
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip

        images = sorted(input["images"], key=lambda x: x["scene_id"])
        narration = sorted(input["narration"], key=lambda x: x["scene_id"])
        music_data = input["music"]
        srt_path = input["subtitles"]["srt_path"]
        output_path = Path(input["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        TARGET_W, TARGET_H = 1920, 1080

        clips = []
        for img_data, audio_data in zip(images, narration):
            narr_clip = AudioFileClip(audio_data["path"])
            # Pre-resize with Pillow to avoid MoviePy's ANTIALIAS incompatibility with Pillow 10+
            pil_img = PILImage.open(img_data["path"]).convert("RGB")
            pil_img = pil_img.resize((TARGET_W, TARGET_H), PILImage.LANCZOS)
            frame = np.array(pil_img)
            img_clip = (
                ImageClip(frame)
                .set_duration(audio_data["duration_s"])
                .set_audio(narr_clip)
            )
            clips.append(img_clip)

        video = concatenate_videoclips(clips, method="compose")

        music_clip = AudioFileClip(music_data["path"]).subclip(0, video.duration).volumex(0.5)
        mixed = CompositeAudioClip([video.audio, music_clip])
        video = video.set_audio(mixed)

        temp_path = output_path.with_suffix(".temp.mp4")
        video.write_videofile(
            str(temp_path), fps=30, codec="libx264", audio_codec="aac",
            verbose=False, logger=None,
        )
        video.close()

        # Burn subtitles with FFmpeg. On Windows, escape drive colon for FFmpeg filter.
        srt_for_ffmpeg = Path(srt_path).as_posix()
        if len(srt_for_ffmpeg) > 1 and srt_for_ffmpeg[1] == ":":
            srt_for_ffmpeg = srt_for_ffmpeg[0] + "\\:" + srt_for_ffmpeg[2:]

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(temp_path),
                    "-vf", f"subtitles='{srt_for_ffmpeg}':force_style='FontName=Arial,FontSize=14,Alignment=2'",
                    "-c:a", "copy", str(output_path),
                ],
                check=True, capture_output=True,
            )
            temp_path.unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            # Subtitle burn failed (common on Windows path edge cases); ship without subtitles
            temp_path.rename(output_path)

        size = output_path.stat().st_size
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
            capture_output=True, text=True, check=True,
        )
        duration_s = float(probe.stdout.strip())
        return {"path": str(output_path), "duration_s": duration_s, "file_size_bytes": size}
