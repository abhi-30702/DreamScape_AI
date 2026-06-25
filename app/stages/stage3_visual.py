from pathlib import Path
from app.stages.base import BaseStage

_COLORS = {"happy": (255, 220, 80), "neutral": (140, 140, 160), "sad": (70, 90, 130)}

class Stage3Visual(BaseStage):
    stage_num = 3

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("SDXL integration — Plan B")

    def _run_stub(self, input: dict) -> dict:
        from PIL import Image, ImageDraw
        scenes = input["scenes"]
        mood = input.get("sentiment", "neutral")
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)
        color = _COLORS.get(mood, (140, 140, 160))
        images = []
        for scene in scenes:
            img = Image.new("RGB", (1024, 576), color)
            draw = ImageDraw.Draw(img)
            label = f"Scene {scene['id']}: {scene['description'][:50]}"
            draw.text((20, 270), label, fill=(255, 255, 255))
            path = asset_dir / f"scene_{scene['id']}.png"
            img.save(str(path))
            images.append({"scene_id": scene["id"], "path": str(path), "width": 1024, "height": 576})
        return {"images": images}
