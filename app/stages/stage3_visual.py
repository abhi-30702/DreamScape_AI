import threading
from pathlib import Path
from app.stages.base import BaseStage

_COLORS = {"happy": (255, 220, 80), "neutral": (140, 140, 160), "sad": (70, 90, 130)}

_PIPE = None  # module-level lazy singleton
_PIPE_LOCK = threading.Lock()


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        with _PIPE_LOCK:
            if _PIPE is None:
                from diffusers import StableDiffusionXLPipeline
                import torch
                _PIPE = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
                _PIPE.enable_model_cpu_offload()
    return _PIPE


def _placeholder(mood: str):
    from PIL import Image
    return Image.new("RGB", (1024, 576), _COLORS.get(mood, _COLORS["neutral"]))


class Stage3Visual(BaseStage):
    stage_num = 3

    def _run_real(self, input: dict) -> dict:
        scenes = input["scenes"]
        style = input.get("style", "cinematic")
        negative_prompt = input.get("negative_prompt", "")
        asset_dir = Path(input["asset_dir"])
        asset_dir.mkdir(parents=True, exist_ok=True)

        pipe = _get_pipe()
        images = []
        for scene in scenes:
            img = self._generate_scene(pipe, scene, style, negative_prompt)
            path = asset_dir / f"scene_{scene['id']}.png"
            img.save(str(path))
            images.append({"scene_id": scene["id"], "path": str(path), "width": 1024, "height": 576})
        return {"images": images}

    def _generate_scene(self, pipe, scene: dict, style: str, negative_prompt: str):
        import numpy as np
        base_prompt = f"{scene['description']}, {style} style, high quality, detailed"
        short_prompt = f"{scene['description'][:80]}, {style}"
        for attempt in range(3):
            prompt = base_prompt if attempt == 0 else short_prompt
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=1024,
                height=576,
                num_inference_steps=30,
                guidance_scale=7.5,
            )
            if not result.images:
                continue
            img = result.images[0]
            if np.array(img).mean() > 5.0:  # not a black safety-checker image
                return img
        return _placeholder(scene.get("mood", "neutral"))

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
