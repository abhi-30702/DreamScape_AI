from app.stages.base import BaseStage
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.stages._llm import llm_generate

_BEATS = [
    ("Wide establishing shot", "The world holds its breath..."),
    ("Close-up on subject", "Something stirs beneath the surface..."),
    ("Medium shot, rising tension", "A moment of decision approaches..."),
    ("Wide shot, climax", "Everything converges at once..."),
]

_EXPAND_PROMPT = """\
You are a creative director planning scenes for a short video.

Story: {prompt}
Mood: {sentiment}
Style: {style}
Total duration: {duration_target_s}s

Create exactly {n_scenes} scenes. Return ONLY valid JSON (no markdown):
{{"scenes": [
  {{
    "id": 0,
    "description": "50-80 word visual description for AI image generation",
    "narration_text": "15-25 word voiceover narration",
    "mood": "{sentiment}",
    "duration_estimate_s": {per_scene_dur}
  }}
]}}

All scenes must have mood = "{sentiment}". Return only the JSON object.
"""


def _safe_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class Stage2Expand(BaseStage):
    stage_num = 2

    def _run_real(self, input: dict) -> dict:
        pp = input["parsed_prompt"]
        duration = max(1, int(pp.get("duration_target_s", 60)))
        sentiment = pp.get("sentiment", "neutral")
        style = pp.get("style", "cinematic")
        n_scenes = max(4, min(8, duration // 15))
        per_scene_dur = round(duration / n_scenes, 1)

        safe_prompt = pp.get("prompt", "").replace("{", "{{").replace("}", "}}")
        safe_sentiment = sentiment.replace("{", "{{").replace("}", "}}")
        safe_style = style.replace("{", "{{").replace("}", "}}")
        prompt_text = _EXPAND_PROMPT.format(
            prompt=safe_prompt,
            sentiment=safe_sentiment,
            style=safe_style,
            duration_target_s=duration,
            n_scenes=n_scenes,
            per_scene_dur=per_scene_dur,
        )
        result = llm_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text, timeout=180.0)
        raw_scenes = result.get("scenes", [])

        if not isinstance(raw_scenes, list) or len(raw_scenes) < 4:
            return self._run_stub(input)

        scenes = []
        for i, s in enumerate(raw_scenes[:8]):
            mood = s.get("mood", sentiment)
            if mood not in ("happy", "neutral", "sad"):
                mood = sentiment
            scenes.append({
                "id": i,
                "description": str(s.get("description", f"Scene {i}"))[:500],
                "narration_text": str(s.get("narration_text", f"Scene {i} unfolds."))[:200],
                "mood": mood,
                "duration_estimate_s": _safe_float(s.get("duration_estimate_s"), per_scene_dur),
            })
        return {"scenes": scenes}

    def _run_stub(self, input: dict) -> dict:
        pp = input["parsed_prompt"]
        entities = pp.get("key_entities", ["subject"])
        mood = pp.get("sentiment", "neutral")
        duration = pp.get("duration_target_s", 60)
        scene_dur = duration / len(_BEATS)
        scenes = []
        for i, (visual, narration) in enumerate(_BEATS):
            entity = entities[i % len(entities)]
            scenes.append({
                "id": i,
                "description": f"{visual} featuring {entity}.",
                "narration_text": f"{narration} The {entity} reveals itself.",
                "mood": mood,
                "duration_estimate_s": round(scene_dur, 1),
            })
        return {"scenes": scenes}
