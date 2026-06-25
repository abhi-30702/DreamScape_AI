from app.stages.base import BaseStage
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.stages._ollama import ollama_generate

_SAD = {"sad", "lonely", "lost", "dark", "grief", "sorrow", "alone", "death", "cry", "melanchol", "despair"}
_HAPPY = {"happy", "joyful", "bright", "celebrat", "wonderful", "cheer", "laugh", "love", "hope", "delight"}

_PARSE_PROMPT = """\
You are a creative director's assistant. Analyze this video prompt and return structured JSON.

Prompt: "{prompt}"

Return ONLY a JSON object with exactly these keys:
- "sentiment": one of "happy", "neutral", "sad" (the overall emotional tone of the story)
- "duration_target_s": integer, use {duration} exactly (do not change this value)
- "style": string, use "{style}" exactly (do not change this value)
- "key_entities": list of 3-5 main nouns, characters, or locations from the prompt
- "prompt": the original prompt text exactly as provided

Example: {{"sentiment": "neutral", "duration_target_s": 60, "style": "cinematic", "key_entities": ["wolf", "moon", "mountain"], "prompt": "A wolf howls"}}
"""


class Stage1Parse(BaseStage):
    stage_num = 1

    def _run_real(self, input: dict) -> dict:
        safe_prompt = input["prompt"].replace("{", "{{").replace("}", "}}")
        prompt_text = _PARSE_PROMPT.format(
            prompt=safe_prompt,
            duration=input.get("duration", 60),
            style=input.get("style", "cinematic"),
        )
        result = ollama_generate(OLLAMA_BASE_URL, OLLAMA_MODEL, prompt_text)
        sentiment = result.get("sentiment", "neutral")
        if sentiment not in ("happy", "neutral", "sad"):
            sentiment = "neutral"
        entities = result.get("key_entities", [])
        if not isinstance(entities, list) or not entities:
            entities = [w.strip(".,!?") for w in input["prompt"].split() if len(w) > 4][:4] or ["subject"]
        return {
            "prompt": input["prompt"],
            "sentiment": sentiment,
            "duration_target_s": int(input.get("duration", 60)),
            "style": input.get("style", "cinematic"),
            "key_entities": [str(e) for e in entities[:5]],
        }

    def _run_stub(self, input: dict) -> dict:
        prompt = input["prompt"].lower()
        if any(w in prompt for w in _SAD):
            sentiment = "sad"
        elif any(w in prompt for w in _HAPPY):
            sentiment = "happy"
        else:
            sentiment = "neutral"
        words = [w.strip(".,!?") for w in prompt.split() if len(w) > 4]
        entities = list(dict.fromkeys(words))[:4] or ["character", "setting", "action", "moment"]
        return {
            "prompt": input["prompt"],
            "sentiment": sentiment,
            "duration_target_s": input.get("duration", 60),
            "style": input.get("style", "cinematic"),
            "key_entities": entities,
        }
