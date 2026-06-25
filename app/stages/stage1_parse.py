from app.stages.base import BaseStage

_SAD = {"sad", "lonely", "lost", "dark", "grief", "sorrow", "alone", "death", "cry", "melanchol", "despair"}
_HAPPY = {"happy", "joyful", "bright", "celebrat", "wonderful", "cheer", "laugh", "love", "hope", "delight"}

class Stage1Parse(BaseStage):
    stage_num = 1

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("Ollama integration — Plan B")

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
