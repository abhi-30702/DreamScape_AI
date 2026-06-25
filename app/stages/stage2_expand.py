from app.stages.base import BaseStage

_BEATS = [
    ("Wide establishing shot", "The world holds its breath..."),
    ("Close-up on subject", "Something stirs beneath the surface..."),
    ("Medium shot, rising tension", "A moment of decision approaches..."),
    ("Wide shot, climax", "Everything converges at once..."),
]

class Stage2Expand(BaseStage):
    stage_num = 2

    def _run_real(self, input: dict) -> dict:
        raise NotImplementedError("Ollama integration — Plan B")

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
