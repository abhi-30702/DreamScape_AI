from pathlib import Path
from app.cache import Cache, prompt_hash
from app.models.schemas import (
    PipelineRun, SpeakerSettings,
    ParsedPrompt, ScenePlan, VisualOutput, NarrationOutput,
    SubtitleOutput, MusicOutput, VideoOutput,
)
from app.stages.stage1_parse import Stage1Parse
from app.stages.stage2_expand import Stage2Expand
from app.stages.stage3_visual import Stage3Visual
from app.stages.stage4_narrate import Stage4Narrate
from app.stages.stage5_subtitle import Stage5Subtitle
from app.stages.stage6_music import Stage6Music
from app.stages.stage7_assemble import Stage7Assemble

_SPEAKER: dict[str, SpeakerSettings] = {
    "happy":   SpeakerSettings(speaker_id="female_en_1", pitch_semitones=2.0,  speed=0.95),
    "neutral": SpeakerSettings(speaker_id="neutral_en_1", pitch_semitones=0.0, speed=1.0),
    "sad":     SpeakerSettings(speaker_id="male_en_1",   pitch_semitones=-1.0, speed=1.05),
}
_MUSIC_COND: dict[str, str] = {
    "happy":   "uplifting, bright, major key, fast tempo",
    "neutral": "calm, ambient, minimal, mid-tempo",
    "sad":     "melancholic, minor key, slow tempo, sparse",
}
_NEG_PROMPT: dict[str, str] = {
    "happy":   "dark, bleak, sad, monochrome",
    "neutral": "",
    "sad":     "bright, colorful, cheerful, optimistic",
}
_SCHEMA_MAP = {
    1: ("parsed_prompt", ParsedPrompt),
    2: ("scene_plan", ScenePlan),
    3: ("visual_output", VisualOutput),
    4: ("narration_output", NarrationOutput),
    5: ("subtitle_output", SubtitleOutput),
    6: ("music_output", MusicOutput),
    7: ("video_output", VideoOutput),
}


class Orchestrator:
    def __init__(self, cache: Cache, stub_stages: set[int]):
        self.cache = cache
        kw = {"cache_dir": cache.asset_dir, "stub_stages": stub_stages}
        self.stages = [
            Stage1Parse(**kw), Stage2Expand(**kw), Stage3Visual(**kw),
            Stage4Narrate(**kw), Stage5Subtitle(**kw), Stage6Music(**kw),
            Stage7Assemble(**kw),
        ]

    def run_pipeline(self, prompt: str, duration: int, style: str, voice: str) -> PipelineRun:
        from app.filters import check_prompt
        check_prompt(prompt)

        phash = prompt_hash(prompt, {"duration": duration, "style": style, "voice": voice})
        run_id = self.cache.find_run_by_hash(phash)
        if run_id is None:
            run_id = self.cache.create_run(
                phash, {"prompt": prompt, "duration": duration, "style": style, "voice": voice}
            )

        run = PipelineRun(
            run_id=run_id, prompt_hash=phash, prompt=prompt,
            duration_target_s=duration, style=style, voice=voice,
        )

        # Load all already-cached stages first
        for stage in self.stages:
            if self.cache.stage_complete(run_id, stage.stage_num):
                self._apply_output(run, stage.stage_num,
                                   self.cache.load_stage_output(run_id, stage.stage_num))

        # Run any incomplete stages
        run.status = "running"
        for stage in self.stages:
            if not self.cache.stage_complete(run_id, stage.stage_num):
                output = stage.run(self._build_input(run, stage.stage_num))
                self.cache.save_stage_output(run_id, stage.stage_num, output)
                self._apply_output(run, stage.stage_num, output)

        run.status = "complete"
        return run

    def run_stage(self, run_id: str, stage_num: int) -> PipelineRun:
        self.cache.invalidate_from(run_id, stage_num)
        params = self.cache.load_run_params(run_id)
        return self.run_pipeline(
            prompt=params["prompt"], duration=params["duration"],
            style=params["style"], voice=params["voice"],
        )

    def _build_input(self, run: PipelineRun, stage_num: int) -> dict:
        sentiment = run.parsed_prompt.sentiment if run.parsed_prompt else "neutral"
        asset_dir = str(self.cache.get_asset_dir(run.run_id, stage_num))
        if stage_num == 1:
            return {"prompt": run.prompt, "duration": run.duration_target_s,
                    "style": run.style, "voice": run.voice}
        if stage_num == 2:
            return {"parsed_prompt": run.parsed_prompt.model_dump()}
        if stage_num == 3:
            return {"scenes": [s.model_dump() for s in run.scene_plan.scenes],
                    "sentiment": sentiment, "negative_prompt": _NEG_PROMPT[sentiment],
                    "style": run.style, "asset_dir": asset_dir}
        if stage_num == 4:
            return {"scenes": [s.model_dump() for s in run.scene_plan.scenes],
                    "speaker_settings": _SPEAKER[sentiment].model_dump(), "asset_dir": asset_dir}
        if stage_num == 5:
            return {"audio_assets": [a.model_dump() for a in run.narration_output.audio],
                    "scenes": [s.model_dump() for s in run.scene_plan.scenes], "asset_dir": asset_dir}
        if stage_num == 6:
            return {"mood": sentiment, "total_duration_s": run.narration_output.total_duration_s,
                    "music_condition": _MUSIC_COND[sentiment], "asset_dir": asset_dir}
        if stage_num == 7:
            return {"images": [i.model_dump() for i in run.visual_output.images],
                    "narration": [a.model_dump() for a in run.narration_output.audio],
                    "music": run.music_output.model_dump(),
                    "subtitles": run.subtitle_output.model_dump(),
                    "output_path": str(self.cache.get_asset_dir(run.run_id, 7) / "output.mp4")}

    def _apply_output(self, run: PipelineRun, stage_num: int, output: dict):
        attr, model_cls = _SCHEMA_MAP[stage_num]
        setattr(run, attr, model_cls(**output))
        if stage_num == 4:
            self._correct_scene_durations(run)

    def _correct_scene_durations(self, run: PipelineRun):
        scene_by_id = {s.id: s for s in run.scene_plan.scenes}
        for audio in run.narration_output.audio:
            scene_by_id[audio.scene_id].duration_estimate_s = audio.duration_s + 1.0
        total = run.narration_output.total_duration_s
        target = run.duration_target_s
        if abs(total - target) > 2:
            ratio = target / total
            for scene in run.scene_plan.scenes:
                scene.duration_estimate_s = round(scene.duration_estimate_s * ratio, 2)
