from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel


class ParsedPrompt(BaseModel):
    prompt: str
    sentiment: Literal["happy", "neutral", "sad"]
    duration_target_s: int
    style: str
    key_entities: list[str]


class SceneData(BaseModel):
    id: int
    description: str
    narration_text: str
    mood: Literal["happy", "neutral", "sad"]
    duration_estimate_s: float


class ScenePlan(BaseModel):
    scenes: list[SceneData]


class SpeakerSettings(BaseModel):
    speaker_id: str
    pitch_semitones: float
    speed: float


class ImageAsset(BaseModel):
    scene_id: int
    path: str
    width: int
    height: int


class VisualOutput(BaseModel):
    images: list[ImageAsset]


class AudioAsset(BaseModel):
    scene_id: int
    path: str
    duration_s: float


class NarrationOutput(BaseModel):
    audio: list[AudioAsset]
    total_duration_s: float


class SubtitleEntry(BaseModel):
    index: int
    start_s: float
    end_s: float
    text: str


class SubtitleOutput(BaseModel):
    srt_path: str
    entries: list[SubtitleEntry]


class MusicOutput(BaseModel):
    path: str
    duration_s: float


class VideoOutput(BaseModel):
    path: str
    duration_s: float
    file_size_bytes: int


class PipelineRun(BaseModel):
    run_id: str
    prompt_hash: str
    prompt: str
    duration_target_s: int
    style: str
    voice: str
    parsed_prompt: Optional[ParsedPrompt] = None
    scene_plan: Optional[ScenePlan] = None
    visual_output: Optional[VisualOutput] = None
    narration_output: Optional[NarrationOutput] = None
    subtitle_output: Optional[SubtitleOutput] = None
    music_output: Optional[MusicOutput] = None
    video_output: Optional[VideoOutput] = None
    created_at: datetime = None
    status: Literal["pending", "running", "complete", "failed"] = "pending"

    def model_post_init(self, __context: object) -> None:
        if self.created_at is None:
            self.__dict__["created_at"] = datetime.now(timezone.utc)
