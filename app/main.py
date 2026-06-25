import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.cache import Cache
from app.orchestrator import Orchestrator


class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 60
    style: str = "cinematic"
    voice: str = "female"


def create_app(cache_dir: Path = None, stub_stages: set[int] = None) -> FastAPI:
    if cache_dir is None:
        cache_dir = Path(os.getenv("DREAMSCAPE_CACHE_DIR", "cache"))
    if stub_stages is None:
        stub_stages = {
            int(s) for s in os.getenv("DREAMSCAPE_STUB_STAGES", "3,4,6").split(",") if s.strip()
        }

    cache = Cache(db_path=cache_dir / "runs.db", asset_dir=cache_dir / "assets")
    orch = Orchestrator(cache=cache, stub_stages=stub_stages)

    app = FastAPI(title="DreamScapeAI")

    @app.post("/generate")
    def generate(req: GenerateRequest):
        try:
            run = orch.run_pipeline(req.prompt, req.duration, req.style, req.voice)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"run_id": run.run_id, "status": run.status}

    @app.get("/run/{run_id}")
    def get_run(run_id: str):
        if not cache.stage_complete(run_id, 7):
            raise HTTPException(status_code=404, detail="Run not found or incomplete")
        output = cache.load_stage_output(run_id, 7)
        return {"run_id": run_id, "status": "complete", "video": output}

    @app.post("/run/{run_id}/regenerate/{stage_num}")
    def regenerate(run_id: str, stage_num: int):
        try:
            run = orch.run_stage(run_id, stage_num)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"run_id": run.run_id, "status": run.status}

    @app.get("/run/{run_id}/download")
    def download(run_id: str):
        if not cache.stage_complete(run_id, 7):
            raise HTTPException(status_code=404, detail="Video not ready")
        output = cache.load_stage_output(run_id, 7)
        path = Path(output["path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(str(path), media_type="video/mp4", filename="dreamscape.mp4")

    return app


app = create_app()
