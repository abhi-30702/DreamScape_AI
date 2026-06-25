import pytest
import os
from fastapi.testclient import TestClient

@pytest.fixture
def client(tmp_path):
    from app.main import create_app
    return TestClient(create_app(
        cache_dir=tmp_path / "cache",
        stub_stages={1, 2, 3, 4, 5, 6},
    ))

def test_generate_returns_run_id(client):
    resp = client.post("/generate", json={"prompt": "A wolf howls at the moon", "duration": 60, "style": "cinematic", "voice": "female"})
    assert resp.status_code == 200
    assert "run_id" in resp.json()

def test_get_run_returns_complete(client):
    resp = client.post("/generate", json={"prompt": "A wolf howls", "duration": 30, "style": "cinematic", "voice": "female"})
    run_id = resp.json()["run_id"]
    resp2 = client.get(f"/run/{run_id}")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "complete"

def test_blocked_prompt_returns_422(client):
    resp = client.post("/generate", json={"prompt": "A murder scene", "duration": 60, "style": "cinematic", "voice": "female"})
    assert resp.status_code == 422

def test_download_returns_mp4(client):
    resp = client.post("/generate", json={"prompt": "A wolf howls", "duration": 30, "style": "cinematic", "voice": "female"})
    run_id = resp.json()["run_id"]
    resp2 = client.get(f"/run/{run_id}/download")
    assert resp2.status_code == 200
    assert "video" in resp2.headers["content-type"]
