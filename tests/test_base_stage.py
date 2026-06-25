import pytest
from pathlib import Path
from app.stages.base import BaseStage

class DummyStage(BaseStage):
    stage_num = 99

    def _run_real(self, input: dict) -> dict:
        return {"mode": "real", "value": input.get("x", 0) * 2}

    def _run_stub(self, input: dict) -> dict:
        return {"mode": "stub", "value": 42}

def test_real_mode(tmp_path):
    stage = DummyStage(cache_dir=tmp_path, stub_stages=set())
    assert stage.run({"x": 5}) == {"mode": "real", "value": 10}

def test_stub_mode(tmp_path):
    stage = DummyStage(cache_dir=tmp_path, stub_stages={99})
    assert stage.run({"x": 5}) == {"mode": "stub", "value": 42}

def test_stub_ignores_input(tmp_path):
    stage = DummyStage(cache_dir=tmp_path, stub_stages={99})
    assert stage.run({"x": 999}) == {"mode": "stub", "value": 42}
