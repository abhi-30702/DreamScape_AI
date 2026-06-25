import pytest
from pathlib import Path
from app.cache import Cache, prompt_hash

@pytest.fixture
def cache(tmp_path):
    return Cache(db_path=tmp_path / "test.db", asset_dir=tmp_path / "assets")

def test_create_run(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    assert run_id is not None

def test_stage_not_complete_initially(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    assert not cache.stage_complete(run_id, 1)

def test_save_and_load_stage_output(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    cache.save_stage_output(run_id, 1, {"sentiment": "sad"})
    assert cache.stage_complete(run_id, 1)
    assert cache.load_stage_output(run_id, 1) == {"sentiment": "sad"}

def test_invalidate_from(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    for i in range(1, 6):
        cache.save_stage_output(run_id, i, {"stage": i})
    cache.invalidate_from(run_id, 3)
    assert cache.stage_complete(run_id, 2)
    assert not cache.stage_complete(run_id, 3)
    assert not cache.stage_complete(run_id, 4)

def test_get_asset_dir_creates_path(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    d = cache.get_asset_dir(run_id, 3)
    assert d.exists()

def test_find_run_by_hash_returns_none_when_missing(cache):
    assert cache.find_run_by_hash("nonexistent") is None

def test_find_run_by_hash_returns_run_id(cache):
    run_id = cache.create_run("myhash", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    assert cache.find_run_by_hash("myhash") == run_id

def test_prompt_hash_is_deterministic():
    h1 = prompt_hash("A wolf howls", {"duration": 60, "style": "cinematic", "voice": "female"})
    h2 = prompt_hash("A wolf howls", {"duration": 60, "style": "cinematic", "voice": "female"})
    assert h1 == h2

def test_prompt_hash_differs_for_different_inputs():
    h1 = prompt_hash("A wolf howls", {"duration": 60, "style": "cinematic", "voice": "female"})
    h2 = prompt_hash("A cat sleeps", {"duration": 60, "style": "cinematic", "voice": "female"})
    assert h1 != h2

def test_load_stage_output_raises_key_error(cache):
    run_id = cache.create_run("abc123", {"prompt": "test", "duration": 60, "style": "cinematic", "voice": "female"})
    with pytest.raises(KeyError):
        cache.load_stage_output(run_id, 99)

def test_load_run_params_raises_key_error(cache):
    with pytest.raises(KeyError):
        cache.load_run_params("nonexistent_run_id")

def test_prune_expired_removes_old_runs(cache, tmp_path):
    from datetime import datetime, timedelta, timezone
    import sqlite3
    # Create a run
    run_id = cache.create_run("oldhash", {"prompt": "old", "duration": 60, "style": "cinematic", "voice": "female"})
    cache.save_stage_output(run_id, 1, {"data": "test"})
    asset_dir = cache.get_asset_dir(run_id, 1)
    # Manually backdate the run in SQLite
    backdated = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(cache.db_path) as conn:
        conn.execute("UPDATE pipeline_runs SET created_at=? WHERE run_id=?", (backdated, run_id))
    # Prune with 24hr TTL
    cache.prune_expired(ttl_hours=24)
    # Run should be gone
    assert cache.find_run_by_hash("oldhash") is None
    assert not cache.stage_complete(run_id, 1)
