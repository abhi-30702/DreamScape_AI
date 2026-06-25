import json
import sqlite3
import hashlib
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path


class Cache:
    def __init__(self, db_path: Path, asset_dir: Path):
        self.db_path = db_path
        self.asset_dir = asset_dir
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    params_json TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stage_outputs (
                    run_id TEXT,
                    stage_num INTEGER,
                    output_json TEXT,
                    completed_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (run_id, stage_num)
                )
            """)

    def create_run(self, phash: str, params: dict) -> str:
        run_id = str(uuid.uuid4())[:8]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, prompt_hash, params_json) VALUES (?, ?, ?)",
                (run_id, phash, json.dumps(params)),
            )
        return run_id

    def find_run_by_hash(self, phash: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE prompt_hash=? ORDER BY created_at DESC LIMIT 1",
                (phash,),
            ).fetchone()
        return row[0] if row else None

    def stage_complete(self, run_id: str, stage_num: int) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM stage_outputs WHERE run_id=? AND stage_num=?",
                (run_id, stage_num),
            ).fetchone()
        return row is not None

    def save_stage_output(self, run_id: str, stage_num: int, output: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO stage_outputs (run_id, stage_num, output_json) VALUES (?, ?, ?)",
                (run_id, stage_num, json.dumps(output)),
            )

    def load_stage_output(self, run_id: str, stage_num: int) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT output_json FROM stage_outputs WHERE run_id=? AND stage_num=?",
                (run_id, stage_num),
            ).fetchone()
        if row is None:
            raise KeyError(f"No cached output for run={run_id} stage={stage_num}")
        return json.loads(row[0])

    def invalidate_from(self, run_id: str, stage_num: int):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM stage_outputs WHERE run_id=? AND stage_num>=?",
                (run_id, stage_num),
            )

    def get_asset_dir(self, run_id: str, stage_num: int) -> Path:
        d = self.asset_dir / run_id / f"stage_{stage_num}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_run_params(self, run_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT params_json FROM pipeline_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Run not found: {run_id}")
        return json.loads(row[0])

    def prune_expired(self, ttl_hours: int = 24):
        cutoff = (datetime.utcnow() - timedelta(hours=ttl_hours)).isoformat()
        with self._conn() as conn:
            old = conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE created_at < ?", (cutoff,)
            ).fetchall()
            for (run_id,) in old:
                conn.execute("DELETE FROM stage_outputs WHERE run_id=?", (run_id,))
                conn.execute("DELETE FROM pipeline_runs WHERE run_id=?", (run_id,))
                run_dir = self.asset_dir / run_id
                if run_dir.exists():
                    shutil.rmtree(run_dir)


def prompt_hash(prompt: str, params: dict) -> str:
    key = json.dumps({"prompt": prompt, **params}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]
