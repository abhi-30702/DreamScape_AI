import json
import re
from pathlib import Path

import pytest

MANIFEST_PATH = Path("study_videos/manifest.json")
EXPECTED_VIDEO_COUNT = 20
ID_PATTERN = re.compile(r"^v\d{2}$")
REQUIRED_KEYS = {"id", "filename", "prompt", "sentiment", "style"}
VALID_SENTIMENTS = {"happy", "neutral", "sad"}
VALID_STYLES = {"cinematic", "documentary", "anime", "noir", "horror"}


def _load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_file_exists():
    assert MANIFEST_PATH.is_file(), f"{MANIFEST_PATH} must exist"


def test_manifest_parses_as_list():
    data = _load_manifest()
    assert isinstance(data, list)


def test_manifest_has_expected_count():
    data = _load_manifest()
    assert len(data) == EXPECTED_VIDEO_COUNT


def test_manifest_entries_have_required_keys():
    data = _load_manifest()
    for entry in data:
        missing = REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"Entry {entry.get('id')!r} missing keys: {missing}"


def test_manifest_ids_are_unique_and_match_pattern():
    data = _load_manifest()
    ids = [entry["id"] for entry in data]
    assert len(set(ids)) == len(ids), "manifest ids must be unique"
    for entry_id in ids:
        assert ID_PATTERN.match(entry_id), f"id {entry_id!r} does not match ^v\\d{{2}}$"


def test_manifest_sentiments_and_styles_are_valid():
    data = _load_manifest()
    for entry in data:
        assert entry["sentiment"] in VALID_SENTIMENTS, (
            f"{entry['id']}: sentiment {entry['sentiment']!r} not in {VALID_SENTIMENTS}"
        )
        assert entry["style"] in VALID_STYLES, (
            f"{entry['id']}: style {entry['style']!r} not in {VALID_STYLES}"
        )


def test_manifest_filenames_match_id():
    data = _load_manifest()
    for entry in data:
        assert entry["filename"] == f"{entry['id']}.mp4", (
            f"{entry['id']}: filename {entry['filename']!r} expected {entry['id']}.mp4"
        )


def test_manifest_referenced_files_exist_or_skip():
    data = _load_manifest()
    study_dir = MANIFEST_PATH.parent
    missing = [e["filename"] for e in data if not (study_dir / e["filename"]).is_file()]
    if missing:
        pytest.skip(f"{len(missing)} MP4(s) not yet generated: {missing[:3]}...")
