import json
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import HfHubHTTPError

from ui import rater_storage


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DREAMSCAPE_RATER_DATASET", "test-user/test-dataset")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")


def test_list_completed_returns_video_ids_for_rater():
    files = [
        "responses/reviewer_a/v01.json",
        "responses/reviewer_a/v02.json",
        "responses/reviewer_a/v07.json",
        "responses/reviewer_b/v01.json",
        "README.md",
    ]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        result = rater_storage.list_completed("reviewer_a")
    assert result == {"v01", "v02", "v07"}


def test_list_completed_excludes_overall_marker():
    files = [
        "responses/reviewer_a/v01.json",
        "responses/reviewer_a/_overall.json",
    ]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        result = rater_storage.list_completed("reviewer_a")
    assert result == {"v01"}
    assert "_overall" not in result


def test_list_completed_empty_for_new_rater():
    files = ["responses/reviewer_a/v01.json"]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        result = rater_storage.list_completed("brand_new_rater")
    assert result == set()


def test_has_completed_overall_true_when_marker_present():
    files = ["responses/reviewer_a/_overall.json"]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        assert rater_storage.has_completed_overall("reviewer_a") is True


def test_has_completed_overall_false_when_absent():
    files = ["responses/reviewer_a/v01.json"]
    with patch("ui.rater_storage.HfApi") as MockApi:
        MockApi.return_value.list_repo_files.return_value = files
        assert rater_storage.has_completed_overall("reviewer_a") is False


def test_save_response_uploads_to_expected_path():
    payload = {"schema_version": 1, "rater_id": "reviewer_a", "video_id": "v07"}
    with patch("ui.rater_storage.HfApi") as MockApi:
        instance = MockApi.return_value
        rater_storage.save_response("reviewer_a", "v07", payload)
    instance.upload_file.assert_called_once()
    kwargs = instance.upload_file.call_args.kwargs
    assert kwargs["path_in_repo"] == "responses/reviewer_a/v07.json"
    assert kwargs["repo_id"] == "test-user/test-dataset"
    assert kwargs["repo_type"] == "dataset"
    uploaded_bytes = kwargs["path_or_fileobj"]
    assert json.loads(uploaded_bytes.decode("utf-8")) == payload


def test_save_response_works_for_overall_marker():
    payload = {"schema_version": 1, "rater_id": "reviewer_a", "overall_comment": "Nice."}
    with patch("ui.rater_storage.HfApi") as MockApi:
        instance = MockApi.return_value
        rater_storage.save_response("reviewer_a", "_overall", payload)
    kwargs = instance.upload_file.call_args.kwargs
    assert kwargs["path_in_repo"] == "responses/reviewer_a/_overall.json"


def test_save_response_raises_when_dataset_env_missing(monkeypatch):
    monkeypatch.delenv("DREAMSCAPE_RATER_DATASET", raising=False)
    with pytest.raises(RuntimeError, match="Rater study not configured"):
        rater_storage.save_response("reviewer_a", "v07", {})


def test_save_response_raises_when_token_env_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Rater study not configured"):
        rater_storage.save_response("reviewer_a", "v07", {})


def test_save_response_propagates_hfhub_http_error():
    with patch("ui.rater_storage.HfApi") as MockApi:
        response = MagicMock(status_code=500)
        MockApi.return_value.upload_file.side_effect = HfHubHTTPError(
            "boom", response=response
        )
        with pytest.raises(HfHubHTTPError):
            rater_storage.save_response("reviewer_a", "v07", {"x": 1})
