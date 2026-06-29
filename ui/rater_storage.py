import json
import os

from huggingface_hub import HfApi


def _get_config() -> tuple[str, str]:
    repo_id = os.getenv("DREAMSCAPE_RATER_DATASET")
    token = os.getenv("HF_TOKEN")
    if not repo_id or not token:
        raise RuntimeError(
            "Rater study not configured: set DREAMSCAPE_RATER_DATASET and HF_TOKEN."
        )
    return repo_id, token


def _list_rater_files(rater_id: str) -> list[str]:
    repo_id, token = _get_config()
    api = HfApi(token=token)
    prefix = f"responses/{rater_id}/"
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [f for f in all_files if f.startswith(prefix)]


def list_completed(rater_id: str) -> set[str]:
    files = _list_rater_files(rater_id)
    completed: set[str] = set()
    for path in files:
        name = path.rsplit("/", 1)[-1]
        if not name.endswith(".json"):
            continue
        stem = name[: -len(".json")]
        if stem == "_overall":
            continue
        completed.add(stem)
    return completed


def has_completed_overall(rater_id: str) -> bool:
    files = _list_rater_files(rater_id)
    return any(f.endswith(f"/{rater_id}/_overall.json") for f in files)


def save_response(rater_id: str, video_id: str, payload: dict) -> None:
    repo_id, token = _get_config()
    api = HfApi(token=token)
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    api.upload_file(
        path_or_fileobj=body,
        path_in_repo=f"responses/{rater_id}/{video_id}.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"rater {rater_id} submission {video_id}",
    )
