# Study videos

This directory holds the 20 MP4s rated in the human evaluation study (Phase 2 of the Evaluation Framework).

- `manifest.json` is the source of truth for the videos in the study.
- MP4 files are tracked with Git LFS (see `.gitattributes`).
- MP4s are generated separately on a GPU runner and committed via `git lfs push`. Until that happens, the file-existence test in `tests/test_manifest.py` will skip.
