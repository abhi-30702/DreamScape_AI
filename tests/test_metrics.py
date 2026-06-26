import pytest
from unittest.mock import MagicMock, patch


def test_wer_perfect_match():
    from eval.metrics import compute_wer
    result = compute_wer(["hello world"], ["hello world"])
    assert result["per_scene"] == [0.0]
    assert result["mean"] == 0.0


def test_wer_partial_match():
    # "hello earth" vs "hello world" — 1 substitution out of 2 words = 0.5
    from eval.metrics import compute_wer
    result = compute_wer(["hello earth"], ["hello world"])
    assert result["per_scene"][0] == pytest.approx(0.5)
    assert result["mean"] == pytest.approx(0.5)
