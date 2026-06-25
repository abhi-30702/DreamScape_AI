import pytest
from app.filters import check_prompt

def test_clean_prompt_passes():
    check_prompt("A lone wolf howls at the moon")

def test_blocked_prompt_raises():
    with pytest.raises(ValueError, match="restricted content"):
        check_prompt("A story about murder and blood")

def test_case_insensitive():
    with pytest.raises(ValueError):
        check_prompt("KILL the dragon")
