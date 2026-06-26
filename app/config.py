from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

STUB_STAGES: set[int] = {
    int(s) for s in os.getenv("DREAMSCAPE_STUB_STAGES", "3,4,6").split(",") if s.strip()
}
CACHE_DIR = Path(os.getenv("DREAMSCAPE_CACHE_DIR", "cache"))
CACHE_TTL_HOURS = int(os.getenv("DREAMSCAPE_CACHE_TTL_HOURS", "24"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("DREAMSCAPE_OLLAMA_MODEL", "llama3.1:8b")
LLM_BACKEND = os.getenv("DREAMSCAPE_LLM_BACKEND", "ollama")
DEFAULT_VOICE = os.getenv("DREAMSCAPE_DEFAULT_VOICE", "female")
DEFAULT_DURATION = int(os.getenv("DREAMSCAPE_DEFAULT_DURATION", "60"))
WHISPER_MODEL_SIZE = os.getenv("DREAMSCAPE_WHISPER_MODEL", "base")
MUSICGEN_MODEL = os.getenv("DREAMSCAPE_MUSICGEN_MODEL", "facebook/musicgen-medium")
