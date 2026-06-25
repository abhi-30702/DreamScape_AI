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
DEFAULT_VOICE = os.getenv("DREAMSCAPE_DEFAULT_VOICE", "female")
DEFAULT_DURATION = int(os.getenv("DREAMSCAPE_DEFAULT_DURATION", "60"))
