import json
import httpx


def ollama_generate(base_url: str, model: str, prompt: str, timeout: float = 120.0) -> dict:
    try:
        resp = httpx.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "format": "json", "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. Is Ollama running? (run: ollama serve)"
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama error {e.response.status_code}: {e.response.text[:200]}") from e
    raw = resp.json().get("response")
    if not raw:
        raise RuntimeError(f"Ollama returned no 'response' field: {list(resp.json().keys())}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama returned non-JSON: {raw[:200]}") from e
