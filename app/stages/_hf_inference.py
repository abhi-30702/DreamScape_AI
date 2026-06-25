import json
import os


def hf_generate(model: str, prompt: str, timeout: float = 120.0) -> dict:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN env var is not set — add it to .env or Space secrets"
        )
    from huggingface_hub import InferenceClient
    try:
        client = InferenceClient(model=model, token=token, timeout=timeout)
        raw = client.text_generation(prompt, max_new_tokens=1024, return_full_text=False)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "unauthorized" in msg.lower():
            raise RuntimeError(
                "HF Inference API auth failed — is HF_TOKEN set and does it have Llama access?"
            ) from e
        if "429" in msg or "rate limit" in msg.lower():
            raise RuntimeError("HF Inference API rate limit hit") from e
        raise RuntimeError(f"HF Inference API error: {e}") from e
    if not raw:
        raise RuntimeError("HF Inference API returned empty response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"HF returned non-JSON: {raw[:200]}") from e
