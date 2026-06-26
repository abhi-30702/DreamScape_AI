import jiwer


def compute_wer(hypotheses: list[str], references: list[str]) -> dict:
    per_scene = [jiwer.wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    mean = sum(per_scene) / len(per_scene) if per_scene else 0.0
    return {"per_scene": per_scene, "mean": round(mean, 4)}
