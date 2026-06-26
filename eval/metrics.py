import jiwer


def compute_wer(hypotheses: list[str], references: list[str]) -> dict:
    per_scene = [jiwer.wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    mean = sum(per_scene) / len(per_scene) if per_scene else 0.0
    return {"per_scene": per_scene, "mean": round(mean, 4)}


def compute_clip_score(image_paths: list[str], texts: list[str]) -> dict:
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    scores = []
    with torch.no_grad():
        for path, text in zip(image_paths, texts):
            image = preprocess(Image.open(path)).unsqueeze(0)
            tokens = tokenizer([text])
            image_feats = model.encode_image(image)
            text_feats = model.encode_text(tokens)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            score = float((image_feats @ text_feats.T).squeeze())
            scores.append(score)

    mean = sum(scores) / len(scores) if scores else 0.0
    return {"per_scene": scores, "mean": round(mean, 4)}
