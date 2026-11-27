"""Writes test/anidex/fixtures.json for test/anidex_test.dart.

    python tool/anidex/make_fixtures.py PHOTO_DIR

- preprocess: a synthetic 640x480 PNG (committed) and the encoder input PyTorch makes from it, so the Dart
  preprocessing can be checked against BioCLIP's own.
- scores: for each photo in PHOTO_DIR, both ONNX encoders' embeddings and the species, score and breed that
  the same scoring in Python gives. Only the numbers are kept, not the photos.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from bioclip.predict import preprocess_img
from PIL import Image

APP = Path(__file__).resolve().parents[2]
ASSETS = APP / "assets" / "anidex"
OUT = APP / "test" / "anidex"


def synthetic_png(path: Path):
    y, x = np.mgrid[0:480, 0:640]
    rgb = np.stack([(x * 255 // 639), (y * 255 // 479), ((x + y) * 255 // 1118)], axis=-1).astype(np.uint8)
    rgb[150:330, 220:420] = (220, 40, 40)
    Image.fromarray(rgb).save(path)


def softmax(rows, e, scale):
    logits = scale * rows @ (e / np.linalg.norm(e))
    p = np.exp(logits - logits.max())
    return p / p.sum()


def score(meta, labels, breed_labels, embedding, breed_embedding):
    p = softmax(labels, embedding, meta["logit_scale"])
    owner = np.array(meta["label_owner"])
    species_p = np.bincount(owner, weights=p, minlength=len(meta["species"]))
    best = int(species_p.argmax())
    s = meta["species"][best]
    breed = None
    if s.get("breeds"):
        start = s["breed_vectors_start"]
        bp = softmax(breed_labels[start:start + len(s["breeds"])], breed_embedding, meta["logit_scale"])
        if bp.max() >= meta["breed_min"]:
            breed = s["breeds"][int(bp.argmax())]["name"]
    return s["scientific"], float(species_p[best]), breed


def main():
    photos = sorted(Path(sys.argv[1]).glob("*.jpg"))
    meta = json.loads((ASSETS / "anidex.json").read_text())
    labels = np.fromfile(ASSETS / meta["labels"], dtype="<f4").reshape(meta["label_count"], meta["dim"])
    breed_labels = np.fromfile(ASSETS / meta["breed_labels"], dtype="<f4").reshape(meta["breed_label_count"],
                                                                                    meta["breed_dim"])
    session = ort.InferenceSession(str(ASSETS / meta["encoder"]))
    breed_session = ort.InferenceSession(str(ASSETS / meta["breed_encoder"]))
    OUT.mkdir(parents=True, exist_ok=True)

    png = OUT / "synthetic.png"
    synthetic_png(png)
    x = preprocess_img(Image.open(png).convert("RGB"))
    fixtures = {
        # every 37th value keeps the file small and still covers all three channels
        "preprocess": {"file": png.name, "shape": list(x.shape), "stride": 37,
                       "values": [round(v, 5) for v in x.flatten().tolist()[::37]]},
        "scores": [],
    }
    for p in photos:
        x = preprocess_img(Image.open(p).convert("RGB"))[None].numpy()
        emb = session.run(["embedding"], {"image": x})[0][0]
        breed_emb = breed_session.run(["embedding"], {"image": x})[0][0]
        species, s, breed = score(meta, labels, breed_labels, emb, breed_emb)
        fixtures["scores"].append({"photo": p.stem, "embedding": [round(float(v), 6) for v in emb],
                                   "breed_embedding": [round(float(v), 6) for v in breed_emb],
                                   "species": species, "score": round(s, 5), "breed": breed})
        print(f"{p.stem:24s} {species:26s} {s:.3f} {breed or ''}")
    (OUT / "fixtures.json").write_text(json.dumps(fixtures))


if __name__ == "__main__":
    with torch.no_grad():
        main()
