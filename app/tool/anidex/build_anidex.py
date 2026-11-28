"""Builds the AniDex assets the app bundles under assets/anidex/.

    python tool/anidex/build_anidex.py [--model v1|v2] [--weights fp16|int8] [--skip-cards] [--skip-encoder]

Writes:
  anidex_encoder.onnx        BioCLIP 2's image encoder (species): a 1x3x224x224 photo in, an L2-normalised
                             embedding out (not committed, like the other models: *.onnx is in .gitignore)
  anidex_labels.bin          its text labels' embeddings, float32, row-major [labels x dim]
  anidex_breed_encoder.onnx  a general CLIP's image encoder (breeds), same input (not committed)
  anidex_breed_labels.bin    the breeds' text embeddings in that model's space, float32 [breeds x breed_dim]
  anidex.json                species, breeds, which label belongs to which species, preprocessing, and the card
                             text (Wikipedia summaries, GBIF taxonomy and IUCN status) so cards work offline

Species: each species has several BioCLIP 2 labels (its Tree of Life text embedding, a plain-English prompt
and one prompt per breed); the app scores a photo against every label and adds up each species' scores.
Breed: BioCLIP 2 was trained on taxonomy, which stops at species, and it named 6 of 28 breed photos right;
a general CLIP (ViT-B/16 trained on DataComp) named 19 of 28, so the breed comes from that model instead,
scored against the winning species' breeds only. Both encoders take the same stretched 224 px input.
Needs pybioclip, onnx, onnxruntime and onnxconverter-common; the first run downloads both models.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import onnx
import onnxruntime as ort
import open_clip
import requests
import torch
import torch.nn.functional as F
from bioclip import TreeOfLifeClassifier
from bioclip._constants import BIOCLIP_V1_MODEL_STR, BIOCLIP_V2_MODEL_STR

sys.path.insert(0, str(Path(__file__).parent))
import animals  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "assets" / "anidex"
MODELS = {"v1": BIOCLIP_V1_MODEL_STR, "v2": BIOCLIP_V2_MODEL_STR}
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
HEADERS = {"User-Agent": "serpentguard-anidex-build/1.0 (student project)"}
SPECIES = [row[0] for row in animals.COMMON]
BREED_MODEL = ("ViT-B-16", "datacomp_xl_s13b_b90k")


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return F.normalize(self.model.encode_image(image), dim=-1)


@torch.no_grad()
def build_labels(clf):
    tokenizer = open_clip.get_tokenizer(clf.model_str)

    def encode(prompts):
        return F.normalize(clf.model.encode_text(tokenizer(prompts)), dim=-1)

    rows = {}
    for i, names in enumerate(clf.txt_names):
        rows.setdefault(names[0][-2] + " " + names[0][-1], []).append(i)
    missing = [s for s in SPECIES if s not in rows]
    if missing:
        raise SystemExit(f"Not in this model's Tree of Life labels: {missing}")
    taxonomy = {s: dict(zip(RANKS, clf.txt_names[rows[s][0]][0][:6])) for s in SPECIES}
    tol = F.normalize(torch.stack([clf.txt_embeddings[:, rows[s]].mean(1) for s in SPECIES]), dim=-1)
    common = torch.stack([
        F.normalize(encode([f"a photo of a {n}.", f"a photo of a {w}.", f"a photo of {s}, a {n}."]).mean(0), dim=-1)
        for s, n, _, w in animals.COMMON])
    labels, owner, breeds = [tol, common], list(range(len(SPECIES))) * 2, {}
    for species, (noun, names) in animals.BREEDS.items():
        start = len(owner)
        labels.append(encode([f"a photo of a {b}, a type of {noun}." for b, _ in names]))
        owner += [SPECIES.index(species)] * len(names)
        breeds[species] = (noun, start, [b for b, _ in names], [w or b for b, w in names])
    return torch.cat(labels).float(), owner, breeds, taxonomy


@torch.no_grad()
def build_breed_labels(model, tokenizer):
    """Each breed's text embedding in the general CLIP's space, and where each species' breeds start."""

    def name(b, noun):
        return b if noun in b.lower() else f"{b} {noun}"

    rows, starts = [], {}
    for species, (noun, names) in animals.BREEDS.items():
        starts[species] = sum(len(r) for r in rows)
        prompts = [[f"a photo of a {name(b, noun)}.", f"a photo of a {b}, a type of {noun}."] for b, _ in names]
        rows.append(torch.stack([F.normalize(F.normalize(model.encode_text(tokenizer(p)), dim=-1).mean(0), dim=-1)
                                 for p in prompts]))
    return torch.cat(rows).float(), starts


def export_encoder(model, path: Path, weights: str):
    raw = path.with_suffix(".fp32.onnx")
    torch.onnx.export(ImageEncoder(model.eval()), torch.randn(1, 3, 224, 224), str(raw),
                      input_names=["image"], output_names=["embedding"], opset_version=17, dynamo=False)
    if weights == "fp16":
        from onnxconverter_common import float16
        model = float16.convert_float_to_float16(onnx.load(str(raw)), keep_io_types=True)
        onnx.save(model, str(path))
    else:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(raw), str(path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    raw.unlink()
    for extra in path.parent.glob(path.name + ".data"):
        raise SystemExit(f"{extra} was written: the model must be a single file")


def get_json(url, params=None):
    for attempt in range(6):
        time.sleep(0.25)  # stay well under Wikipedia's and GBIF's rate limits
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return {}
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 0)) or 5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    print(f"  gave up on {url}", flush=True)
    return {}


def wiki(title):
    d = get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_")),
                 {"redirect": "true"})
    if not d.get("extract") or d.get("type") == "disambiguation":
        return {}
    return {"extract": d["extract"], "image": (d.get("thumbnail") or {}).get("source", ""),
            "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page", "")}


def card_data(species, group, wiki_title, has_breeds):
    m = get_json("https://api.gbif.org/v1/species/match", {"name": species, "kingdom": "Animalia"})
    matched = m.get("matchType") not in (None, "NONE")
    key = (m.get("acceptedUsageKey") or m.get("usageKey")) if matched else None
    out = {"wiki": wiki(wiki_title), "gbif_key": key,
           "taxonomy": {r: m[r] for r in RANKS if matched and m.get(r)}, "iucn": None, "subspecies": []}
    if key and group not in (animals.PET, animals.FARM):
        out["iucn"] = get_json(f"https://api.gbif.org/v1/species/{key}/iucnRedListCategory").get("code")
    if key and not has_breeds:
        kids = get_json(f"https://api.gbif.org/v1/species/{key}/children", {"limit": 200}).get("results", [])
        out["subspecies"] = sorted({k.get("canonicalName") or k["scientificName"]
                                    for k in kids if k.get("rank") == "SUBSPECIES"})
    return out


def check(encoder_path: Path, breed_path: Path, clf, breed_model, labels, owner, photos):
    """The exported encoders must give the same answers as PyTorch."""
    session, breed_session = ort.InferenceSession(str(encoder_path)), ort.InferenceSession(str(breed_path))
    owner_t = torch.tensor(owner)
    worst, worst_breed, changed = 1.0, 1.0, []
    for p in photos:
        x = clf.preprocess(clf.ensure_rgb_image(str(p)))[None]
        with torch.no_grad():
            want = F.normalize(clf.model.encode_image(x), dim=-1)[0]
            want_breed = F.normalize(breed_model.encode_image(x), dim=-1)[0]
        got = torch.from_numpy(session.run(["embedding"], {"image": x.numpy()})[0][0])
        got_breed = torch.from_numpy(breed_session.run(["embedding"], {"image": x.numpy()})[0][0])
        worst = min(worst, F.cosine_similarity(want, got, dim=0).item())
        worst_breed = min(worst_breed, F.cosine_similarity(want_breed, got_breed, dim=0).item())
        a, b = [torch.zeros(len(SPECIES)).index_add_(0, owner_t, (100 * e @ labels.T).softmax(-1)).argmax().item()
                for e in (want, got)]
        if a != b:
            changed.append((p.name, SPECIES[a], SPECIES[b]))
    print(f"ONNX vs PyTorch on {len(photos)} photos: lowest cosine {worst:.4f} (species), {worst_breed:.4f} (breed); "
          f"species changed: {changed or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="v2")
    ap.add_argument("--weights", choices=["fp16", "int8"], default="int8")
    ap.add_argument("--skip-cards", action="store_true",
                    help="reuse the card text in the existing anidex.json, fetching only what is missing")
    ap.add_argument("--skip-encoder", action="store_true", help="keep the existing exported encoders")
    ap.add_argument("--check-photos", type=Path, help="folder of .jpg photos to compare ONNX with PyTorch on")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    clf = TreeOfLifeClassifier(model_str=MODELS[args.model])
    labels, owner, breeds, tol_taxonomy = build_labels(clf)
    clf.txt_embeddings = clf.txt_names = None
    labels_path = OUT / "anidex_labels.bin"
    labels.numpy().astype("<f4").tofile(labels_path)

    encoder_path = OUT / "anidex_encoder.onnx"
    if not (args.skip_encoder and encoder_path.exists()):
        export_encoder(clf.model, encoder_path, args.weights)

    breed_model = open_clip.create_model_and_transforms(BREED_MODEL[0], pretrained=BREED_MODEL[1])[0].eval()
    breed_labels, breed_starts = build_breed_labels(breed_model, open_clip.get_tokenizer(BREED_MODEL[0]))
    breed_labels_path = OUT / "anidex_breed_labels.bin"
    breed_labels.numpy().astype("<f4").tofile(breed_labels_path)
    breed_path = OUT / "anidex_breed_encoder.onnx"
    if not (args.skip_encoder and breed_path.exists()):
        export_encoder(breed_model, breed_path, args.weights)
    if args.check_photos:
        check(encoder_path, breed_path, clf, breed_model, labels, owner, sorted(args.check_photos.glob("*.jpg")))

    old = {}
    if args.skip_cards and (OUT / "anidex.json").exists():
        old = {s["scientific"]: s for s in json.loads((OUT / "anidex.json").read_text())["species"]}
    species_out = []
    for i, (s, name, group, wiki_title) in enumerate(animals.COMMON):
        if s in old:
            card = {k: old[s][k] for k in ("wiki", "gbif_key", "taxonomy", "iucn", "subspecies")}
            breed_cards = {b["name"]: b["wiki"] for b in old[s].get("breeds", [])}
            if not card["wiki"]:
                print(f"refetch: {name}", flush=True)
                card["wiki"] = wiki(wiki_title)
            if s in breeds:
                for b, w in zip(breeds[s][2], breeds[s][3]):
                    if not breed_cards.get(b):
                        print(f"refetch: {b}", flush=True)
                        breed_cards[b] = wiki(w)
        else:
            print(f"cards {i + 1}/{len(SPECIES)}: {name}", flush=True)
            card = card_data(s, group, wiki_title, s in breeds)
            breed_cards = {b: wiki(w) for b, w in zip(breeds[s][2], breeds[s][3])} if s in breeds else {}
        card["taxonomy"] = card["taxonomy"] or tol_taxonomy[s]
        row = {"no": i + 1, "scientific": s, "name": name, "group": group, **card}
        if s in breeds:
            noun, start, names, _ = breeds[s]
            row["breed_noun"], row["breed_labels_start"] = noun, start
            row["breed_vectors_start"] = breed_starts[s]
            row["breeds"] = [{"name": b, "wiki": breed_cards.get(b, {})} for b in names]
        species_out.append(row)

    meta = {
        "model": MODELS[args.model],
        "weights": args.weights,
        "encoder": encoder_path.name,
        "encoder_sha256": hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
        "labels": labels_path.name,
        "label_count": labels.shape[0],
        "dim": labels.shape[1],
        "breed_model": f"open_clip {BREED_MODEL[0]} {BREED_MODEL[1]}",
        "breed_encoder": breed_path.name,
        "breed_encoder_sha256": hashlib.sha256(breed_path.read_bytes()).hexdigest(),
        "breed_labels": breed_labels_path.name,
        "breed_label_count": breed_labels.shape[0],
        "breed_dim": breed_labels.shape[1],
        "input_size": 224,
        "resize": "stretch",
        "mean": MEAN,
        "std": STD,
        "logit_scale": 100.0,
        "low_confidence": 0.5,
        "breed_min": 0.4,
        "label_owner": owner,
        "groups": [animals.PET, animals.FARM, animals.BIRD, animals.WILD, animals.REPTILE, animals.WATER, animals.BUG],
        "built": time.strftime("%Y-%m-%d"),
        "species": species_out,
    }
    (OUT / "anidex.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"Wrote {encoder_path.name} ({encoder_path.stat().st_size / 1e6:.0f} MB), {labels_path.name} "
          f"({labels.shape[0]} x {labels.shape[1]}), {breed_path.name} ({breed_path.stat().st_size / 1e6:.0f} MB), "
          f"{breed_labels_path.name} ({breed_labels.shape[0]} x {breed_labels.shape[1]}), "
          f"anidex.json ({len(species_out)} species, {sum(len(r.get('breeds', [])) for r in species_out)} breeds).")


if __name__ == "__main__":
    main()
