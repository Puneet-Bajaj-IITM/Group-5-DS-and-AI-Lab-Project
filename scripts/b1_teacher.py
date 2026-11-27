"""B1 (Bolon et al. 2022, the published SnakeCLEF 2021 ViT-L) on a list of photos: the kd teacher.

Fed exactly as notebook 11 fed it (one photo per run, at its own size, read with load_rgb), with its 772
labels mapped to our 184 species the same way. Before the list runs, B1 is scored on photos whose saved
notebook-11 probabilities are known; they must come back (the instrument check), and with the colour
channels swapped they must not (the negative control).

Usage: python b1_teacher.py <data root> <photos.csv with a path column> <saved B1 predictions.npz>
                            <downloads dir> <out.npz>
Writes <out.npz> with paths and probs (photos x 184), plus <out>.check.json; progress is kept in
<out>.parts/ so a restarted run carries on.
"""
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serpentguard.data import load_rgb  # noqa: E402

LINDAT = ("https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/handle/"
          "20.500.12800/1-4773")
MODEL_FILE = "ViT_PreProcessing-ops11-preprocessing-int-dynam_graph.onnx?sequence=7&isAllowed=y"
B1_FILES = {"model": (f"{LINDAT}/{MODEL_FILE}", "838ae0f8ed19ffeee89aba5fa8b50956"),
            "labels": (f"{LINDAT}/min-train_metadata.csv?sequence=2&isAllowed=y",
                       "52ec43c8595a3835da6aa16364aff75e")}
SYNONYMS = {  # notebook 11's older names -> the name in our classes.csv
    "Xenochrophis piscator": "Fowlea piscator",
    **{f"Trimeresurus {e}": f"Craspedocephalus {e}"
       for e in ("malabaricus", "gramineus", "anamallensis", "travancoricus", "macrolepis")},
    "Macropisthodon plumbicolor": "Rhabdophis plumbicolor",
    "Ramphotyphlops braminus": "Indotyphlops braminus",
}
CHECK_PHOTOS = 64
CHECK_TOLERANCE = 2e-3  # largest probability difference from notebook 11's saved run
PART_PHOTOS = 2000
WORKERS = 3


def download(url, path, md5):
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out, 2 ** 24)
        partial.rename(path)
    digest = hashlib.md5()
    with open(path, "rb") as stored:
        for chunk in iter(lambda: stored.read(2 ** 24), b""):
            digest.update(chunk)
    assert digest.hexdigest() == md5, f"{path.name}: MD5 {digest.hexdigest()}, expected {md5}"
    return path


def normalise(name):
    return " ".join(str(name).replace("_", " ").split()).lower()


def label_map(labels_file, n_outputs, our_species):
    meta = pd.read_csv(labels_file, low_memory=False)
    pairs = meta[["class_id", "binomial"]].dropna().drop_duplicates()
    names = [None] * n_outputs
    for class_id, name in zip(pairs["class_id"].astype(int), pairs["binomial"].astype(str)):
        names[class_id] = name
    ours = {normalise(name): i for i, name in enumerate(our_species)}
    renames = {normalise(old): normalise(new) for old, new in SYNONYMS.items()}
    index_map = np.full(n_outputs, -1)
    for i, name in enumerate(names):
        key = None if name is None else normalise(name)
        if key in ours:
            index_map[i] = ours[key]
        elif renames.get(key) in ours:
            index_map[i] = ours[renames[key]]
    return index_map


class Photos(Dataset):
    def __init__(self, files, dtype, swap_channels=False):
        self.files, self.dtype, self.swap = list(files), dtype, swap_channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        pixels = load_rgb(self.files[i]).astype(np.float32)
        pixels = pixels[..., ::-1] if self.swap else pixels
        return np.ascontiguousarray(pixels.astype(self.dtype))


class B1:
    def __init__(self, downloads, n_species):
        import onnxruntime
        (model_url, model_md5), (labels_url, labels_md5) = B1_FILES["model"], B1_FILES["labels"]
        model = download(model_url, downloads / "bolon_vit.onnx", model_md5)
        labels = download(labels_url, downloads / "snakeclef_min_train_metadata.csv", labels_md5)
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in onnxruntime.get_available_providers()]
        self.session = onnxruntime.InferenceSession(str(model), providers=providers)
        self.providers = self.session.get_providers()
        assert "CUDAExecutionProvider" in self.providers, f"B1 would run on {self.providers}"
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.n_outputs = self.output.shape[-1]
        self.dtype = {"tensor(uint8)": np.uint8, "tensor(int32)": np.int32,
                      "tensor(float)": np.float32}[self.input.type]
        self.labels_file, self.n_species = labels, n_species

    def probs(self, files, swap_channels=False, report_every=None):
        loader = DataLoader(Photos(files, self.dtype, swap_channels), batch_size=None,
                            num_workers=WORKERS)
        scores, started = [], time.perf_counter()
        for item in loader:
            raw = self.session.run([self.output.name], {self.input.name: item.numpy()})[0]
            scores.append(np.asarray(raw).reshape(-1, self.n_outputs))
            if report_every and len(scores) % report_every == 0:
                print(f"B1: {len(scores):,}/{len(files):,} photos, "
                      f"{len(scores) / (time.perf_counter() - started):.1f}/s", flush=True)
        scores = np.concatenate(scores).astype(np.float64)
        if not ((scores >= 0).all() and np.allclose(scores.sum(axis=1), 1, atol=1e-3)):
            scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            scores /= scores.sum(axis=1, keepdims=True)
        mapped = self.index_map >= 0
        probs = scores[:, mapped] @ np.eye(self.n_species)[self.index_map[mapped]]
        flat = np.zeros(self.n_species)
        flat[sorted(set(self.index_map[mapped]))] = 1
        probs[probs.sum(axis=1) <= 0] = flat
        return (probs / probs.sum(axis=1, keepdims=True)).astype(np.float32)


def main():
    data_root, photos_csv, saved_file, downloads, out = map(Path, sys.argv[1:6])
    downloads.mkdir(parents=True, exist_ok=True)
    species = pd.read_csv(data_root / "classes.csv").sort_values("class_id")["species"].tolist()
    b1 = B1(downloads, len(species))
    b1.index_map = label_map(b1.labels_file, b1.n_outputs, species)
    print(f"B1 on {b1.providers}; {int((b1.index_map >= 0).sum())} of {b1.n_outputs} labels mapped",
          flush=True)

    saved = np.load(saved_file)
    pick = np.random.default_rng(42).choice(len(saved["paths"]), CHECK_PHOTOS, replace=False)
    files = [str(data_root / "data" / p) for p in saved["paths"][pick]]
    again, swapped = b1.probs(files), b1.probs(files, swap_channels=True)
    check = {"photos": CHECK_PHOTOS,
             "max_abs_diff": float(np.abs(again - saved["probs"][pick]).max()),
             "top1_agreement": float((again.argmax(1) == saved["probs"][pick].argmax(1)).mean()),
             "swapped_channels_max_abs_diff": float(np.abs(swapped - saved["probs"][pick]).max())}
    Path(f"{out}.check.json").write_text(json.dumps(check, indent=1))
    print(f"B1 instrument check: {check}", flush=True)
    assert check["max_abs_diff"] < CHECK_TOLERANCE, "B1 no longer gives notebook 11's probabilities"
    assert check["swapped_channels_max_abs_diff"] > CHECK_TOLERANCE, "the control must differ"

    paths = pd.read_csv(photos_csv)["path"].astype(str).tolist()
    parts = Path(f"{out}.parts")
    parts.mkdir(exist_ok=True)
    started = time.perf_counter()
    for start in range(0, len(paths), PART_PHOTOS):
        part = parts / f"{start:06d}.npz"
        if part.exists():
            continue
        chunk = paths[start:start + PART_PHOTOS]
        probs = b1.probs([str(data_root / "data" / p) for p in chunk])
        np.savez(part, paths=np.array(chunk), probs=probs)
        done = start + len(chunk)
        print(f"B1 teacher: {done:,}/{len(paths):,} photos, "
              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    merged = [np.load(p) for p in sorted(parts.glob("*.npz"))]
    np.savez(out, paths=np.concatenate([m["paths"] for m in merged]),
             probs=np.concatenate([m["probs"] for m in merged]))
    print(f"B1 teacher written: {out}", flush=True)


if __name__ == "__main__":
    main()
