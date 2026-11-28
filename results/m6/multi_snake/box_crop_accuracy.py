"""Does naming a snake from its detector box crop work as well as from the U-Net mask crop?

On 600 validation snake photos (seed 42), the frozen M5 classifier (mirror + prior) is run on the frozen crop (mask
box) and on the detector's best box padded the same way. Top-1 against the label for each.
"""
import json, random, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "m5-app/app/tool/m5"))
import reference_m5 as m5
ref = m5.ref
SAMPLE, SEED = 600, 42


def main():
    p = m5.AppPipeline(); s = p.settings; size = s["input_size"]
    species = p.tables["species"]
    folder = {sp.lower().replace(" ", "_"): sp for sp in species}
    photos = [x for x in sorted((WORK / "m5-g4-probe/photos/snakes/val").rglob("*.jpg")) if x.parent.name in folder]
    photos = random.Random(SEED).sample(photos, SAMPLE)
    hits = {"mask_crop": 0, "box_crop": 0}; both = n = 0
    for path in photos:
        rgb = ref.load_photo(path, s["exposure_fix"]); h, w = rgb.shape[:2]
        gate, mask = p.gate(rgb)
        out = p.detector.run(None, {"images": ref.detector_input(rgb, size, s["detector"]["pad_value"])})[0][0]
        k = int(np.argmax(out[4])); cx, cy, bw, bh = out[:4, k]
        ratio = min(size / h, size / w); nw, nh = max(ref.py_round(w * ratio), 1), max(ref.py_round(h * ratio), 1)
        left, top = ref.py_round((size - nw) / 2 - 0.1), ref.py_round((size - nh) / 2 - 0.1)
        x0, y0 = (cx - bw/2 - left) / ratio, (cy - bh/2 - top) / ratio
        x1, y1 = (cx + bw/2 - left) / ratio, (cy + bh/2 - top) / ratio
        box = np.zeros((h, w), np.uint8)
        box[max(int(y0), 0):max(int(np.ceil(y1)), 0), max(int(x0), 0):max(int(np.ceil(x1)), 0)] = 1
        truth = folder[path.parent.name]; n += 1; got = {}
        for name, m in (("mask_crop", ref.clean_mask(mask, s["mask_cleanup"])), ("box_crop", box)):
            win = ref.crop_window(m, h, w, s["crop"]["padding"])
            crop = rgb if win is None else rgb[win[0]:win[1], win[2]:win[3]]
            top1 = ref.verdict(p.classify(crop), p.thresholds, p.tables)["top3"][0]["species"]
            got[name] = top1 == truth; hits[name] += got[name]
        both += got["mask_crop"] == got["box_crop"]
    summary = {"photos": n, "top1_mask_crop": hits["mask_crop"] / n, "top1_box_crop": hits["box_crop"] / n,
               "same_right_or_wrong": both / n}
    (HERE / "box_crop_accuracy.json").write_text(json.dumps(summary, indent=1)); print(summary)


if __name__ == "__main__":
    main()
