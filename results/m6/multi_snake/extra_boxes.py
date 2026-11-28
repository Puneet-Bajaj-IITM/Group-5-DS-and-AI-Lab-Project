"""How often does the detector give a second snake box on one-snake validation photos?

Same rules as the app's detectAll (score >= 0.25, NMS at IoU 0.5) on the M4 detector, run on 1,000 validation
snake photos (seed 42) and the validation non-snakes. Writes per-photo box counts and boxes for a look by eye.
"""
import json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
# The working folder that holds the M4 reference pipeline, the app tools and the validation photos.
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "m4-final/for-jayesh"))
import reference_pipeline as ref

MIN_SCORES = (0.25, 0.4, 0.5)
IOU_LIMIT, SAMPLE, SEED = 0.5, 1000, 42


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    i = w * h
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def boxes(out, width, height, size, snake_class, min_score):
    out = out[0]; anchors = out.shape[1]; scores = out[4 + snake_class]
    ratio = min(size / height, size / width)
    nw, nh = max(ref.py_round(width * ratio), 1), max(ref.py_round(height * ratio), 1)
    left, top = ref.py_round((size - nw) / 2 - 0.1), ref.py_round((size - nh) / 2 - 0.1)
    cand = []
    for i in np.where(scores >= min_score)[0]:
        cx, cy, w, h = out[0, i], out[1, i], out[2, i], out[3, i]
        b = [np.clip((cx - w/2 - left)/ratio, 0, width), np.clip((cy - h/2 - top)/ratio, 0, height),
             np.clip((cx + w/2 - left)/ratio, 0, width), np.clip((cy + h/2 - top)/ratio, 0, height)]
        if b[2] > b[0] and b[3] > b[1]: cand.append((float(scores[i]), [float(v) for v in b]))
    cand.sort(key=lambda c: -c[0]); kept = []
    for c in cand:
        if all(iou(k[1], c[1]) < IOU_LIMIT for k in kept): kept.append(c)
    return kept


def main():
    p = ref.Pipeline(WORK / "m5-results/g3/app_models_A3")
    s = p.settings; size = s["input_size"]
    sets = {"snakes_val": sorted((WORK / "m5-g4-probe/photos/snakes/val").rglob("*.jpg")),
            "nonsnakes_val": sorted((WORK / "m5-g4-probe/photos/val_nonsnake").rglob("*.jpg"))}
    print({k: len(v) for k, v in sets.items()})
    summary, rows = {}, []
    for name, photos in sets.items():
        photos = random.Random(SEED).sample(photos, min(SAMPLE, len(photos)))
        counts = {m: [] for m in MIN_SCORES}
        for path in photos:
            rgb = ref.load_photo(path, s["exposure_fix"])
            out = p.detector.run(None, {"images": ref.detector_input(rgb, size, s["detector"]["pad_value"])})[0]
            for m in MIN_SCORES:
                k = boxes(out, rgb.shape[1], rgb.shape[0], size, s["detector"]["snake_class"], m)
                counts[m].append(len(k))
                if m == MIN_SCORES[0]:
                    rows.append({"set": name, "photo": str(path), "boxes": k})
        summary[name] = {str(m): {"photos": len(c), "0 boxes": c.count(0), "1 box": c.count(1),
                                  "2+ boxes": sum(x >= 2 for x in c)} for m, c in counts.items()}
    (HERE / "extra_boxes.json").write_text(json.dumps(summary, indent=1))
    (HERE / "extra_boxes_rows.json").write_text(json.dumps(rows))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
