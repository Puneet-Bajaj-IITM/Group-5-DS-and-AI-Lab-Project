"""Laptop CPU time per frame: M3's final model, M4_H2 with one look, M4_H2 with the detail crop, same frames."""
import os, platform, statistics, sys, time
from pathlib import Path
import torch
W = Path(os.environ.get("SERPENTGUARD_WORK", "."))  # the working folder that holds the weights and m4-checks
sys.path.insert(0, str(W / "m4-checks"))
from test_detail import snake_photos
from serpentguard.train import load_checkpoint
WARMUP, RUNS, ROUNDS = 5, 30, 2
frames = snake_photos()  # six real snake photos, 320 x 320
_, m3, _ = load_checkpoint(W / "m3-video-demo/models")
_, h2, _ = load_checkpoint(W / "kaggle/m4_runs/heads/out_h2_full/M4_H2")
assert h2.detail_crop and not m3.detail_crop


def median_ms(model, detail):
    model.detail_crop = detail
    times = []
    with torch.inference_mode():
        for i in range(WARMUP + RUNS):
            frame = frames[i % len(frames)][None]
            started = time.perf_counter()
            model(frame)
            if i >= WARMUP:
                times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


results = {}
for r in range(ROUNDS):
    for name, model, detail in (("M3 final, one look", m3, False), ("M4_H2, one look", h2, False),
                                ("M4_H2, with the detail crop", h2, True)):
        results.setdefault(name, []).append(median_ms(model, detail))
print(f"{platform.machine()} {platform.processor()}, torch {torch.__version__}, {torch.get_num_threads()} threads, "
      f"{len(frames)} photos cycled, {RUNS} timed runs after {WARMUP} warm-up, {ROUNDS} rounds")
for name, values in results.items():
    print(f"{name:30} " + "  ".join(f"{v:6.1f}" for v in values) + f"   median of rounds {statistics.median(values):.1f} ms")
