"""One Kaggle kernel folder per variant (plain, distil, shuffled, distil_oi), all private, GPU T4, on Vikas's account.

Usage: python make_kernels.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = "vikas25s"
OUR_DATASET = f"{OWNER}/sg-m5-g4-train"
SNAKE_DATASET = "puneetbajajiitm/serpentguard-indian-snakes-clean"
VARIANTS = ("plain", "distil", "shuffled", "distil_oi", "distil_oi_seed43", "distil_oi_v2")


def main():
    source = (HERE / "train_g4.py").read_text()
    for variant in VARIANTS:
        folder = HERE / "runs" / variant
        folder.mkdir(parents=True, exist_ok=True)
        slug = f"sg-m5-g4-{variant}"
        (folder / f"{slug}.py").write_text(source.replace("__VARIANT__", variant)
                                           .replace("__OUR_DATASET__", OUR_DATASET))
        meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [SNAKE_DATASET, OUR_DATASET], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(folder)


if __name__ == "__main__":
    main()
