"""Prepare a blinded photo and response package for the SerpentGuard expert audit."""

import argparse
import csv
import random
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageOps


REQUIRED = {"split", "path", "obs_id", "observer_id", "species", "tier", "who_category"}


def select_validation_rows(rows, count, seed):
    if count < 1:
        raise ValueError("The requested sample must be positive")
    eligible = sorted((row for row in rows if row["split"] == "val"), key=lambda row: row["path"])
    if any(not all(row[field].strip() for field in REQUIRED) for row in eligible):
        raise ValueError("Every validation row needs a complete path, label and observation record")
    if len({row["path"] for row in eligible}) != len(eligible):
        raise ValueError("Validation paths must be unique")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    unique_observations = {}
    for row in eligible:
        unique_observations.setdefault(row["obs_id"], row)
    if count > len(unique_observations):
        raise ValueError(f"Only {len(unique_observations)} unique validation observations are available")

    strata = defaultdict(list)
    for row in unique_observations.values():
        strata[row["tier"], row["who_category"]].append(row)
    quotas = {
        group: Fraction(count * len(photos), len(unique_observations))
        for group, photos in strata.items()
    }
    allocation = {group: int(quota) for group, quota in quotas.items()}
    left = count - sum(allocation.values())
    remainder_order = sorted(strata, key=lambda group: (allocation[group] - quotas[group], group))
    for group in remainder_order[:left]:
        allocation[group] += 1
    sample = [row for group in sorted(strata) for row in strata[group][:allocation[group]]]
    rng.shuffle(sample)
    return sample


def write_csv(path, fieldnames, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare(metadata, dataset_root, output, *, count=200, seed=42):
    with metadata.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError("metadata.csv lacks required project columns")
        sample = select_validation_rows(reader, count, seed)
    root = dataset_root.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix=".audit-build-", dir=output.parent) as temporary:
        staging = Path(temporary) / "package"
        images = staging / "review" / "images"
        private = staging / "private"
        images.mkdir(parents=True)
        private.mkdir()
        responses, key = [], []
        for index, row in enumerate(sample, 1):
            source = (root / row["path"]).resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("A selected photo path is outside the dataset root or is not a file")
            audit_id = f"A{index:04d}"
            with Image.open(source) as image:
                photo = ImageOps.exif_transpose(image).convert("RGB")
                photo.info.clear()
                photo.save(images / f"{audit_id}.jpg", quality=95)
            responses.append({"audit_id": audit_id, "image": f"images/{audit_id}.jpg"})
            key.append({"audit_id": audit_id, **{field: row[field] for field in sorted(REQUIRED)}})
        write_csv(staging / "review" / "responses.csv",
                  ["audit_id", "image", "decision", "taxon_label", "notes"], responses)
        write_csv(private / "answer-key.csv", ["audit_id", *sorted(REQUIRED)], key)
        staging.rename(output)
    return len(sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        size = prepare(args.metadata, args.dataset_root, args.output, count=args.count, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Prepared {size} validation photos. Review sheets and source labels are in separate folders.")


if __name__ == "__main__":
    main()
