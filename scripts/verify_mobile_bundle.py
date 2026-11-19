"""Verify a supplied, untracked ONNX bundle against its checksums and goldens."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def compare(actual, expected, tolerance, path="result"):
    if isinstance(expected, dict):
        return max((compare(actual[key], value, tolerance, f"{path}.{key}")
                    for key, value in expected.items()), default=0.0)
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        return max((compare(a, b, tolerance, f"{path}[{i}]")
                    for i, (a, b) in enumerate(zip(actual, expected))), default=0.0)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        gap = abs(actual - expected)
        assert math.isfinite(actual) and gap <= tolerance, (path, actual, expected, tolerance)
        return gap
    assert actual == expected, (path, actual, expected)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photo-cache", type=Path, required=True)
    args = parser.parse_args()
    bundle, models = args.bundle.resolve(), args.bundle.resolve() / "models"
    hashes = {}
    for line in (models / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (models / name.strip()).resolve()
        assert path.is_relative_to(models), "Unsafe checksum path"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Checksum mismatch: {path.name}"
        hashes[path.name] = {"sha256": actual, "bytes": path.stat().st_size}

    spec = importlib.util.spec_from_file_location("supplied_reference", bundle / "reference_pipeline.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    pipeline = reference.Pipeline(models)
    golden = json.loads((bundle / "golden_decisions.json").read_text())
    decision_gap, control_failures = 0.0, 0
    for row in golden["rows"]:
        outputs = {key: reference.np.asarray(value) for key, value in row["outputs"].items()}
        result = reference.verdict(outputs, golden["thresholds"], pipeline.tables)
        decision_gap = max(decision_gap, compare(result, row["expected"], 1e-6))
        altered = {**outputs, "species_logits": reference.np.roll(outputs["species_logits"], 1)}
        try:
            compare(reference.verdict(altered, golden["thresholds"], pipeline.tables), row["expected"], 1e-6)
        except AssertionError:
            control_failures += 1
    assert control_failures > 0, "Negative control did not detect changed species outputs"
    print(f"Decision fixtures passed: {len(golden['rows'])}; negative control failures: {control_failures}", flush=True)

    args.photo_cache.mkdir(parents=True, exist_ok=True)
    photo_goldens = json.loads((bundle / "golden_photos.json").read_text())
    photos = []
    for row in photo_goldens["photos"]:
        path = args.photo_cache / f"{row['photo_id']}.jpg"
        if not path.exists() or hashlib.sha1(path.read_bytes()).hexdigest() != row["sha1"]:
            assert row["url"].startswith("https://"), "Photo URL must use HTTPS"
            request = urllib.request.Request(row["url"], headers={"User-Agent": "SerpentGuard-course-verification/1.0"})
            body = urllib.request.urlopen(request, timeout=60).read()
            assert hashlib.sha1(body).hexdigest() == row["sha1"], "Photo checksum mismatch"
            path.write_bytes(body)
        start = time.perf_counter()
        result = pipeline.answer(path)
        elapsed = time.perf_counter() - start
        gap = compare(result, row["answer"], photo_goldens["tolerance"])
        expected_window = row["steps"].get("crop_window_top_bottom_left_right")
        actual_window = result.get("crop_window")
        assert (list(actual_window) if actual_window is not None else None) == expected_window
        photos.append({"photo_id": row["photo_id"], "sha1": row["sha1"], "verdict": result["verdict"],
                       "gate": result["gate"]["snake"], "max_numeric_gap": gap,
                       "reference_seconds_this_machine": elapsed})
        print(f"Photo {row['photo_id']}: passed ({result['verdict']})", flush=True)
    report = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "onnxruntime", "opencv-python-headless", "Pillow"]},
        "checksums": hashes, "decision_fixtures": {"passed": len(golden["rows"]), "total": len(golden["rows"]),
            "max_numeric_gap": decision_gap, "negative_control_failures": control_failures},
        "photo_fixtures": {"passed": len(photos), "total": len(photos), "tolerance": photo_goldens["tolerance"], "photos": photos},
        "scope": "Supplied Python/FP32 ONNX reference only; not Android integration or a reproduced 300-photo PyTorch parity study",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
