"""Bundle the repo code and the extra kernel into one Kaggle script kernel per account (push_<account>/).

Each bundle: m5-wt's src/ and data/, m5_extra/corrupt.py, this folder's kernel modules, and job.json (that
account's share of the conditions plus Grad-CAM or int8). Unpacked under /kaggle/tmp/repo and run from there.

Usage: m3-final/.venv/bin/python -I m5-results/m5_extra/kernel/build_kernel.py
"""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA = HERE.parent
WORK = HERE.parents[2]
REPO = WORK / "m5-wt"
FROM_REPO = ("src", "data")
FROM_HERE = ("extra_kernel.py", "gradcam_step.py", "compression_step.py")
DATASET_TITLE = "serpentguard-m5-extra-inputs"
CLEAN_DATASET = "puneetbajajiitm/serpentguard-indian-snakes-clean"
TWO_T4 = "NvidiaTeslaT4"
JOBS = {
    "jayeshbansal497": {"conditions": ["blur_s1", "blur_s2", "blur_s3", "noise_s1", "noise_s2", "noise_s3",
                                       "jpeg_q30", "jpeg_q10", "rotate_15", "rotate_30"],
                        "gradcam": True, "compression": False},
    # v2 ran the 8 corruptions below and failed only in compression (onnxruntime wrote into /kaggle/input);
    # v3 reruns the clean pass and compression only.
    "vikas25s": {"conditions": [],
                 "gradcam": False, "compression": True},
}

RUNNER = '''import base64, io, subprocess, sys, zipfile
from pathlib import Path

repo = Path("/kaggle/tmp/repo")
zipfile.ZipFile(io.BytesIO(base64.b64decode("{bundle}"))).extractall(repo)
code = subprocess.run([sys.executable, "-u", str(repo / "extra_kernel.py")], cwd=repo).returncode
raise SystemExit(code)
'''


def bundle(job):
    buffer, files = io.BytesIO(), {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        entries = [(REPO / e, REPO) for e in FROM_REPO] + [(HERE / e, HERE) for e in FROM_HERE]
        entries.append((EXTRA / "corrupt.py", EXTRA))
        for entry, base in entries:
            for path in sorted(entry.rglob("*")) if entry.is_dir() else [entry]:
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".png", ".jpg"):
                    name = path.relative_to(base).as_posix()
                    archive.write(path, name)
                    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        text = json.dumps(job, indent=1) + "\n"
        archive.writestr("job.json", text)
        files["job.json"] = hashlib.sha256(text.encode()).hexdigest()
    return base64.b64encode(buffer.getvalue()).decode(), files


def main():
    for account, job in JOBS.items():
        job = {**job, "inputs_dataset": f"{account}/{DATASET_TITLE}", "account": account}
        encoded, files = bundle(job)
        push = HERE / f"push_{account}"
        push.mkdir(exist_ok=True)
        (push / "runner.py").write_text(RUNNER.format(bundle=encoded))
        kernel_id = f"{account}/sg-m5-extra"
        meta = {"id": kernel_id, "title": "sg-m5-extra", "code_file": "runner.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "machine_shape": TWO_T4,
                "dataset_sources": [CLEAN_DATASET, job["inputs_dataset"]],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (push / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        (push / "bundle_files.json").write_text(json.dumps(files, indent=1) + "\n")
        print(f"{kernel_id}: {len(files)} files, runner {(push / 'runner.py').stat().st_size // 1024} KB, "
              f"{len(job['conditions'])} corruptions + clean")


if __name__ == "__main__":
    main()
