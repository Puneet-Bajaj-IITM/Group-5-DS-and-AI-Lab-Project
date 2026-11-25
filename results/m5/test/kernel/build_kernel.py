"""Bundle the repo code and the test kernel into one Kaggle script kernel (push/runner.py + kernel-metadata.json).

The bundle: m5-wt's src/ and data/, scripts/b1_teacher.py, and this folder's test_kernel.py and gate_worker.py. It
is unpacked under /kaggle/tmp/repo (outside the saved output) and test_kernel.py runs from there.

Usage: m3-final/.venv/bin/python -I results/m5/test/kernel/build_kernel.py
"""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
REPO = WORK / "m5-wt"
PUSH = HERE / "push"
KERNEL_ID = "jayeshbansal497/sg-m5-test"
FROM_REPO = ("src", "data", "scripts/b1_teacher.py")
FROM_HERE = ("test_kernel.py", "gate_worker.py")
DATASETS = ["puneetbajajiitm/serpentguard-indian-snakes-clean", "puneetbajajiitm/serpentguard-m3-subset",
            "jayeshbansal497/serpentguard-m5-inputs", "jayeshbansal497/serpentguard-m3-b1-predictions",
            "jayeshbansal497/serpentguard-m5-test-inputs"]
TWO_T4 = "NvidiaTeslaT4"

RUNNER = '''import base64, io, subprocess, sys, zipfile
from pathlib import Path

repo = Path("/kaggle/tmp/repo")
zipfile.ZipFile(io.BytesIO(base64.b64decode("{bundle}"))).extractall(repo)
code = subprocess.run([sys.executable, "-u", str(repo / "test_kernel.py")], cwd=repo).returncode
raise SystemExit(code)
'''


def bundle():
    buffer, files = io.BytesIO(), {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        entries = [(REPO / e, REPO) for e in FROM_REPO] + [(HERE / e, HERE) for e in FROM_HERE]
        for entry, base in entries:
            for path in sorted(entry.rglob("*")) if entry.is_dir() else [entry]:
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".png", ".jpg"):
                    name = path.relative_to(base).as_posix()
                    archive.write(path, name)
                    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return base64.b64encode(buffer.getvalue()).decode(), files


def main():
    encoded, files = bundle()
    PUSH.mkdir(exist_ok=True)
    (PUSH / "runner.py").write_text(RUNNER.format(bundle=encoded))
    meta = {"id": KERNEL_ID, "title": KERNEL_ID.split("/")[1], "code_file": "runner.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": TWO_T4, "dataset_sources": DATASETS,
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (PUSH / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (PUSH / "bundle_files.json").write_text(json.dumps(files, indent=1) + "\n")
    print(f"{KERNEL_ID}: {len(files)} files, runner {(PUSH / 'runner.py').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
