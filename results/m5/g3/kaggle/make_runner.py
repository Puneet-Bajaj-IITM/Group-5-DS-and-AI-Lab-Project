"""Build a Kaggle script kernel that runs one M3 notebook with the repo's code bundled inside, and saves the executed notebook with outputs.

Usage: python make_m3_runner.py <repo_dir> <notebook rel path> <out_dir> <owner/slug> <gpu 0|1> <params.json> [source ...]

A source is a dataset handle (owner/slug), or kernel:owner/slug for another notebook's output; a batch kernel can
read a notebook's output only when it is declared here.

The kernel unpacks src/, configs/, data/ (small CSVs such as classes.csv) and the notebook into /kaggle/working/repo, so the notebook's "../src" fallback finds the code
and no code dataset has to be shared between accounts. params.json maps notebook variable names to Python literals that replace
their assignment lines (for example {"RUN_CONFIGS": ["E1_mnv4", "E1_effb0"]}). The executed notebook lands in
/kaggle/working/executed/; a cell error fails the run after saving.
"""
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

REPO, NOTEBOOK_REL, OUT_DIR, KERNEL_ID, GPU, PARAMS = (Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4],
                                                       sys.argv[5] == "1", json.loads(Path(sys.argv[6]).read_text()))
KERNEL_PREFIX = "kernel:"
DATASETS = [handle for handle in sys.argv[7:] if not handle.startswith(KERNEL_PREFIX)]
KERNELS = [handle[len(KERNEL_PREFIX):] for handle in sys.argv[7:] if handle.startswith(KERNEL_PREFIX)]
BUNDLED = ("src", "data", "scripts/b1_teacher.py", "results/m4/full/A3")
TWO_T4 = "NvidiaTeslaT4"  # Kaggle gives two T4s on this shape; earlier M2 GPU kernels used it

RUNNER = '''import base64, io, json, subprocess, sys, zipfile
from pathlib import Path

NOTEBOOK = "{notebook}"
BUNDLE_B64 = "{bundle}"

repo, out = Path("/kaggle/working/repo"), Path("/kaggle/working/executed")
out.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))).extractall(repo)
run = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors",
                      "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3",
                      "--output-dir", str(out), str(repo / NOTEBOOK)], capture_output=True, text=True, cwd=repo / Path(NOTEBOOK).parent)
print(run.stdout[-2000:], run.stderr[-4000:], sep="\\n")
executed = json.loads((out / Path(NOTEBOOK).name).read_text())
errors = [o for c in executed["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
for e in errors:
    print("CELL ERROR:", e.get("ename"), e.get("evalue"))
    print("\\n".join(e.get("traceback", []))[-3000:])
if run.returncode or errors:
    raise SystemExit(1)
print("executed without errors:", NOTEBOOK)
'''


def set_params(notebook, params):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for name, value in params.items():
            source, count = re.subn(rf"^{re.escape(name)}\s*=.*$", f"{name} = {value!r}", source, flags=re.M)
            if count:
                params = {k: v for k, v in params.items() if k != name}
        cell["source"] = source.splitlines(keepends=True)
    if params:
        raise SystemExit(f"no assignment line found for {sorted(params)}")
    return notebook


def bundle(repo, notebook_rel, notebook):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in BUNDLED:
            root = repo / entry
            for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".png", ".jpg"):
                    archive.write(path, path.relative_to(repo).as_posix())
        archive.writestr(notebook_rel, json.dumps(notebook, indent=1))
    return base64.b64encode(buffer.getvalue()).decode()


notebook = set_params(json.loads((REPO / NOTEBOOK_REL).read_text()), dict(PARAMS))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "runner.py").write_text(RUNNER.format(notebook=NOTEBOOK_REL, bundle=bundle(REPO, NOTEBOOK_REL, notebook)))
meta = {"id": KERNEL_ID, "title": KERNEL_ID.split("/")[1], "code_file": "runner.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": GPU, "enable_tpu": False, "enable_internet": True,
        **({"machine_shape": TWO_T4} if GPU else {}),
        "dataset_sources": DATASETS, "competition_sources": [], "kernel_sources": KERNELS, "model_sources": []}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("runner", KERNEL_ID, "->", OUT_DIR, f"({(OUT_DIR / 'runner.py').stat().st_size // 1024} KB)")
