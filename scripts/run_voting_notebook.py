"""Execute notebook 26's plain Python cells and save their real outputs.

Uses the project's Python environment without requiring a Jupyter installation.
This notebook contains only standard Python; IPython magics are not supported.
"""
import contextlib
import io
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "notebooks/26_community_voting.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    previous = Path.cwd()
    count = 0
    try:
        os.chdir(repo)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec"), namespace)
            cell["execution_count"] = count
            cell["outputs"] = [
                {"output_type": "stream", "name": name, "text": stream.getvalue().splitlines(keepends=True)}
                for name, stream in (("stdout", out), ("stderr", err)) if stream.getvalue()
            ]
    finally:
        os.chdir(previous)
    notebook.setdefault("metadata", {})["execution"] = {
        "method": "Sequential standard-Python cells; scripts/run_voting_notebook.py",
        "python_version": platform.python_version(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "code_cells": count,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Executed and saved {count} code cells in {path.name}.")
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            print("".join(output.get("text", [])), end="")


if __name__ == "__main__":
    main()
