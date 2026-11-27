"""results/m5/g5/code/summarize.py, unchanged, run on the frozen results folder (without the dev set, not re-scored)."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
source = (HERE.parents[1] / "g5/code/summarize.py").read_text()
source = source.replace('R = Path(__file__).resolve().parents[1] / "results"', f'R = Path({str(HERE / "results")!r})')
source = source.replace('["new_test_scores", "old_test_second_look", "dev_scores"]', '["new_test_scores", "old_test_second_look"]')
exec(compile(source, "summarize.py", "exec"))
