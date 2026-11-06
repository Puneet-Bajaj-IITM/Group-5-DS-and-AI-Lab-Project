"""H1: an Optuna search over the winning configuration, one worker process per GPU on one SQLite study, run as
random-sampler slices (one per Kaggle kernel) that merge_studies() joins; then the retrain configs and the plots."""
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import torch
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.importance import FanovaImportanceEvaluator
from optuna.trial import TrialState

from serpentguard.config import RunConfig, load_config, save_config
from serpentguard.metrics import TIERS
from serpentguard.train import CHECKPOINT_FILE, CONFIG_FILE, FINAL_FILE, METRICS_FILE, PROTOTYPES_FILE, train

N_TRIALS, EPOCHS, EVAL_EVERY = 24, 8, 2
STORAGE, STUDY_NAME = "sqlite:///study.db", "H1"
LR_RANGE = (1e-4, 1e-3)
WEIGHT_DECAY_RANGE = (1e-4, 0.1)
AUG_STRENGTH_RANGE = (0.5, 1.5)
ARCFACE_M_RANGE = (0.1, 0.5)
TRIPLET_MARGIN_RANGE = (0.1, 0.4)
EMBEDDING_DIMS = (128, 256, 512)
PRUNER_STARTUP_TRIALS = 5  # recorded with the study; the reported search ran with pruning off, so nothing prunes
TPE, RANDOM = "tpe", "random"
SAMPLERS = (TPE, RANDOM)
SAMPLER_SEED = 42
SQLITE_TIMEOUT_SECONDS = 60  # how long a worker waits for the other's write lock before failing
POLL_SECONDS = 300
TOP_K, RETRAIN_SEEDS = 3, (1, 2)
SRC_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = SRC_DIR.parent / "configs" / "m3"
COMPARED_SETTINGS = ("base_config", "epochs", "eval_every", "prune")  # what slices must share to be merged
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)
LOG_TAIL_CHARS, ERROR_CHARS = 5000, 2000

PLOT_DPI, FIGURE_SIZE = 150, (8, 5)
IMPORTANCE_SEED, MIN_IMPORTANCE_TRIALS = 42, 2
PARAM_ORDER = ("lr", "weight_decay", "aug_strength", "arcface_m", "triplet_margin", "embedding_dim")
COLORS = {"trial": "#0072b2", "pruned": "#999999", "tuned": "#d55e00", "default": "#009e73"}  # Okabe-Ito
GROUP_LABELS = {"default": "default recipe", "tuned": "tuned"}
PLOT_FILES = {"history": "h1_history.png", "parallel": "h1_parallel.png", "importances": "h1_importances.png",
              "f1_by_epoch": "h1_f1_by_epoch.png", "tiers": "h1_tier_f1.png"}
BAR_WIDTH = 0.7

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialInputs:
    data_root: Path
    masks_root: Path
    subset_csv: Path
    out_dir: Path
    preds_root: Path | None = None
    device: str | None = None
    num_workers: int = 4
    pretrained: bool = True
    keep_checkpoints: bool = False  # the retrain remakes the top configs' checkpoints; trials keep their metrics


@dataclass
class Worker:
    index: int
    process: subprocess.Popen
    log: object
    log_path: Path


def search_space(base_cfg):
    """lr, weight decay, augmentation strength and one more: the ArcFace margin, the triplet margin or the embedding."""
    space = {"lr": FloatDistribution(*LR_RANGE, log=True),
             "weight_decay": FloatDistribution(*WEIGHT_DECAY_RANGE, log=True),
             "aug_strength": FloatDistribution(*AUG_STRENGTH_RANGE)}
    if base_cfg.head == "arcface":
        space["arcface_m"] = FloatDistribution(*ARCFACE_M_RANGE)
    elif "triplet" in base_cfg.losses:
        space["triplet_margin"] = FloatDistribution(*TRIPLET_MARGIN_RANGE)
    else:
        space["embedding_dim"] = CategoricalDistribution(EMBEDDING_DIMS)
    return space


def fixed_params(base_cfg):
    """The triplet margin, held at its base value when an ArcFace head's margin is tuned instead."""
    both = base_cfg.head == "arcface" and "triplet" in base_cfg.losses
    return {"triplet_margin": base_cfg.triplet_margin} if both else {}


def suggest_params(trial, space):
    return {name: trial.suggest_categorical(name, distribution.choices)
            if isinstance(distribution, CategoricalDistribution)
            else trial.suggest_float(name, distribution.low, distribution.high, log=distribution.log)
            for name, distribution in space.items()}


def make_storage(url, timeout=SQLITE_TIMEOUT_SECONDS):
    """SQLite storage whose connections wait up to `timeout` seconds for a lock; any other URL as it is."""
    if url is None or not str(url).startswith("sqlite"):
        return url
    return optuna.storages.RDBStorage(str(url), engine_kwargs={"connect_args": {"timeout": timeout}})


def make_sampler(kind, seed):
    if kind == TPE:
        return optuna.samplers.TPESampler(seed=seed, constant_liar=True)
    return optuna.samplers.RandomSampler(seed=seed)


def worker_seed(sampler_seed, worker, offset=0):
    """A sampler seed per kernel and worker (seed + worker index would collide across kernels)."""
    return int(np.random.SeedSequence([sampler_seed, worker, offset]).generate_state(1)[0])


def create_search(base_cfg, storage, study_name=STUDY_NAME, epochs=EPOCHS, eval_every=EVAL_EVERY,
                  startup_trials=PRUNER_STARTUP_TRIALS, sampler=TPE, sampler_seed=SAMPLER_SEED, prune=True):
    """Creates the study with its settings as user attributes, which the workers and every later step read."""
    study = optuna.create_study(storage=make_storage(storage), study_name=study_name, direction="maximize")
    record = {"base_config": asdict(base_cfg), "epochs": epochs, "eval_every": eval_every,
              "pruner_startup_trials": startup_trials, "sampler": sampler, "prune": prune,
              "sampler_seed": sampler_seed, "tuned": list(search_space(base_cfg)), "fixed": fixed_params(base_cfg)}
    for key, value in json.loads(json.dumps(record)).items():
        study.set_user_attr(key, value)
    return study


class BestSoFar:
    """train()'s on_eval: reports the best validation macro-F1 so far to optuna; a nan macro-F1 is never a best."""

    def __init__(self, trial):
        self.trial, self.best = trial, math.nan

    def __call__(self, epoch, line):
        score = line["macro_f1"]
        if not math.isnan(score) and (math.isnan(self.best) or score > self.best):
            self.best = score
        if not math.isnan(self.best):
            self.trial.report(self.best, epoch)


def objective(trial, settings, inputs, worker=0):
    """Trains one trial into its own folder and returns its best validation macro-F1 (nan fails it)."""
    trial.set_user_attr("worker", worker)
    base = RunConfig(**settings["base_config"])
    cfg = replace(base, **suggest_params(trial, search_space(base)), run_id=f"H1_trial{trial.number:03d}",
                  epochs=settings["epochs"], eval_every=settings["eval_every"])
    run_dir = Path(inputs.out_dir) / f"trial_{trial.number:03d}"
    trial.set_user_attr("run_dir", str(run_dir))
    hook = BestSoFar(trial)
    try:
        result = train(cfg, inputs.data_root, inputs.masks_root, inputs.subset_csv, run_dir,
                       preds_root=inputs.preds_root, device=inputs.device, num_workers=inputs.num_workers,
                       pretrained=inputs.pretrained, on_eval=hook)
    except Exception as error:
        trial.set_user_attr("error", f"{type(error).__name__}: {error}"[:ERROR_CHARS])
        raise
    finally:
        if not inputs.keep_checkpoints:
            for path in [run_dir / CHECKPOINT_FILE, run_dir / PROTOTYPES_FILE, *run_dir.glob("*_outputs.npz")]:
                path.unlink(missing_ok=True)
    trial.set_user_attr("best_epoch", result.best_epoch)
    return hook.best


def run_worker(storage, study_name, n_trials, sampler_seed, inputs, worker=0):
    """Runs n_trials of the study in this process; a trial that raises is recorded as failed and the next one runs."""
    study = optuna.load_study(study_name=study_name, storage=make_storage(storage))
    study.sampler = make_sampler(study.user_attrs["sampler"], sampler_seed)
    study.optimize(partial(objective, settings=study.user_attrs, inputs=inputs, worker=worker), n_trials=n_trials,
                   catch=(Exception,), gc_after_trial=True)


def run_study(base_cfg, n_trials=N_TRIALS, epochs=EPOCHS, storage=STORAGE, *, data_root, masks_root, subset_csv,
              out_dir, preds_root=None, eval_every=EVAL_EVERY, devices=None, num_workers=4, pretrained=True,
              study_name=STUDY_NAME, startup_trials=PRUNER_STARTUP_TRIALS, sampler=TPE, sampler_seed=SAMPLER_SEED,
              prune=True, keep_checkpoints=False, poll_seconds=POLL_SECONDS) -> optuna.Study:
    """n_trials trials of a new study, one worker process per device (default every GPU, else the CPU).
    startup_trials and prune are only recorded: the reported search ran without pruning."""
    create_search(base_cfg, storage, study_name, epochs, eval_every, startup_trials, sampler, sampler_seed, prune)
    devices = devices or [f"cuda:{index}" for index in range(torch.cuda.device_count())] or ["cpu"]
    shares = [n_trials // len(devices) + (index < n_trials % len(devices)) for index in range(len(devices))]
    inputs = TrialInputs(Path(data_root), Path(masks_root), Path(subset_csv), Path(out_dir),
                         None if preds_root is None else Path(preds_root), num_workers=num_workers,
                         pretrained=pretrained, keep_checkpoints=keep_checkpoints)
    inputs.out_dir.mkdir(parents=True, exist_ok=True)
    workers = [start_worker(index, device, share, storage, study_name, worker_seed(sampler_seed, index), inputs)
               for index, (device, share) in enumerate(zip(devices, shares)) if share]
    wait_for(workers, study_name, storage, n_trials, poll_seconds)
    return optuna.load_study(study_name=study_name, storage=make_storage(storage))


def start_worker(index, device, n_trials, storage, study_name, sampler_seed, inputs):
    """Runs run_worker in `python -m serpentguard.tune <job as JSON>` on one GPU, logging to worker_<index>.log."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC_DIR), os.environ.get("PYTHONPATH")]))}
    if device.startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = device.partition(":")[2] or "0"
        device = "cuda:0"
    job = {"storage": str(storage), "study_name": study_name, "n_trials": n_trials, "sampler_seed": sampler_seed,
           "inputs": asdict(replace(inputs, device=device)), "worker": index}
    command = [sys.executable, "-u", "-m", "serpentguard.tune", json.dumps(job, default=str)]
    log_path = inputs.out_dir / f"worker_{index}.log"
    log = open(log_path, "w")
    LOG.info("worker %d (%s, CUDA_VISIBLE_DEVICES=%s): %d trials, log %s", index, device,
             env.get("CUDA_VISIBLE_DEVICES", "unset"), n_trials, log_path)
    return Worker(index, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log, log_path)


def wait_for(workers, study_name, storage, n_trials, poll_seconds):
    """Waits for every worker, logging progress every poll_seconds; raises with the log tail of each that failed."""
    started = time.monotonic()
    for worker in workers:
        while True:
            try:
                worker.process.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                LOG.info("%.1f min: %s", (time.monotonic() - started) / 60, progress(study_name, storage, n_trials))
    for worker in workers:
        worker.log.close()
    failed = [f"worker {worker.index} exited with {worker.process.returncode}; {worker.log_path}:\n"
              f"{worker.log_path.read_text(errors='replace')[-LOG_TAIL_CHARS:]}"
              for worker in workers if worker.process.returncode]
    if failed:
        raise RuntimeError("\n\n".join(failed))


def progress(study_name, storage, n_trials):
    trials = optuna.load_study(study_name=study_name, storage=make_storage(storage)).get_trials(deepcopy=False)
    states = Counter(trial.state for trial in trials)
    complete = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    best = max(complete, key=lambda trial: trial.value, default=None)
    return (f"{states[TrialState.COMPLETE]} complete, {states[TrialState.PRUNED]} pruned, "
            f"{states[TrialState.RUNNING]} running, {states[TrialState.FAIL]} failed of {n_trials}"
            + (f"; best macro-F1 {best.value:.4f} (trial {best.number})" if best else ""))


def ranked_trials(study):
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return sorted(complete, key=lambda trial: (-trial.value, trial.number))


def merge_studies(db_paths, study_name=STUDY_NAME, storage=None) -> optuna.Study:
    """One study with every finished trial of the slice databases, each tagged with its slice (storage None: memory)."""
    with tempfile.TemporaryDirectory() as scratch:
        slices = []
        for index, path in enumerate(map(Path, db_paths)):
            copy = Path(scratch) / f"{index}_{path.name}"  # read a copy: the slice files sit in read-only inputs
            shutil.copyfile(path, copy)
            storage_copy = make_storage(f"sqlite:///{copy}")
            slices += [(optuna.load_study(study_name=name, storage=storage_copy), path)
                       for name in optuna.get_all_study_names(storage_copy)]
        reference = slices[0][0]
        copies, seen = [], set()
        for study, path in slices:
            if any(study.user_attrs.get(key) != reference.user_attrs.get(key) for key in COMPARED_SETTINGS):
                raise ValueError(f"slice {study.study_name!r} in {path} was run with other settings")
            for trial in study.get_trials(deepcopy=False, states=FINISHED):
                point = tuple(sorted(trial.params.items()))
                if point in seen:
                    raise ValueError(f"{study.study_name!r} trial {trial.number} repeats a point: one sampler seed?")
                seen.add(point)
                source = {"slice": study.study_name, "slice_trial": trial.number, "slice_db": str(path)}
                copies.append(optuna.trial.FrozenTrial(
                    number=0, trial_id=0, state=trial.state, value=None, values=trial.values,
                    datetime_start=trial.datetime_start, datetime_complete=trial.datetime_complete,
                    params=trial.params, distributions=trial.distributions,
                    user_attrs={**trial.user_attrs, **source}, system_attrs={},
                    intermediate_values=trial.intermediate_values))
        samplers = {study.user_attrs.get("sampler") for study, _ in slices}
        record = {**{key: reference.user_attrs[key] for key in ("tuned", "fixed", "pruner_startup_trials",
                                                                 *COMPARED_SETTINGS)},
                  "sampler": samplers.pop() if len(samplers) == 1 else "mixed",
                  "slices": [{"study": study.study_name, "db": str(path), "sampler": study.user_attrs.get("sampler"),
                              "sampler_seed": study.user_attrs.get("sampler_seed"),
                              "finished_trials": len(study.get_trials(deepcopy=False, states=FINISHED))}
                             for study, path in slices]}
    merged = optuna.create_study(storage=make_storage(storage), study_name=study_name, direction="maximize")
    for key, value in record.items():
        merged.set_user_attr(key, value)
    merged.add_trials(copies)
    return merged


def top_configs(study, k=TOP_K, out_dir=CONFIGS_DIR, seeds=RETRAIN_SEEDS):
    """Writes the k best trials as retrain configs H1_top{rank}_s{seed}.yaml, headed by their source trial."""
    base, epochs = RunConfig(**study.user_attrs["base_config"]), study.user_attrs["epochs"]
    fixed = study.user_attrs.get("fixed") or {}
    paths = []
    for rank, trial in enumerate(ranked_trials(study)[:k], 1):
        where = f" ({Path(trial.user_attrs['slice_db']).name})" if "slice_db" in trial.user_attrs else ""
        header = [f"source_trial: {trial.user_attrs.get('slice_trial', trial.number)}",
                  f"source_study: {trial.user_attrs.get('slice', study.study_name)}{where}",
                  f"H1 top {rank} of {k}: best validation macro-F1 {trial.value:.4f} over {epochs} epochs "
                  f"(trial {trial.number} of {study.study_name!r}), parameters on base config {base.run_id}"]
        if fixed:
            header.append("kept at the base value while the ArcFace margin is tuned: "
                          + ", ".join(f"{name} {value}" for name, value in fixed.items()))
        for seed in seeds:
            path = Path(out_dir) / f"H1_top{rank}_s{seed}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_config(replace(base, **trial.params, run_id=path.stem, seed=seed), path)
            path.write_text("".join(f"# {line}\n" for line in header) + path.read_text())
            paths.append(path)
    return paths


def new_axes():
    figure = Figure(figsize=FIGURE_SIZE, layout="constrained")
    return figure, figure.add_subplot()


def saved(figure, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=PLOT_DPI)
    return figure


def trials_in(study, *states):
    return sorted(study.get_trials(deepcopy=False, states=states), key=lambda trial: trial.number)


def plot_history(study, path):
    """Each completed trial's score, each pruned trial's best score when it stopped, and the best so far."""
    figure, axes = new_axes()
    complete, pruned = trials_in(study, TrialState.COMPLETE), trials_in(study, TrialState.PRUNED)
    if complete:
        numbers, values = [trial.number for trial in complete], [trial.value for trial in complete]
        axes.scatter(numbers, values, color=COLORS["trial"], label="completed trial", zorder=3)
        axes.step(numbers, np.maximum.accumulate(values), where="post", color=COLORS["tuned"], label="best so far")
    if pruned:
        last = [trial.intermediate_values[max(trial.intermediate_values)] if trial.intermediate_values else math.nan
                for trial in pruned]
        axes.scatter([trial.number for trial in pruned], last, marker="x", color=COLORS["pruned"],
                     label="pruned trial (best when stopped)", zorder=3)
    if complete or pruned:
        axes.legend(loc="lower right")
    slices = {trial.user_attrs["slice"] for trial in complete + pruned if "slice" in trial.user_attrs}
    across = f" across {len(slices)} slices" if slices else ""
    axes.set(title=f"H1 optimisation history: {len(complete)} completed, {len(pruned)} pruned{across}",
             xlabel="trial number" + (" (slice by slice)" if slices else ""), ylabel="best validation macro-F1")
    return saved(figure, path)


def position(distribution, value):
    """Where value sits in its searched range, 0 to 1 (log scale for log distributions)."""
    if isinstance(distribution, CategoricalDistribution):
        return distribution.choices.index(value) / max(1, len(distribution.choices) - 1)
    low, high = distribution.low, distribution.high
    if distribution.log:
        low, high, value = math.log(low), math.log(high), math.log(value)
    return (value - low) / (high - low)


def plot_parallel(study, path):
    """One line per trial across the parameters' ranges and the score, coloured by score, pruned grey, best marked."""
    figure, axes = new_axes()
    complete, pruned = trials_in(study, TrialState.COMPLETE), trials_in(study, TrialState.PRUNED)
    names = [name for name in PARAM_ORDER if any(name in trial.params for trial in complete + pruned)]
    distributions = {name: next(trial.distributions[name] for trial in complete + pruned if name in trial.params)
                     for name in names}
    xs = list(range(len(names) + 1))

    def params_at(trial):
        return [position(distributions[name], trial.params[name]) for name in names]

    for trial in pruned:
        if set(names) <= trial.params.keys():
            axes.plot(xs[:-1], params_at(trial), color=COLORS["pruned"], alpha=0.6, lw=1)
    ends = {}
    if complete:
        values = [trial.value for trial in complete]
        norm, cmap = Normalize(min(values), max(values)), colormaps["viridis"]

        def score_at(value):
            return norm(value) if norm.vmax > norm.vmin else 0.5

        for trial in complete:
            axes.plot(xs, params_at(trial) + [score_at(trial.value)], color=cmap(norm(trial.value)), lw=1.5)
        best = ranked_trials(study)[0]
        axes.plot(xs, params_at(best) + [score_at(best.value)], color=COLORS["tuned"], lw=3,
                  label=f"best: trial {best.number}")
        axes.legend(loc="upper left")
        figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="best validation macro-F1")
        ends[len(names)] = (min(values), max(values))
    for index, name in enumerate(names):
        dist = distributions[name]
        categorical = isinstance(dist, CategoricalDistribution)
        ends[index] = (dist.choices[0], dist.choices[-1]) if categorical else (dist.low, dist.high)
    for index, (low, high) in ends.items():
        axes.axvline(index, color="black", lw=0.8)
        axes.text(index, -0.04, f"{low:.3g}", ha="center", va="top", fontsize=8)
        axes.text(index, 1.04, f"{high:.3g}", ha="center", va="bottom", fontsize=8)
    axes.set_xticks(xs, [*names, "macro-F1"])
    axes.set(ylim=(-0.15, 1.15), title="H1 parallel coordinates: one line per trial",
             xlabel="hyperparameter (lr and weight decay on log scales)", ylabel="position in the searched range")
    return saved(figure, path)


def plot_importances(study, path):
    """fANOVA importances over the completed trials."""
    figure, axes = new_axes()
    complete = trials_in(study, TrialState.COMPLETE)
    if len(complete) < MIN_IMPORTANCE_TRIALS:
        axes.text(0.5, 0.5, f"needs at least {MIN_IMPORTANCE_TRIALS} completed trials; the study has {len(complete)}",
                  ha="center", va="center", transform=axes.transAxes)
    else:
        importances = optuna.importance.get_param_importances(
            study, evaluator=FanovaImportanceEvaluator(seed=IMPORTANCE_SEED))
        names = list(importances)[::-1]
        axes.barh(names, [importances[name] for name in names], color=COLORS["trial"])
    axes.set(title=f"H1 parameter importances (fANOVA over {len(complete)} completed trials)",
             xlabel="share of the variance in best validation macro-F1", ylabel="hyperparameter")
    return saved(figure, path)


def as_runs(runs):
    if runs is None:
        return []
    return [Path(runs)] if isinstance(runs, (str, Path)) else [Path(run) for run in runs]


def plot_f1_by_epoch(default_runs, tuned_runs, path):
    """Validation macro-F1 at each evaluated epoch, from each run folder's metrics.jsonl."""
    figure, axes = new_axes()
    for group, runs in (("default", default_runs), ("tuned", tuned_runs)):
        for run in as_runs(runs):
            lines = [json.loads(line) for line in (run / METRICS_FILE).read_text().splitlines() if line.strip()]
            axes.plot([line["epoch"] for line in lines], [line["macro_f1"] for line in lines], marker="o",
                      color=COLORS[group], label=f"{GROUP_LABELS[group]}: {load_config(run / CONFIG_FILE).run_id}")
    axes.legend(loc="lower right")
    axes.set(title="Validation macro-F1 by epoch: default recipe vs tuned", xlabel="epoch",
             ylabel="validation macro-F1 (head)")
    return saved(figure, path)


def plot_tier_f1(tuned_runs, path, default_runs=None):
    """Tier 1 and tier 2 head macro-F1 from final_metrics.json: a bar per group's mean, a dot per run (seed)."""
    figure, axes = new_axes()
    groups = [(group, as_runs(runs)) for group, runs in (("default", default_runs), ("tuned", tuned_runs))
              if as_runs(runs)]
    xs, width = np.arange(len(TIERS)), BAR_WIDTH / len(groups)
    for index, (group, runs) in enumerate(groups):
        scores = np.array([[json.loads((run / FINAL_FILE).read_text())["head"][f"macro_f1_{tier}"] for tier in TIERS]
                           for run in runs])
        offset = xs + (index - (len(groups) - 1) / 2) * width
        axes.bar(offset, scores.mean(axis=0), width, color=COLORS[group], label=GROUP_LABELS[group])
        if len(runs) > 1:
            for row in scores:
                axes.scatter(offset, row, color="black", s=12, zorder=3)
    axes.set_xticks(xs, [tier.replace("tier", "tier ") for tier in TIERS])
    axes.legend(loc="upper right")
    axes.set(title="Macro-F1 by tier: tuned vs default recipe", xlabel="species tier",
             ylabel="validation macro-F1 (head, best epoch)")
    return saved(figure, path)


def save_study_plots(study, out_dir):
    """The optimisation history, parallel coordinates and parameter importances; returns their paths."""
    plots = {"history": plot_history, "parallel": plot_parallel, "importances": plot_importances}
    paths = {key: Path(out_dir) / PLOT_FILES[key] for key in plots}
    for key, plot in plots.items():
        plot(study, paths[key])
    return paths


def save_run_plots(default_runs, tuned_runs, out_dir):
    """Macro-F1 by epoch and by tier from finished run folders; returns their paths."""
    paths = {key: Path(out_dir) / PLOT_FILES[key] for key in ("f1_by_epoch", "tiers")}
    plot_f1_by_epoch(default_runs, tuned_runs, paths["f1_by_epoch"])
    plot_tier_f1(tuned_runs, paths["tiers"], default_runs=default_runs)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    job = json.loads(sys.argv[1])
    run_worker(job["storage"], job["study_name"], job["n_trials"], job["sampler_seed"], TrialInputs(**job["inputs"]),
               job["worker"])
