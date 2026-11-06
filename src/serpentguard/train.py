"""Trains one M3 run and saves everything the comparisons read: config.yaml, versions.json, subset_hash.txt,
class_mix.json, metrics.jsonl, best.pt, final_metrics.json and the val, calibration, unknown, prototype and (K1)
held-out outputs."""
import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from serpentguard.config import load_config, save_config
from serpentguard.losses import total_loss
from serpentguard.models import SnakeID
from serpentguard.run_data import (check_run_config, class_mix_report, eval_loader, frame_files, prediction_coverage,
                                   read_tables, run_frames, seed_everything, train_loader, warm_page_cache)
# re-exported: notebooks and other modules import the run file names from here
from serpentguard.run_files import (CALIBRATION_OUTPUTS, CHECKPOINT_FILE, CLASS_MIX_FILE, CONFIG_FILE,  # noqa: F401
                                    FINAL_FILE, HASH_FILE, HELDOUT_OUTPUTS, METRICS_FILE, PROTOTYPES_FILE,
                                    UNKNOWN_OUTPUTS, VAL_OUTPUTS, VERSIONS_FILE)
from serpentguard.run_outputs import (AMP_DTYPE, HELDOUT_KEYS, build_prototypes, class_tables, device_name, final_metrics,
                                      library_versions, peak_gpu_memory_mb, predict, save_outputs, softmax,
                                      species_scores, subset_hash, write_json)

GRAD_CLIP_NORM = 5.0
ENCODER_LR_SCALE = 0.1
LOSS_TARGETS = ("class_id", "genus_id", "family_id", "who_risk_id")
MASK_TARGETS = ("mask", "mask_valid")

LOG = logging.getLogger(__name__)


@dataclass
class RunResult:
    best_epoch: int
    metrics: dict
    checkpoint: Path


@dataclass
class EpochStats:
    seconds: float
    photos: int
    wait: float
    losses: dict
    lr: float


def train(cfg, data_root, masks_root, subset_csv, out_dir, preds_root=None, device=None, num_workers=4, limit=None,
          pretrained=True, on_eval=None) -> RunResult:
    """Trains cfg, keeps the best epoch by validation macro-F1 and saves its outputs. on_eval(epoch, line) runs after
    each evaluation is saved; an exception it raises ends the run before final_metrics.json."""
    started = time.perf_counter()
    check_run_config(cfg)
    cfg = replace(cfg, subset_path=str(subset_csv))  # config.yaml records the subset the run actually read
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        start_cuda(device)
    seed_everything(cfg.seed)

    tables = read_tables(cfg, data_root, masks_root, subset_csv, preds_root)
    frames = run_frames(cfg, tables, data_root, masks_root, preds_root, limit)
    warmed = warm_page_cache(frame_files(frames))
    LOG.info("%s: read %d files (%.0f MB, %d missing) into the page cache in %.0f s", cfg.run_id, warmed["files"],
             warmed["mb"], warmed["missing"], warmed["seconds"])
    taxa = class_tables(tables.classes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / CONFIG_FILE)
    write_json(out_dir / VERSIONS_FILE, library_versions(device))
    digest = subset_hash(tables.subset["path"])
    (out_dir / HASH_FILE).write_text(digest)
    (out_dir / METRICS_FILE).write_text("")
    coverage = prediction_coverage(frames)
    LOG.info("%s on %s: %d training photos after caps, %d val, %d calibration, %d unknown; subset %s%s", cfg.run_id,
             device_name(device), len(frames.train), len(frames.val), len(frames.calibration), len(frames.unknown),
             digest[:12], f"; front-end predictions cover {coverage}" if coverage else "")

    loader = train_loader(cfg, frames.train, frames.mode, num_workers, device)
    val_loader = eval_loader(cfg, frames.val, frames.mode, num_workers, device)
    model = SnakeID.from_config(cfg, taxa.n_classes, taxa.n_genus, taxa.n_family, pretrained=pretrained).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(cfg.warmup_epochs * len(loader), cfg.epochs * len(loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    train_counts = np.bincount(frames.train["class_id"], minlength=taxa.n_classes)
    frequencies = np.maximum(train_counts, 1) / train_counts.sum()  # a class with no photo counts as one: finite
    log_prior = torch.log(torch.as_tensor(frequencies, dtype=torch.float32)).to(device)

    best_epoch, best_score, best_state, best_outputs, epochs, eval_seconds = None, -math.inf, None, None, [], 0.0
    for epoch in range(1, cfg.epochs + 1):
        drawn = np.zeros(taxa.n_classes, dtype=np.int64) if epoch == 1 else None
        stats = run_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, log_prior, drawn)
        epochs.append(stats)
        if drawn is not None:
            write_json(out_dir / CLASS_MIX_FILE, {"epoch": epoch, **class_mix_report(frames.train, drawn, tables.classes)})
        if epoch % cfg.eval_every and epoch != cfg.epochs:
            continue
        eval_started = time.perf_counter()
        outputs = predict(model, val_loader, device, amp)
        report = species_scores(frames.val, softmax(outputs["species_logits"]), train_counts, taxa)
        line = {"epoch": epoch, "macro_f1": report["macro_f1"], "top1": report["top1"], "top3": report["top3"],
                **{f"loss_{name}": value for name, value in stats.losses.items()}, "lr": stats.lr,
                "seconds": stats.seconds, "photos_per_second": stats.photos / stats.seconds,
                "data_wait_share": stats.wait / stats.seconds, "eval_seconds": time.perf_counter() - eval_started}
        eval_seconds += line["eval_seconds"]
        with open(out_dir / METRICS_FILE, "a") as log:
            log.write(f"{json.dumps(line)}\n")
        LOG.info(progress_line(cfg, line))
        score = -math.inf if math.isnan(line["macro_f1"]) else line["macro_f1"]  # a nan epoch never wins
        if best_epoch is None or score > best_score:
            best_epoch, best_score, best_outputs = epoch, score, outputs
            best_state = {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "config": asdict(cfg), "epoch": epoch, "n_classes": taxa.n_classes,
                        "n_genus": taxa.n_genus, "n_family": taxa.n_family}, out_dir / CHECKPOINT_FILE)
        if on_eval is not None:
            on_eval(epoch, line)

    model.load_state_dict(best_state)

    def outputs_on(frame):
        return predict(model, eval_loader(cfg, frame, frames.mode, num_workers, device), device, amp)

    prototypes = build_prototypes(outputs_on(frames.train)["embedding"], frames.train, taxa.n_classes)
    np.savez_compressed(out_dir / PROTOTYPES_FILE, plain=prototypes[0], balanced=prototypes[1])
    save_outputs(out_dir / VAL_OUTPUTS, frames.val, best_outputs)
    save_outputs(out_dir / CALIBRATION_OUTPUTS, frames.calibration, outputs_on(frames.calibration))
    save_outputs(out_dir / UNKNOWN_OUTPUTS, frames.unknown, outputs_on(frames.unknown))
    if frames.heldout is not None:
        save_outputs(out_dir / HELDOUT_OUTPUTS, frames.heldout, outputs_on(frames.heldout), HELDOUT_KEYS)
    metrics = final_metrics(frames.val, best_outputs, prototypes, train_counts, taxa, cfg.arcface_s)
    train_seconds = sum(stats.seconds for stats in epochs)
    metrics["timing"] = {
        "seconds_per_epoch": train_seconds / len(epochs), "epoch_seconds": [stats.seconds for stats in epochs],
        "photos_per_second": sum(stats.photos for stats in epochs) / train_seconds,
        "data_wait_share": sum(stats.wait for stats in epochs) / train_seconds, "eval_seconds": eval_seconds,
        "warm_cache_seconds": warmed["seconds"], "warm_cache_files": warmed["files"],
        "total_seconds": time.perf_counter() - started, "peak_gpu_memory_mb": peak_gpu_memory_mb(device)}
    metrics["run"] = {"run_id": cfg.run_id, "best_epoch": best_epoch, "epochs": cfg.epochs,
                      "train_photos": len(frames.train), "val_photos": len(frames.val),
                      "calibration_photos": len(frames.calibration), "unknown_photos": len(frames.unknown),
                      "heldout_photos": 0 if frames.heldout is None else len(frames.heldout),
                      "prediction_coverage": coverage, "subset_hash": digest, "device": device_name(device),
                      "num_workers": num_workers,  # the augmentation stream depends on the worker count
                      "encoder_normalization": encoder_normalization(model)}
    return RunResult(best_epoch, write_json(out_dir / FINAL_FILE, metrics), out_dir / CHECKPOINT_FILE)


def encoder_normalization(model):
    """The encoder's own input mean and std, and whether its adapter converts the pipeline's ImageNet input to them."""
    norm = model.encoder.input_norm
    return {"mean": list(norm.mean), "std": list(norm.std), "adapter": norm.active}


def start_cuda(device):
    """Starts CUDA before resetting the peak-memory counter, which otherwise raises "Invalid device argument"."""
    torch.cuda.init()
    if device.index is not None:
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)


def run_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, log_prior, drawn=None):
    """One pass over the loader; `drawn`, when given, counts the classes drawn."""
    model.train()
    amp, targets = scaler.is_enabled(), LOSS_TARGETS + (MASK_TARGETS if model.mask_head is not None else ())
    sums, steps, photos, wait = defaultdict(float), 0, 0, 0.0
    started = ready = time.perf_counter()
    for batch in loader:
        wait += time.perf_counter() - ready
        if drawn is not None:
            drawn += torch.bincount(batch["class_id"], minlength=len(drawn)).numpy()
        images = batch["image"].to(device, non_blocking=True)
        on_device = {name: batch[name].to(device, non_blocking=True) for name in targets}
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=amp):
            out = model(images)
        out = {name: None if value is None else value.float() for name, value in out.items()}
        loss, parts = total_loss(out, on_device, cfg, log_prior=log_prior)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for name, value in parts.items():
            sums[name] += value
        steps += 1
        photos += len(images)
        ready = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return EpochStats(time.perf_counter() - started, photos, wait, {name: total / steps for name, total in sums.items()},
                      optimizer.param_groups[-1]["lr"])


def make_optimizer(model, cfg):
    """AdamW with the encoder at ENCODER_LR_SCALE x lr and every other part at lr."""
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    rest = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return torch.optim.AdamW([{"params": encoder, "lr": cfg.lr * ENCODER_LR_SCALE}, {"params": rest, "lr": cfg.lr}],
                             weight_decay=cfg.weight_decay)


def warmup_cosine(warmup_steps, total_steps):
    """LambdaLR factor per step: linear warm-up to 1 over warmup_steps, then cosine decay to 0 at total_steps."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def progress_line(cfg, line):
    return (f"{cfg.run_id} epoch {line['epoch']}/{cfg.epochs}: macro-F1 {line['macro_f1']:.3f}, top-1 {line['top1']:.3f}, "
            f"top-3 {line['top3']:.3f}, loss {line['loss_total']:.3f}, lr {line['lr']:.2e}, "
            f"{line['photos_per_second']:.0f} photos/s, data wait {line['data_wait_share']:.0%}, "
            f"epoch {line['seconds']:.0f} s + eval {line['eval_seconds']:.0f} s")


def headline(metrics):
    run, head, times = metrics["run"], metrics["head"], metrics["timing"]
    return (f"{run['run_id']} best epoch {run['best_epoch']}: macro-F1 head {head['macro_f1']:.3f}, "
            f"prototype {metrics['prototype']['macro_f1']:.3f}, balanced {metrics['prototype_balanced']['macro_f1']:.3f}; "
            f"top-1 {head['top1']:.3f}, top-3 {head['top3']:.3f}; WHO recall {metrics['safety']['who_recall_raw']:.3f}, "
            f"listed as not listed {metrics['safety']['top1_error_raw']:.3f}; ECE {metrics['calibration']['ece']:.3f}; "
            f"{times['seconds_per_epoch'] / 60:.1f} min/epoch, {times['total_seconds'] / 60:.1f} min in all, "
            f"data wait {times['data_wait_share']:.0%}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m serpentguard.train")
    for name in ("--config", "--data-root", "--masks-root", "--subset", "--out"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--preds-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    result = train(cfg, args.data_root, args.masks_root, args.subset, args.out, preds_root=args.preds_root,
                   device=args.device, num_workers=args.num_workers, limit=args.limit)
    print(headline(result.metrics), flush=True)


if __name__ == "__main__":
    main()
