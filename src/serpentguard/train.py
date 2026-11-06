"""Trains one run and writes its run folder (the file names below), which every later step reads."""
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
import timm
import torch
import yaml

from serpentguard.data import (PREDICTED_INPUT_MODES, TARGETS, class_sampling_probs, eval_loader,
                               frame_files, has_input, input_mode, make_loaders, run_config,
                               seed_everything, warm_cache)
from serpentguard.metrics import (calibration, class_tables, photographer_groups, safety_report,
                                  softmax, species_report, taxonomy_report, write_json)
from serpentguard.models import SnakeID, cosine_to, prototypes_from, total_loss

CONFIG_FILE = "config.yaml"
VERSIONS_FILE = "versions.json"
HASH_FILE = "subset_hash.txt"
CLASS_MIX_FILE = "class_mix.json"
METRICS_FILE = "metrics.jsonl"
FINAL_FILE = "final_metrics.json"
CHECKPOINT_FILE = "best.pt"
VAL_OUTPUTS = "val_outputs.npz"
CALIBRATION_OUTPUTS = "calib_outputs.npz"
UNKNOWN_OUTPUTS = "unknown_outputs.npz"
HELDOUT_OUTPUTS = "heldout_outputs.npz"
PROTOTYPES_FILE = "prototypes.npz"


def check_config(cfg):
    """Raises ValueError for settings that would quietly train something other than intended."""
    for name in ("run_id", "encoder", "head", "losses", "front_end", "sampler"):
        if cfg[name] is None:
            raise ValueError(f"every run must set {name}")
    if "kd" in cfg["losses"]:
        raise ValueError("the kd loss needs a teacher model, which is not built")
    if "triplet" in cfg["losses"] and cfg["sampler"] != "pk":
        raise ValueError("a triplet loss needs sampler 'pk', so every batch holds k photos of each "
                         "class")
    if "arcface" in cfg["losses"] and cfg["head"] != "arcface":
        raise ValueError(f"the arcface loss needs head 'arcface', got head {cfg['head']!r}")
    if cfg["front_end"] == "F4" and not (cfg["mask_head"] and cfg["pooling"] == "gap"):
        raise ValueError("F4 trains a mask head and pools without it: set mask_head true and "
                         "pooling 'gap'")
    if (cfg["front_end"] == "F5") != (cfg["mask_head"] and cfg["pooling"] == "mask"):
        raise ValueError("F5, and only F5, pools with its mask: set mask_head true and pooling "
                         "'mask'")
    if input_mode(cfg) in PREDICTED_INPUT_MODES and cfg["mask_source"] != "predicted":
        raise ValueError(f"{cfg['front_end']} trains and validates on front-end predictions: "
                         "set mask_source 'predicted'")


def train(cfg, out_dir, data_root, masks_root, subset_csv, preds_root=None, device="cuda",
          num_workers=4, limit=None, pretrained=True, on_eval=None):
    """Trains cfg, keeps the best epoch by val macro-F1, writes the run folder, returns metrics."""
    # on_eval(epoch, line) runs after each evaluation is saved; raising in it ends the run early
    started = time.perf_counter()
    cfg = run_config({**cfg, "subset_path": str(subset_csv)})  # records the subset actually read
    check_config(cfg)
    device = torch.device(device)
    amp = device.type == "cuda"
    if amp:
        start_cuda(device)
    seed_everything(cfg["seed"])

    frames, loader, val_loader = make_loaders(cfg, data_root, masks_root, subset_csv, preds_root,
                                              num_workers, limit)
    mode = input_mode(cfg)
    warmed = warm_cache(frame_files(frames, mode))
    print(f"{cfg['run_id']}: read {warmed['files']} files ({warmed['mb']:.0f} MB, "
          f"{warmed['missing']} missing) into the page cache in {warmed['seconds']:.0f} s",
          flush=True)
    classes = pd.read_csv(Path(data_root, "classes.csv"))
    taxa = class_tables(classes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CONFIG_FILE).write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out_dir / VERSIONS_FILE, library_versions(device))
    digest = subset_hash(pd.read_csv(subset_csv)["path"])
    (out_dir / HASH_FILE).write_text(digest)
    (out_dir / METRICS_FILE).write_text("")
    coverage = prediction_coverage(frames, mode)
    print(f"{cfg['run_id']} on {device_name(device)}: {len(frames['train'])} training photos "
          f"after caps, {len(frames['val'])} val, {len(frames['calibration'])} calibration, "
          f"{len(frames['unknown'])} unknown; subset {digest[:12]}"
          + (f"; front-end predictions cover {coverage}" if coverage else ""), flush=True)

    model = SnakeID(cfg, taxa.n_classes, taxa.n_genus, taxa.n_family,
                    pretrained=pretrained).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(cfg["warmup_epochs"] * len(loader), cfg["epochs"] * len(loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    train_counts = np.bincount(frames["train"]["class_id"], minlength=taxa.n_classes)
    frequencies = np.maximum(train_counts, 1) / train_counts.sum()  # no photo counts as one
    log_prior = torch.log(torch.as_tensor(frequencies, dtype=torch.float32)).to(device)

    best_epoch, best_score, best_state, best_outputs, epochs, eval_seconds = (None, -math.inf, None,
                                                                              None, [], 0.0)
    for epoch in range(1, cfg["epochs"] + 1):
        drawn = np.zeros(taxa.n_classes, dtype=np.int64) if epoch == 1 else None
        stats = run_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, log_prior,
                          drawn)
        epochs.append(stats)
        if drawn is not None:
            write_json(out_dir / CLASS_MIX_FILE,
                       {"epoch": epoch, **class_mix_report(frames["train"], drawn, classes)})
        if epoch % cfg["eval_every"] and epoch != cfg["epochs"]:
            continue
        eval_started = time.perf_counter()
        outputs = predict(model, val_loader, device, amp)
        report = species_scores(frames["val"], softmax(outputs["species_logits"]), train_counts,
                                taxa)
        line = {"epoch": epoch, "macro_f1": report["macro_f1"], "top1": report["top1"],
                "top3": report["top3"],
                **{f"loss_{name}": value for name, value in stats["losses"].items()},
                "lr": stats["lr"], "seconds": stats["seconds"],
                "photos_per_second": stats["photos"] / stats["seconds"],
                "data_wait_share": stats["wait"] / stats["seconds"],
                "eval_seconds": time.perf_counter() - eval_started}
        eval_seconds += line["eval_seconds"]
        with open(out_dir / METRICS_FILE, "a") as log:
            log.write(f"{json.dumps(line)}\n")
        print(f"{cfg['run_id']} epoch {epoch}/{cfg['epochs']}: macro-F1 {line['macro_f1']:.3f}, "
              f"top-1 {line['top1']:.3f}, top-3 {line['top3']:.3f}, "
              f"loss {line['loss_total']:.3f}, lr {line['lr']:.2e}, "
              f"{line['photos_per_second']:.0f} photos/s, "
              f"data wait {line['data_wait_share']:.0%}, epoch {line['seconds']:.0f} s + eval "
              f"{line['eval_seconds']:.0f} s", flush=True)
        score = -math.inf if math.isnan(line["macro_f1"]) else line["macro_f1"]  # nan never wins
        if best_epoch is None or score > best_score:
            best_epoch, best_score, best_outputs = epoch, score, outputs
            best_state = {name: value.detach().to("cpu", copy=True)
                          for name, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "config": cfg, "epoch": epoch,
                        "n_classes": taxa.n_classes, "n_genus": taxa.n_genus,
                        "n_family": taxa.n_family}, out_dir / CHECKPOINT_FILE)
        if on_eval is not None:
            on_eval(epoch, line)

    model.load_state_dict(best_state)

    def outputs_on(frame):
        return predict(model, eval_loader(cfg, frame, num_workers), device, amp)

    prototypes = build_prototypes(outputs_on(frames["train"])["embedding"], frames["train"],
                                  taxa.n_classes)
    np.savez_compressed(out_dir / PROTOTYPES_FILE, plain=prototypes[0], balanced=prototypes[1])
    save_outputs(out_dir / VAL_OUTPUTS, frames["val"], best_outputs)
    save_outputs(out_dir / CALIBRATION_OUTPUTS, frames["calibration"],
                 outputs_on(frames["calibration"]))
    save_outputs(out_dir / UNKNOWN_OUTPUTS, frames["unknown"], outputs_on(frames["unknown"]))
    if frames["heldout"] is not None:
        save_outputs(out_dir / HELDOUT_OUTPUTS, frames["heldout"], outputs_on(frames["heldout"]),
                     ("split", "obs_id"))
    metrics = final_metrics(frames["val"], best_outputs, prototypes, train_counts, taxa,
                            cfg["arcface_s"])
    train_seconds = sum(stats["seconds"] for stats in epochs)
    metrics["timing"] = {
        "seconds_per_epoch": train_seconds / len(epochs),
        "epoch_seconds": [stats["seconds"] for stats in epochs],
        "photos_per_second": sum(stats["photos"] for stats in epochs) / train_seconds,
        "data_wait_share": sum(stats["wait"] for stats in epochs) / train_seconds,
        "eval_seconds": eval_seconds,
        "warm_cache_seconds": warmed["seconds"], "warm_cache_files": warmed["files"],
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20 if amp else None}
    metrics["run"] = {"run_id": cfg["run_id"], "best_epoch": best_epoch, "epochs": cfg["epochs"],
                      "train_photos": len(frames["train"]), "val_photos": len(frames["val"]),
                      "calibration_photos": len(frames["calibration"]),
                      "unknown_photos": len(frames["unknown"]),
                      "heldout_photos": 0 if frames["heldout"] is None else len(frames["heldout"]),
                      "prediction_coverage": coverage, "subset_hash": digest,
                      "device": device_name(device),
                      "num_workers": num_workers,  # the augmentation stream depends on it
                      "encoder_normalization": encoder_normalization(model)}
    return write_json(out_dir / FINAL_FILE, metrics)


def start_cuda(device):
    """Starts CUDA before resetting the peak-memory counter, which raises otherwise."""
    torch.cuda.init()
    if device.index is not None:
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)


def run_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, log_prior, drawn=None):
    """One pass over the loader; `drawn`, when given, counts the classes drawn."""
    model.train()
    amp = scaler.is_enabled()
    targets = TARGETS + (("mask", "mask_valid") if model.mask_head is not None else ())
    sums, steps, photos, wait = defaultdict(float), 0, 0, 0.0
    started = ready = time.perf_counter()
    for batch in loader:
        wait += time.perf_counter() - ready
        if drawn is not None:
            drawn += torch.bincount(batch["class_id"], minlength=len(drawn)).numpy()
        images = batch["image"].to(device, non_blocking=True)
        on_device = {name: batch[name].to(device, non_blocking=True) for name in targets}
        # float16: the T4 has no bfloat16 cores
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(images)
        out = {name: None if value is None else value.float() for name, value in out.items()}
        loss, parts = total_loss(out, on_device, cfg, log_prior=log_prior)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
    return {"seconds": time.perf_counter() - started, "photos": photos, "wait": wait,
            "losses": {name: total / steps for name, total in sums.items()},
            "lr": optimizer.param_groups[-1]["lr"]}


def make_optimizer(model, cfg):
    """AdamW with the encoder at 0.1 x lr and every other part at lr."""
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    rest = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return torch.optim.AdamW([{"params": encoder, "lr": cfg["lr"] * 0.1},
                              {"params": rest, "lr": cfg["lr"]}],
                             weight_decay=cfg["weight_decay"])


def warmup_cosine(warmup_steps, total_steps):
    """LambdaLR factor: linear warm-up to 1, then cosine decay to 0 at total_steps."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return factor


@torch.inference_mode()
def predict(model, loader, device, amp):
    """The model's outputs on every photo of the loader, in order, as float32 arrays."""
    model.eval()
    collected = defaultdict(list)
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(batch["image"].to(device, non_blocking=True))
        for name in ("embedding", "species_logits", "who_logits", "genus_logits", "family_logits"):
            if out[name] is not None:
                collected[name].append(out[name].float())  # kept on the device until the end
    return {name: torch.cat(parts).cpu().numpy() for name, parts in collected.items()}


def load_checkpoint(run_dir, device="cpu"):
    """(settings, SnakeID in eval mode, best epoch) from a run's best.pt."""
    checkpoint = torch.load(Path(run_dir) / CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    cfg = run_config(checkpoint["config"])
    model = SnakeID(cfg, checkpoint["n_classes"], checkpoint["n_genus"], checkpoint["n_family"],
                    pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return cfg, model.to(device).eval(), checkpoint.get("epoch")


def prediction_coverage(frames, mode):
    """For an input read from front-end predictions: each frame's share of photos that have one."""
    if mode not in PREDICTED_INPUT_MODES:
        return None
    return {name: float(has_input(frame, mode).mean())
            for name, frame in frames.items() if frame is not None}


def class_mix_report(frame, drawn_counts, classes):
    """Per training class: its share of the draws against its class_sampling_probs share."""
    present = np.unique(frame["class_id"])
    expected = class_sampling_probs(frame)
    drawn = drawn_counts[present] / drawn_counts.sum()
    names = classes.set_index("class_id")["species"]
    return {
        "draws": int(drawn_counts.sum()),
        "total_variation": float(np.abs(drawn - expected).sum() / 2),
        "classes": [{"class_id": int(class_id), "species": names[class_id],
                     "expected": float(share), "drawn": float(got), "ratio": float(got / share)}
                    for class_id, share, got in zip(present, expected, drawn)],
    }


def build_prototypes(embedding, frame, n_classes):
    """Plain and photographer-balanced class prototypes from training embeddings, (C, D) each."""
    embedding, labels = (torch.from_numpy(embedding),
                         frame["class_id"].to_numpy(dtype=np.int64, copy=True))
    plain = prototypes_from(embedding, labels, n_classes)
    balanced = prototypes_from(embedding, labels, n_classes, groups=photographer_groups(frame))
    return plain.numpy(), balanced.numpy()


def prototype_probs(embedding, prototypes, scale):
    """softmax(scale x cosine) of each embedding against each prototype."""
    return softmax(scale * cosine_to(torch.from_numpy(embedding),
                                     torch.from_numpy(prototypes)).numpy())


def species_scores(frame, probs, train_counts, taxa, val_counts=None):
    """species_report on a frame; a slice must pass the full set's val_counts."""
    y_true = frame["class_id"].to_numpy()
    if val_counts is None:
        val_counts = np.bincount(y_true, minlength=taxa.n_classes)
    return species_report(y_true, probs, train_counts, val_counts, taxa.tier)


def final_metrics(val, outputs, prototypes, train_counts, taxa, scale, val_counts=None):
    """Head and prototype species reports, raw safety, calibration and taxonomy on validation."""
    y_true = val["class_id"].to_numpy()
    head = softmax(outputs["species_logits"])
    plain, balanced = prototypes
    genus_head, family_head = (softmax(outputs[name]) if name in outputs else None
                               for name in ("genus_logits", "family_logits"))
    return {
        "head": species_scores(val, head, train_counts, taxa, val_counts),
        "prototype": species_scores(val, prototype_probs(outputs["embedding"], plain, scale),
                                    train_counts, taxa, val_counts),
        "prototype_balanced": species_scores(
            val, prototype_probs(outputs["embedding"], balanced, scale), train_counts, taxa,
            val_counts),
        "safety": safety_report(y_true, head, taxa.who, taxa.listed_or_venomous),
        "calibration": calibration(head, y_true),
        "taxonomy": taxonomy_report(head, genus_head, family_head, taxa.genus_of_class,
                                    taxa.family_of_class, val["genus_id"].to_numpy(),
                                    val["family_id"].to_numpy()),
    }


def save_outputs(path, frame, outputs, extra_columns=()):
    """The frame's identities and labels next to its predictions, one row per photo, no pickles."""
    def plain(column):
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            return column.to_numpy()
        return column.astype("string").fillna("").to_numpy(dtype=str)

    columns = {"paths": "path", "labels": "class_id", "species": "species", "genus_id": "genus_id",
               "family_id": "family_id", "who_risk_id": "who_risk_id",
               "observer_id": "observer_id", **{name: name for name in extra_columns}}
    np.savez_compressed(path, **{key: plain(frame[column]) for key, column in columns.items()},
                        **outputs)


def subset_hash(paths):
    """SHA-256 of the sorted subset paths joined by newlines."""
    return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()


def device_name(device):
    return (torch.cuda.get_device_name(device) if device.type == "cuda"
            else platform.processor() or device.type)


def library_versions(device=None):
    """Library versions; on CUDA also cuDNN and the GPU (the run's, or the first)."""
    import albumentations  # here, so data.py has turned off its update check first

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    on_gpu = device.type == "cuda"
    return {"python": platform.python_version(), "torch": torch.__version__,
            "timm": timm.__version__, "albumentations": albumentations.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "opencv": cv2.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if on_gpu else None,
            "gpu": device_name(device) if on_gpu else None}


def encoder_normalization(model):
    """The encoder's own input mean and std, and whether the adapter converts ImageNet input."""
    norm = model.encoder.input_norm
    return {"mean": list(norm.mean), "std": list(norm.std), "adapter": norm.active}
