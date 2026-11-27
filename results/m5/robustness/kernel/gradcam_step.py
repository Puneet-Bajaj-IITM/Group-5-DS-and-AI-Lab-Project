"""Grad-CAM on the frozen species classifier (M5_KD, plain pass) over the clean held-out photos that have a mask.

Maps: the gradient of the species logit of the frozen answer's class (flip + prior, predictions_heldout.csv) with
respect to a layer's output; weights = spatially averaged gradients, map = ReLU(sum of weighted channels), upsampled
bilinearly to the input. Layers (timm MobileViT-S inside the encoder): "final" = final_conv, the last conv block
(640 x 10 x 10 at 320 input; asked for), and as layer checks "stage4" = stages.4 (the last MobileViT block, before
final_conv), "stride16" = stages.3 and "stride8" = stages.2. A laptop look on 37 local validation photos (not test
photos) found the final block's maps sitting on the padding and the photo border, so the earlier layers are kept
as checks. Negative control: the same architecture with every weight re-initialised (seed 42), same inputs and
target classes.

Per photo: the share of the map's mass inside the U-Net mask (the mask as the classifier's own crop carries it into
the input), the share of the input the mask covers, and the share of the input that is letterbox padding.
"""
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BATCH = 32
SEED = 42
PAD_TOLERANCE = 0.6  # grey levels, after undoing the normalisation
LAYERS = {"final": "final_conv", "stage4": "stages.4", "stride16": "stages.3", "stride8": "stages.2"}
SAVED_LAYERS = ("final", "stride16", "stride8")


def grad_cams(model, images, targets):
    """{layer: map}, logits; targets < 0 (the laptop stand-in only) take the model's own plain answer."""
    modules = dict(model.encoder.model.named_modules())
    found, hooks = {}, []
    for name, module_name in LAYERS.items():
        def keep(module, inputs, output, name=name):
            output.retain_grad()
            found[name] = output
        hooks.append(modules[module_name].register_forward_hook(keep))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(images)["species_logits"]
        targets = torch.where(targets < 0, logits.argmax(1), targets)
        logits.gather(1, targets[:, None]).sum().backward()
    finally:
        for hook in hooks:
            hook.remove()
    cams = {}
    for name, m in found.items():
        weights = m.grad.mean((2, 3), keepdim=True)
        cam = F.relu((weights * m).sum(1, keepdim=True))
        cams[name] = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0].detach()
    return cams, logits.detach()


def randomised(model):
    """Every parameter re-drawn (modules' own reset_parameters, ArcFace weight by xavier), BN statistics reset."""
    torch.manual_seed(SEED)
    control = copy.deepcopy(model)
    for module in control.modules():
        if module is not control and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    torch.nn.init.xavier_uniform_(control.species_head.weight)
    return control.eval()


def mass_inside(cam, mask):
    total = cam.sum((1, 2))
    return torch.where(total > 0, (cam * mask).sum((1, 2)) / total.clamp_min(1e-12), torch.full_like(total, float("nan")))


def run(ctx, frame):
    np, pd, data, device, log = ctx["np"], ctx["pd"], ctx["data"], ctx["device"], ctx["log"]
    out = ctx["out"] / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    cfg, model, _ = ctx["load_checkpoint"](ctx["m5_dir"], device)
    model = model.float().eval()
    control = randomised(model)
    trained = dict(model.named_parameters())
    unchanged = [n for n, p in control.named_parameters() if torch.equal(p, trained[n])]
    ctx["status"]["checks"]["gradcam_control_unchanged_parameters"] = unchanged

    targets = ctx["lists"]["gradcam_targets"].set_index("path")["target_class"]
    chosen = set(ctx["lists"]["gradcam_photos"]["path"])
    masked = frame[frame["mask_valid"]].reset_index(drop=True)
    masked = masked[masked["path"].isin(targets.index)].reset_index(drop=True)
    mode = data.input_mode(cfg)
    dataset = data.SnakeDataset(data.scored_frame(masked, mode), train=False, mode=mode, size=cfg["input_size"])
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=ctx["loader_workers"])
    mean = torch.tensor(data.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(data.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    grey = torch.tensor(data.GREY_RGB, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    rows, kept, start = [], {}, 0
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device).float()
        mask = batch["mask"].to(device).float()[:, 0]
        paths = list(masked["path"].iloc[start:start + len(images)])
        start += len(images)
        target = torch.tensor(targets.loc[paths].to_numpy(int), device=device)
        cams, logits = grad_cams(model, images, target)
        target = torch.where(target < 0, logits.argmax(1), target)
        control_cams, _ = grad_cams(control, images, target)
        if batch_index == 0:
            with torch.no_grad():
                own = model(images)["species_logits"]
            ctx["status"]["checks"]["gradcam_forward_vs_model"] = float((own - logits).abs().max())
            assert float((own - logits).abs().max()) < 1e-3, "the written-out forward differs from the model's"
        canvas = (images * std + mean) * 255
        pad = ((canvas - grey).abs() < PAD_TOLERANCE).all(1).float()
        record = {"path": paths, "target_class": target.cpu().numpy(),
                  "plain_pred": logits.argmax(1).cpu().numpy(),
                  "mask_share_of_input": mask.mean((1, 2)).cpu().numpy(),
                  "pad_share_of_input": pad.mean((1, 2)).cpu().numpy()}
        for name, cam in cams.items():
            record[f"mass_in_mask_{name}"] = mass_inside(cam, mask).cpu().numpy()
            record[f"mass_on_pad_{name}"] = mass_inside(cam, pad).cpu().numpy()
        for name, cam in control_cams.items():
            record[f"control_mass_in_mask_{name}"] = mass_inside(cam, mask).cpu().numpy()
            record[f"control_mass_on_pad_{name}"] = mass_inside(cam, pad).cpu().numpy()
        rows.append(pd.DataFrame(record))
        for i, path in enumerate(paths):
            if path in chosen:
                kept[path] = {"canvas": canvas[i].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                              "mask": mask[i].byte().cpu().numpy(),
                              **{f"cam_{k}": cams[k][i].cpu().numpy().astype(np.float16) for k in SAVED_LAYERS},
                              **{f"control_cam_{k}": control_cams[k][i].cpu().numpy().astype(np.float16)
                                 for k in SAVED_LAYERS}}
        if batch_index % 100 == 0:
            log(f"gradcam: {start:,}/{len(masked):,}")
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / "gradcam_mass.csv", index=False)
    order = sorted(kept)
    np.savez_compressed(out / "gradcam_selected.npz", paths=np.array(order),
                        **{key: np.stack([kept[p][key] for p in order]) for key in kept[order[0]]} if order else {})
    ctx["status"]["checks"]["gradcam"] = {"photos": len(table), "selected": len(order),
                                          "median_mass_in_mask_final": float(table["mass_in_mask_final"].median()),
                                          "median_mask_share": float(table["mask_share_of_input"].median()),
                                          "median_control_mass_final": float(
                                              table["control_mass_in_mask_final"].median())}
    log("gradcam", ctx["status"]["checks"]["gradcam"])
