"""Training losses: ArcFace, batch-hard triplet, logit-adjusted CE, danger-weighted WHO CE, Dice + BCE, and the
run's total_loss."""
import math

import torch
import torch.nn.functional as F

CE, LA_CE, ARCFACE, TRIPLET, KD = "ce", "la_ce", "arcface", "triplet", "kd"  # run_data refuses kd (no teacher yet)
SPECIES_LOSSES = (CE, LA_CE, ARCFACE)
TAXONOMY_WEIGHT = 0.3           # on genus CE + family CE
WHO_WEIGHT = 0.5
LA_TAU = 1.0
PROB_EPS = 1e-6                 # keeps log(1 - p_not_listed) finite
SINE_EPS = 1e-7                 # keeps the ArcFace sine's gradient finite at cosine = +-1
DICE_SMOOTH = 1.0
NOT_LISTED = 0


def arcface_loss(cosine, labels, s, m):
    """Additive angular margin (Deng et al. 2019), with the cos - sin(pi - m) * m fallback past pi."""
    cosine = cosine.clamp(-1.0, 1.0)
    sine = (1.0 - cosine ** 2).clamp_min(SINE_EPS).sqrt()
    phi = cosine * math.cos(m) - sine * math.sin(m)
    phi = torch.where(cosine > math.cos(math.pi - m), phi, cosine - math.sin(math.pi - m) * m)
    one_hot = F.one_hot(labels, cosine.shape[1]).bool()
    return F.cross_entropy(s * torch.where(one_hot, phi, cosine), labels)


def batch_hard_triplet(embedding, labels, margin=0.2):
    """Hardest positive and hardest negative per anchor on cosine distance; anchors lacking either are skipped."""
    distance = 1.0 - embedding @ embedding.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask, negative_mask = same & ~eye, ~same
    usable = positive_mask.any(1) & negative_mask.any(1)
    if not usable.any():
        return embedding.sum() * 0
    hardest_positive = distance.masked_fill(~positive_mask, -torch.inf).max(1).values
    hardest_negative = distance.masked_fill(~negative_mask, torch.inf).min(1).values
    return F.relu(hardest_positive - hardest_negative + margin)[usable].mean()


def logit_adjusted_ce(logits, labels, log_prior, tau=LA_TAU):
    """Menon et al. 2021: cross-entropy on logits + tau * log class frequency."""
    return F.cross_entropy(logits + tau * log_prior, labels)


def danger_weighted_ce(who_logits, who_true, weight=5.0):
    """CE plus (weight - 1) * -log(1 - p_not_listed) on WHO-listed photos."""
    plain = F.cross_entropy(who_logits, who_true, reduction="none")
    p_not_listed = F.softmax(who_logits, dim=1)[:, NOT_LISTED]
    listed = (who_true != NOT_LISTED).float()
    extra = -(weight - 1.0) * listed * torch.log((1.0 - p_not_listed).clamp_min(PROB_EPS))
    return (plain + extra).mean()


def dice_bce(mask_logits, masks, valid):
    """BCE-with-logits plus soft Dice per sample, averaged over the samples whose mask is valid."""
    if not valid.any():
        return mask_logits.sum() * 0
    logits, targets = mask_logits[valid], masks[valid].to(mask_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").flatten(1).mean(1)
    probs = torch.sigmoid(logits).flatten(1)
    flat = targets.flatten(1)
    dice = 1.0 - (2 * (probs * flat).sum(1) + DICE_SMOOTH) / (probs.sum(1) + flat.sum(1) + DICE_SMOOTH)
    return (bce + dice).mean()


def total_loss(out, batch, cfg, log_prior=None, teacher_probs=None):
    """(loss, parts): the run's weighted sum, and each unweighted term as a float for logging. teacher_probs is
    ignored: kd (D2) was never built, and run_data refuses it."""
    species, = [name for name in cfg.losses if name in SPECIES_LOSSES]  # exactly one, or a ValueError
    logits, labels = out["species_logits"], batch["class_id"]
    if species == ARCFACE:
        terms = {"species": arcface_loss(logits / cfg.arcface_s, labels, cfg.arcface_s, cfg.arcface_m)}
    elif species == LA_CE:
        terms = {"species": logit_adjusted_ce(logits, labels, log_prior)}
    else:
        terms = {"species": F.cross_entropy(logits, labels)}
    total = terms["species"]
    if out["genus_logits"] is not None and out["family_logits"] is not None:
        terms["genus"] = F.cross_entropy(out["genus_logits"], batch["genus_id"])
        terms["family"] = F.cross_entropy(out["family_logits"], batch["family_id"])
        total = total + TAXONOMY_WEIGHT * (terms["genus"] + terms["family"])
    terms["who"] = danger_weighted_ce(out["who_logits"], batch["who_risk_id"], cfg.danger_weight)
    total = total + WHO_WEIGHT * terms["who"]
    if TRIPLET in cfg.losses:
        terms["triplet"] = batch_hard_triplet(out["embedding"], batch["class_id"], cfg.triplet_margin)
        total = total + cfg.triplet_weight * terms["triplet"]
    if out["mask_logits"] is not None:
        terms["mask"] = dice_bce(out["mask_logits"], batch["mask"], batch["mask_valid"])
        total = total + cfg.mask_weight * terms["mask"]
    return total, {**{name: value.item() for name, value in terms.items()}, "total": total.item()}
