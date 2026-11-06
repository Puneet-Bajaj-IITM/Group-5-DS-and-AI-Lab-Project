"""SnakeID, the phone-sized model every run trains, its training losses and prototype scoring.

SnakeID: an encoder, an optional mask head, pooling, an L2-normalised embedding and the species,
genus, family and WHO heads. Encoders return feature maps: the mask head and mask pooling need one.
"""
import math

import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.data import resolve_model_data_config
from timm.models.mobilenetv3 import MobileNetV3
from torch import nn

from serpentguard.data import IMAGENET_MEAN, IMAGENET_STD, run_config

MOBILE_ENCODERS = ("mobilenetv4_conv_small", "efficientnet_b0", "mobilevit_s")
DINOV2 = "vit_small_patch14_dinov2"
ARCFACE_S = 30.0    # the default cosine scale of the ArcFace head
STRIDE_8_STAGE = 2  # timm feature index of the stride-8 map


class InputNormalization(nn.Module):
    """Maps the ImageNet-normalised batch to the encoder's own mean and std (MobileViT-S: 0-1)."""
    # the identity for ImageNet encoders; the buffers are not saved, so checkpoints keep their keys

    def __init__(self, mean, std, mean_in=IMAGENET_MEAN, std_in=IMAGENET_STD):
        super().__init__()
        self.mean, self.std = (tuple(float(value) for value in mean),
                               tuple(float(value) for value in std))
        self.active = not (np.allclose(self.mean, mean_in) and np.allclose(self.std, std_in))
        mean, std, mean_in, std_in = (torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1)
                                      for values in (self.mean, self.std, mean_in, std_in))
        self.register_buffer("scale", std_in / std, persistent=False)
        self.register_buffer("shift", (mean_in - mean) / std, persistent=False)

    def forward(self, images):
        return images * self.scale + self.shift if self.active else images


def input_norm_of(model):
    config = resolve_model_data_config(model)
    return InputNormalization(config["mean"], config["std"])


class TwoScale(nn.Module):
    """A timm CNN's stride-8 map and its final map from one forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_norm = input_norm_of(model)

    def forward(self, images):
        final, (fine,) = self.model.forward_intermediates(
            self.input_norm(images), indices=[STRIDE_8_STAGE], intermediates_only=False)
        return [fine, final]


class PatchGrid(nn.Module):
    """A ViT's final patch tokens as one (B, C, H / patch, W / patch) map, cls token dropped."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.input_norm = input_norm_of(vit)

    def forward(self, images):
        tokens = self.vit.forward_features(self.input_norm(images))[:, self.vit.num_prefix_tokens:]
        height, width = self.vit.patch_embed.grid_size
        return [tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)]


def build_encoder(name, pretrained=True, input_size=320):
    """(module, channels): the module maps images to a list of feature maps, finest first."""
    # MobileNetV4's conv_head comes after pooling, so it is pruned; DINOv2 needs a multiple of 14
    if name == DINOV2:
        if input_size % 14:
            raise ValueError(f"{DINOV2} needs an input size that is a multiple of 14, got "
                             f"{input_size}")
        vit = timm.create_model(name, pretrained=pretrained, img_size=input_size, num_classes=0)
        return PatchGrid(vit), [vit.embed_dim]
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
    if isinstance(model, MobileNetV3):
        model.prune_intermediate_layers(indices=len(model.stage_ends), prune_head=True)
    return TwoScale(model), [model.feature_info[STRIDE_8_STAGE]["num_chs"], model.num_features]


class ArcFaceHead(nn.Module):
    """Cosine between the embedding and one learned direction per class; the loss adds margin m."""

    def __init__(self, dim, n_classes, s=ARCFACE_S, m=0.3):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding):
        return F.linear(F.normalize(embedding, dim=1), F.normalize(self.weight, dim=1))


class HierarchyHeads(nn.Module):
    """Genus, family and WHO-category logits; taxonomy=False drops genus and family, not WHO."""

    def __init__(self, dim, n_genus, n_family, taxonomy=True):
        super().__init__()
        self.genus = nn.Linear(dim, n_genus) if taxonomy else None
        self.family = nn.Linear(dim, n_family) if taxonomy else None
        self.who = nn.Linear(dim, 3)  # not listed, WHO Category 2, WHO Category 1

    def forward(self, features):
        return {"genus_logits": None if self.genus is None else self.genus(features),
                "family_logits": None if self.family is None else self.family(features),
                "who_logits": self.who(features)}


class MaskHead(nn.Module):
    """Snake-mask logits (B, 1, H, W) from an FPN-style merge of the stride-8 and -32 maps."""

    def __init__(self, channels):
        super().__init__()
        fine, coarse = channels
        self.lateral_fine = nn.Conv2d(fine, fine, 1)
        self.lateral_coarse = nn.Conv2d(coarse, fine, 1)
        self.smooth = nn.Sequential(nn.Conv2d(fine, fine, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(fine), nn.ReLU(inplace=True))
        self.logits = nn.Conv2d(fine, 1, 1)

    def forward(self, maps, size):
        fine, coarse = maps
        coarse = F.interpolate(self.lateral_coarse(coarse), size=fine.shape[-2:], mode="bilinear",
                               align_corners=False)
        merged = self.smooth(self.lateral_fine(fine) + coarse)
        return F.interpolate(self.logits(merged), size=tuple(size), mode="bilinear",
                             align_corners=False)


def mask_guided_pool(features, mask_logits, keep=0.2):
    """sum(w * f) / sum(w), w = keep + (1 - keep) * sigmoid(mask); a flat mask gives the mean."""
    snake = F.interpolate(torch.sigmoid(mask_logits), size=features.shape[-2:], mode="area")
    weights = (keep + (1 - keep) * snake).clamp_min(1e-6)  # keep 0, empty mask: the plain mean
    return (features * weights).sum((2, 3)) / weights.sum((2, 3))


class SnakeID(nn.Module):
    """The model; forward(images) returns a dict of embedding, features and each head's logits."""
    # genus and family logits are None without hierarchy, mask_logits None without a mask head.
    # The linear head reads features, the ArcFace head the embedding. F4 trains the mask head but
    # pools with "gap"; F5 pools with "mask".

    def __init__(self, cfg, n_classes, n_genus, n_family, pretrained=True):
        super().__init__()
        if cfg["pooling"] == "mask" and not cfg["mask_head"]:
            raise ValueError("pooling 'mask' needs mask_head true")
        dim = cfg["embedding_dim"]
        self.encoder, channels = build_encoder(cfg["encoder"], pretrained=pretrained,
                                               input_size=cfg["input_size"])
        self.mask_head = MaskHead(channels) if cfg["mask_head"] else None
        self.head, self.pooling, self.mask_keep = cfg["head"], cfg["pooling"], cfg["mask_keep"]
        self.embed = nn.Sequential(nn.Linear(channels[-1], dim, bias=False), nn.BatchNorm1d(dim))
        self.species_head = (ArcFaceHead(dim, n_classes, cfg["arcface_s"], cfg["arcface_m"])
                             if self.head == "arcface" else nn.Linear(dim, n_classes))
        self.hierarchy_heads = HierarchyHeads(dim, n_genus, n_family, taxonomy=cfg["hierarchy"])

    def forward(self, images):
        maps = self.encoder(images)
        mask_logits = None if self.mask_head is None else self.mask_head(maps, images.shape[-2:])
        pooled = (mask_guided_pool(maps[-1], mask_logits, self.mask_keep) if self.pooling == "mask"
                  else maps[-1].mean((2, 3)))
        features = self.embed(pooled)
        embedding = F.normalize(features, dim=1)
        species_logits = (self.species_head.s * self.species_head(embedding)
                          if self.head == "arcface" else self.species_head(features))
        return {"embedding": embedding, "features": features, "species_logits": species_logits,
                **self.hierarchy_heads(features), "mask_logits": mask_logits}


def build_model(cfg, n_classes=184, n_genus=69, n_family=13, pretrained=True):
    """SnakeID for a run's settings (missing ones take data.DEFAULTS)."""
    return SnakeID(run_config(cfg), n_classes, n_genus, n_family, pretrained=pretrained)


def means_by(values, index, n_rows):
    """Row means of values grouped by index; a group with no rows gets zeros."""
    sums = values.new_zeros(n_rows, values.shape[1]).index_add_(0, index, values)
    counts = torch.bincount(index, minlength=n_rows).clamp_min(1)
    return sums / counts.unsqueeze(1).to(values.dtype)


def prototypes_from(embeddings, labels, n_classes, groups=None):
    """One unit prototype per class: the mean of its normalised embeddings (no photos: zeros)."""
    # with groups (photographer ids) each (class, photographer) pair is averaged first, so one
    # photographer can't set a class's prototype
    embeddings = F.normalize(embeddings, dim=1)
    labels = torch.as_tensor(labels, device=embeddings.device).long()
    if groups is not None:
        group_names, group_codes = np.unique(np.asarray(groups), return_inverse=True)
        pair_ids = (labels * len(group_names)
                    + torch.as_tensor(group_codes, device=labels.device).long())
        pairs, pair_of_photo = torch.unique(pair_ids, return_inverse=True)
        embeddings = means_by(embeddings, pair_of_photo, len(pairs))
        labels = pairs // len(group_names)
    return F.normalize(means_by(embeddings, labels, n_classes), dim=1)


def cosine_to(embedding, prototypes):
    """(B, n_classes) torch cosine of each embedding to each prototype; a zero row scores 0."""
    return F.normalize(embedding, dim=1) @ F.normalize(prototypes, dim=1).T


def unit(vectors):
    """Rows scaled to length 1 (numpy); a zero row stays zero."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def masked_cosine(embeddings, prototypes):
    """(N, C) numpy cosine to each prototype; a zero prototype row scores -inf, so never wins."""
    embeddings, prototypes = unit(embeddings), np.asarray(prototypes, dtype=float)
    norms = np.linalg.norm(prototypes, axis=1)
    active = norms > 0
    scores = np.full((len(embeddings), len(prototypes)), -np.inf)
    scores[:, active] = embeddings @ (prototypes[active] / norms[active, None]).T
    return scores


def arcface_loss(cosine, labels, s, m):
    """Additive angular margin (Deng et al. 2019), with the cos - sin(pi - m) * m fallback."""
    cosine = cosine.clamp(-1.0, 1.0)
    sine = (1.0 - cosine ** 2).clamp_min(1e-7).sqrt()  # keeps the gradient finite at cosine = +-1
    phi = cosine * math.cos(m) - sine * math.sin(m)
    phi = torch.where(cosine > math.cos(math.pi - m), phi, cosine - math.sin(math.pi - m) * m)
    one_hot = F.one_hot(labels, cosine.shape[1]).bool()
    return F.cross_entropy(s * torch.where(one_hot, phi, cosine), labels)


def batch_hard_triplet(embedding, labels, margin=0.2):
    """Batch-hard triplet loss on cosine distance; anchors lacking a positive or negative skip."""
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


def logit_adjusted_ce(logits, labels, log_prior, tau=1.0):
    """Menon et al. 2021: cross-entropy on logits + tau * log class frequency."""
    return F.cross_entropy(logits + tau * log_prior, labels)


def danger_weighted_ce(who_logits, who_true, weight=5.0):
    """CE plus (weight - 1) * -log(1 - p_not_listed) on WHO-listed photos."""
    plain = F.cross_entropy(who_logits, who_true, reduction="none")
    p_not_listed = F.softmax(who_logits, dim=1)[:, 0]
    listed = (who_true != 0).float()
    extra = -(weight - 1.0) * listed * torch.log((1.0 - p_not_listed).clamp_min(1e-6))
    return (plain + extra).mean()


def dice_bce(mask_logits, masks, valid):
    """BCE-with-logits plus soft Dice per sample, averaged over the samples whose mask is valid."""
    if not valid.any():
        return mask_logits.sum() * 0
    logits, targets = mask_logits[valid], masks[valid].to(mask_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").flatten(1).mean(1)
    probs = torch.sigmoid(logits).flatten(1)
    flat = targets.flatten(1)
    dice = 1.0 - (2 * (probs * flat).sum(1) + 1.0) / (probs.sum(1) + flat.sum(1) + 1.0)
    return (bce + dice).mean()


def total_loss(out, batch, cfg, log_prior=None):
    """(loss, parts): the run's weighted sum, and each unweighted term as a float for logging."""
    species, = [name for name in cfg["losses"] if name in ("ce", "la_ce", "arcface")]  # exactly one
    logits, labels = out["species_logits"], batch["class_id"]
    if species == "arcface":
        terms = {"species": arcface_loss(logits / cfg["arcface_s"], labels, cfg["arcface_s"],
                                         cfg["arcface_m"])}
    elif species == "la_ce":
        terms = {"species": logit_adjusted_ce(logits, labels, log_prior)}
    else:
        terms = {"species": F.cross_entropy(logits, labels)}
    total = terms["species"]
    if out["genus_logits"] is not None and out["family_logits"] is not None:
        terms["genus"] = F.cross_entropy(out["genus_logits"], batch["genus_id"])
        terms["family"] = F.cross_entropy(out["family_logits"], batch["family_id"])
        total = total + 0.3 * (terms["genus"] + terms["family"])
    terms["who"] = danger_weighted_ce(out["who_logits"], batch["who_risk_id"], cfg["danger_weight"])
    total = total + 0.5 * terms["who"]
    if "triplet" in cfg["losses"]:
        terms["triplet"] = batch_hard_triplet(out["embedding"], batch["class_id"],
                                              cfg["triplet_margin"])
        total = total + cfg["triplet_weight"] * terms["triplet"]
    if out["mask_logits"] is not None:
        terms["mask"] = dice_bce(out["mask_logits"], batch["mask"], batch["mask_valid"])
        total = total + cfg["mask_weight"] * terms["mask"]
    return total, {**{name: value.item() for name, value in terms.items()}, "total": total.item()}
