"""SnakeID, the phone-sized model every M3 run trains: an encoder, an optional mask head, pooling, an L2-normalised
embedding and the species, genus, family and WHO heads. Each encoder returns spatial feature maps, because the mask
head and mask-guided pooling need a map."""
import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.data import resolve_model_data_config
from timm.models.mobilenetv3 import MobileNetV3
from torch import nn

from serpentguard.data import IMAGENET_MEAN, IMAGENET_STD

MOBILE_ENCODERS = ("mobilenetv4_conv_small", "efficientnet_b0", "mobilevit_s")
DINOV2 = "vit_small_patch14_dinov2"
STRIDE_8_STAGE = 2  # timm feature index of the stride-8 map
DINOV2_PATCH = 14
INPUT_SIZE = 320
ARCFACE_S, ARCFACE_M = 30.0, 0.3
N_GENUS, N_FAMILY = 69, 13
N_WHO = 3  # notebook 06's who_risk_id: 0 = not listed, 1 = WHO Category 2, 2 = WHO Category 1
LINEAR, ARCFACE = "linear", "arcface"
GAP, MASK = "gap", "mask"
EMBEDDING_DIM = 256
MASK_KEEP = 0.2       # background's weight in mask-guided pooling, so some context survives
WEIGHT_FLOOR = 1e-6   # keep 0 with an empty mask leaves every weight at the floor: the plain mean


class InputNormalization(nn.Module):
    """Maps the pipeline's ImageNet-normalised batch to the encoder's own mean and std (MobileViT-S expects 0-1
    pixels); the identity for ImageNet encoders. The buffers are not saved, so checkpoints keep their keys."""

    def __init__(self, mean, std, mean_in=IMAGENET_MEAN, std_in=IMAGENET_STD):
        super().__init__()
        self.mean, self.std = tuple(float(value) for value in mean), tuple(float(value) for value in std)
        self.active = not (np.allclose(self.mean, mean_in) and np.allclose(self.std, std_in))
        mean, std, mean_in, std_in = (torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1)
                                      for values in (self.mean, self.std, mean_in, std_in))
        self.register_buffer("scale", std_in / std, persistent=False)
        self.register_buffer("shift", (mean_in - mean) / std, persistent=False)

    @classmethod
    def of(cls, model):
        config = resolve_model_data_config(model)
        return cls(config["mean"], config["std"])

    def forward(self, images):
        return images * self.scale + self.shift if self.active else images


class TwoScale(nn.Module):
    """A timm CNN's stride-8 map and its final map from one forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_norm = InputNormalization.of(model)

    def forward(self, images):
        final, (fine,) = self.model.forward_intermediates(self.input_norm(images), indices=[STRIDE_8_STAGE],
                                                         intermediates_only=False)
        return [fine, final]


class PatchGrid(nn.Module):
    """A ViT's final patch tokens as one (B, C, H / patch, W / patch) map; the cls token is dropped."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.input_norm = InputNormalization.of(vit)

    def forward(self, images):
        tokens = self.vit.forward_features(self.input_norm(images))[:, self.vit.num_prefix_tokens:]
        height, width = self.vit.patch_embed.grid_size
        return [tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)]


def build_encoder(name, pretrained=True, input_size=INPUT_SIZE):
    """(module, channels): the module maps images to a list of feature maps, finest first. MobileNetV4's conv_head
    comes after pooling, so it is pruned. DINOv2 needs an input size that is a multiple of 14, such as 322."""
    if name == DINOV2:
        if input_size % DINOV2_PATCH:
            raise ValueError(f"{DINOV2} needs an input size that is a multiple of {DINOV2_PATCH}, got {input_size}")
        vit = timm.create_model(name, pretrained=pretrained, img_size=input_size, num_classes=0)
        return PatchGrid(vit), [vit.embed_dim]
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
    if isinstance(model, MobileNetV3):
        model.prune_intermediate_layers(indices=len(model.stage_ends), prune_head=True)
    return TwoScale(model), [model.feature_info[STRIDE_8_STAGE]["num_chs"], model.num_features]


class ArcFaceHead(nn.Module):
    """Cosine between the embedding and one learned direction per class; the loss applies the margin m."""

    def __init__(self, dim, n_classes, s=ARCFACE_S, m=ARCFACE_M):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding):
        return F.linear(F.normalize(embedding, dim=1), F.normalize(self.weight, dim=1))


class HierarchyHeads(nn.Module):
    """Genus, family and WHO-category logits; taxonomy=False drops genus and family, never the WHO head."""

    def __init__(self, dim, n_genus=N_GENUS, n_family=N_FAMILY, n_who=N_WHO, taxonomy=True):
        super().__init__()
        self.genus = nn.Linear(dim, n_genus) if taxonomy else None
        self.family = nn.Linear(dim, n_family) if taxonomy else None
        self.who = nn.Linear(dim, n_who)

    def forward(self, features):
        return {"genus_logits": None if self.genus is None else self.genus(features),
                "family_logits": None if self.family is None else self.family(features),
                "who_logits": self.who(features)}


def means_by(values, index, n_rows):
    """Row means of values grouped by index; a group with no rows gets zeros."""
    sums = values.new_zeros(n_rows, values.shape[1]).index_add_(0, index, values)
    counts = torch.bincount(index, minlength=n_rows).clamp_min(1)
    return sums / counts.unsqueeze(1).to(values.dtype)


def prototypes_from(embeddings, labels, n_classes, groups=None):
    """One unit prototype per class: the mean of its normalised embeddings. With groups (photographer ids), each
    (class, photographer) pair is averaged first, so one photographer can't set a class's prototype. A class with no
    photos gets a zero row."""
    embeddings = F.normalize(embeddings, dim=1)
    labels = torch.as_tensor(labels, device=embeddings.device).long()
    if groups is not None:
        group_names, group_codes = np.unique(np.asarray(groups), return_inverse=True)
        pair_ids = labels * len(group_names) + torch.as_tensor(group_codes, device=labels.device).long()
        pairs, pair_of_photo = torch.unique(pair_ids, return_inverse=True)
        embeddings = means_by(embeddings, pair_of_photo, len(pairs))
        labels = pairs // len(group_names)
    return F.normalize(means_by(embeddings, labels, n_classes), dim=1)


class MaskHead(nn.Module):
    """Snake-mask logits (B, 1, H, W) at the input size, from an FPN-style merge of the stride-8 and stride-32 maps."""

    def __init__(self, channels):
        super().__init__()
        fine, coarse = channels
        self.lateral_fine = nn.Conv2d(fine, fine, 1)
        self.lateral_coarse = nn.Conv2d(coarse, fine, 1)
        self.smooth = nn.Sequential(nn.Conv2d(fine, fine, 3, padding=1, bias=False), nn.BatchNorm2d(fine),
                                    nn.ReLU(inplace=True))
        self.logits = nn.Conv2d(fine, 1, 1)

    def forward(self, maps, size):
        fine, coarse = maps
        coarse = F.interpolate(self.lateral_coarse(coarse), size=fine.shape[-2:], mode="bilinear", align_corners=False)
        merged = self.smooth(self.lateral_fine(fine) + coarse)
        return F.interpolate(self.logits(merged), size=tuple(size), mode="bilinear", align_corners=False)


def mask_guided_pool(features, mask_logits, keep=MASK_KEEP):
    """sum(w * f) / sum(w) with w = keep + (1 - keep) * sigmoid(mask); a flat mask gives global average pooling."""
    snake = F.interpolate(torch.sigmoid(mask_logits), size=features.shape[-2:], mode="area")
    weights = (keep + (1 - keep) * snake).clamp_min(WEIGHT_FLOOR)
    return (features * weights).sum((2, 3)) / weights.sum((2, 3))


class SnakeID(nn.Module):
    """forward(images) returns embedding (unit rows), features, species_logits, genus_logits and family_logits (None
    without hierarchy), who_logits and mask_logits (None without a mask head). The linear head reads features, the
    ArcFace head the embedding. F4 trains the mask head but pools with "gap"; F5 pools with "mask"."""

    def __init__(self, encoder_name, n_classes, n_genus, n_family, head=LINEAR, embedding_dim=EMBEDDING_DIM,
                 hierarchy=True, mask_head=False, pooling=GAP, mask_keep=MASK_KEEP, arcface_s=ARCFACE_S,
                 arcface_m=ARCFACE_M, pretrained=True, input_size=INPUT_SIZE):
        super().__init__()
        if pooling == MASK and not mask_head:
            raise ValueError("pooling='mask' needs mask_head=True")
        self.encoder, channels = build_encoder(encoder_name, pretrained=pretrained, input_size=input_size)
        self.mask_head = MaskHead(channels) if mask_head else None
        self.head, self.pooling, self.mask_keep = head, pooling, mask_keep
        self.embed = nn.Sequential(nn.Linear(channels[-1], embedding_dim, bias=False), nn.BatchNorm1d(embedding_dim))
        self.species_head = (ArcFaceHead(embedding_dim, n_classes, arcface_s, arcface_m) if head == ARCFACE
                             else nn.Linear(embedding_dim, n_classes))
        self.hierarchy_heads = HierarchyHeads(embedding_dim, n_genus, n_family, N_WHO, taxonomy=hierarchy)

    @classmethod
    def from_config(cls, cfg, n_classes, n_genus, n_family, pretrained=True):
        return cls(cfg.encoder, n_classes, n_genus, n_family, head=cfg.head, embedding_dim=cfg.embedding_dim,
                   hierarchy=cfg.hierarchy, mask_head=cfg.mask_head, pooling=cfg.pooling, mask_keep=cfg.mask_keep,
                   arcface_s=cfg.arcface_s, arcface_m=cfg.arcface_m, pretrained=pretrained, input_size=cfg.input_size)

    def forward(self, images):
        maps = self.encoder(images)
        mask_logits = None if self.mask_head is None else self.mask_head(maps, images.shape[-2:])
        pooled = (mask_guided_pool(maps[-1], mask_logits, self.mask_keep) if self.pooling == MASK
                  else maps[-1].mean((2, 3)))
        features = self.embed(pooled)
        embedding = F.normalize(features, dim=1)
        species_logits = (self.species_head.s * self.species_head(embedding) if self.head == ARCFACE
                          else self.species_head(features))
        return {"embedding": embedding, "features": features, "species_logits": species_logits,
                **self.hierarchy_heads(features), "mask_logits": mask_logits}

    @staticmethod
    def cosine_to(embedding, prototypes):
        """(B, n_classes) cosine of each embedding to each prototype row; a zero row scores 0."""
        return F.normalize(embedding, dim=1) @ F.normalize(prototypes, dim=1).T
