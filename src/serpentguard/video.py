"""Synthetic clips by M1 section 3.4's recipe, and the live-video steps the app shares.

The shared steps: the frame schedule, 4-of-6 voting and how long a label must stay on screen.
"""
import math
import os
from collections import Counter
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from serpentguard.metrics import assert_no_test_paths

os.environ.setdefault("YOLO_OFFLINE", "1")  # ultralytics: no pip installs, no analytics

# The clip recipe (M1 3.4); every value is drawn per clip from its seed.
SEED = 42
CLIP_SECONDS, CLIP_FPS = 10, 15
CLIP_MAX_SIDE = 640               # a phone video frame's long side
LEAF_COVER_RANGE = (0.30, 0.70)   # the share of the snake the leaves hide
PATH_KEYPOINTS = 4                # camera poses over the clip, cosine-eased between
PAN_MAX = 0.12                    # share of the frame's width or height
ZOOM_RANGE = (1.0, 1.35)
ROTATION_MAX_DEG = 12.0
SHAKE_RANGE = (0.002, 0.012)      # hand shake's standard deviation, share of the long side
SHAKE_ROTATION_DEG = 0.5          # rotation shake's standard deviation at the largest shake
SHAKE_PERSISTENCE = 0.7           # AR(1) coefficient
EXPOSURE_RANGE = (0.3, 1.0)       # share of the frame interval the shutter is open
MAX_BLUR_PX = 25
LIGHT_GAIN_RANGE = (0.1, 1.0)     # scene light in linear units; 1 is the photo as taken
SRGB_GAMMA = 2.2
NOISE_SIGMA_RANGE = (1.5, 4.0)    # at full light, 0-255 scale; it grows as 1 / sqrt(light gain)
JPEG_QUALITY_RANGE = (40, 90)
LEAF_LENGTH_RANGE = (0.25, 0.55)  # share of sqrt(snake area)
LEAF_ASPECT_RANGE = (2.0, 3.5)    # length over width
LEAF_TRIES = 300
LEAF_POINTS = 24                  # outline points along each edge of a leaf
LEAF_GREEN_LOW, LEAF_GREEN_HIGH = (25, 70, 15), (90, 150, 60)
MIDRIB_DARKEN = 0.7
CENTRE_REGION = 0.5               # no usable mask: leaves cover a central ellipse this size

# A label shows when 4 of the last 6 processed frames agree; it is stable after 1 s on screen.
VOTE_K, VOTE_N = 4, 6
VOTE = "vote_4_of_6"
STABLE_SECONDS = 1.0
TARGET_FPS = 10                   # the live-video target: at most 10 frames a second are processed


def load_raw_rgb(path):
    """The photo as stored, without load_rgb's exposure fix: the scene the camera films."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def synthesize_clip(img, mask=None, seconds=CLIP_SECONDS, fps=CLIP_FPS, *, seed):
    """M1 3.4's clip from one photo (.frames, .params), every parameter drawn from `seed`."""
    # the leaves sit in the scene, so they move with it
    rng = np.random.default_rng(seed)
    small = phone_frame(img)
    if small is not img and mask is not None:
        mask = cv2.resize(mask.astype(np.uint8), small.shape[1::-1],
                          interpolation=cv2.INTER_NEAREST)
    img = small
    height, width = img.shape[:2]
    n_frames = int(round(seconds * fps))
    keypoints = {"pan": rng.uniform(-PAN_MAX, PAN_MAX, (PATH_KEYPOINTS, 2)),
                 "zoom": rng.uniform(*ZOOM_RANGE, PATH_KEYPOINTS),
                 "rotation_deg": rng.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG, PATH_KEYPOINTS)}
    shake, exposure, gain = (rng.uniform(*SHAKE_RANGE), rng.uniform(*EXPOSURE_RANGE),
                             rng.uniform(*LIGHT_GAIN_RANGE))
    noise = rng.uniform(*NOISE_SIGMA_RANGE) / math.sqrt(gain)
    quality = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
    target = rng.uniform(*LEAF_COVER_RANGE)
    scene, leaves = add_leaves(img, mask, target, rng)
    matrices = camera_path(keypoints, shake, n_frames, (width, height), rng)
    centre = np.array([width / 2, height / 2, 1.0])
    brightness = gain ** (1 / SRGB_GAMMA)
    frames = []
    for index, matrix in enumerate(matrices):
        frame = cv2.warpAffine(scene, matrix, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        neighbour = matrices[index - 1] if index else matrices[min(1, n_frames - 1)]
        frame = motion_blur(frame, (matrix - neighbour) @ centre * exposure)
        frame = frame.astype(np.float32) * brightness + rng.normal(0, noise, frame.shape)
        _, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8),
                                                       cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
        frames.append(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    params = {"seed": int(seed), "seconds": seconds, "fps": fps, "frames": n_frames,
              "size": [width, height],
              **{name: np.round(values, 4).tolist() for name, values in keypoints.items()},
              "shake_share": float(shake), "exposure_share": float(exposure),
              "light_gain": float(gain), "noise_sigma": float(noise), "jpeg_quality": quality,
              "leaf_cover_target": float(target), **leaves}
    return SimpleNamespace(frames=frames, params=params)


def phone_frame(image, max_side=CLIP_MAX_SIDE):
    """The image scaled down to max_side on its long side, as a phone films it; smaller stays."""
    scale = max_side / max(image.shape[:2])
    if scale >= 1:
        return image
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def camera_path(keypoints, shake, n_frames, size, rng):
    """One affine matrix per frame: eased pan, zoom and rotation, plus AR(1) hand shake."""
    width, height = size
    times = np.linspace(0, 1, n_frames)
    pan = (np.stack([eased(keypoints["pan"][:, axis], times) for axis in range(2)], axis=1)
           * (width, height))
    zoom, rotation = eased(keypoints["zoom"], times), eased(keypoints["rotation_deg"], times)
    steps = rng.normal(size=(n_frames, 3))
    jitter = np.empty_like(steps)
    jitter[0] = steps[0]  # a clip with no frames is refused here
    for index in range(1, n_frames):
        jitter[index] = (SHAKE_PERSISTENCE * jitter[index - 1]
                         + math.sqrt(1 - SHAKE_PERSISTENCE ** 2) * steps[index])
    side = max(width, height)
    jitter *= (shake * side, shake * side, SHAKE_ROTATION_DEG * shake / SHAKE_RANGE[1])
    matrices = []
    for (dx, dy), scale, angle, (sx, sy, sa) in zip(pan, zoom, rotation, jitter):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(angle + sa), float(scale))
        matrix[:, 2] += (dx + sx, dy + sy)
        matrices.append(matrix)
    return matrices


def eased(values, times):
    """Cosine-eased interpolation through values at evenly spaced keypoints."""
    position = times * (len(values) - 1)
    start = np.minimum(position.astype(int), len(values) - 2)
    fraction = (1 - np.cos(np.pi * (position - start))) / 2
    return values[start] + fraction * (values[start + 1] - values[start])


def motion_blur(frame, motion):
    speed = float(np.hypot(*motion))
    length = min(speed, MAX_BLUR_PX)
    if length < 1:
        return frame
    size = math.ceil(length) | 1
    kernel = np.zeros((size, size), np.float32)
    half = (size - 1) / 2
    dx, dy = motion / speed * length / 2
    cv2.line(kernel, (round(half - dx), round(half - dy)), (round(half + dx), round(half + dy)),
             1.0, 1)
    return cv2.filter2D(frame, -1, kernel / kernel.sum())


def add_leaves(img, mask, target, rng):
    """Leaves over the snake until they hide `target` of it, never past LEAF_COVER_RANGE."""
    region_name = "mask" if mask is not None and mask.any() else "centre"
    region = mask.astype(bool) if region_name == "mask" else centre_region(img.shape[:2])
    area = int(region.sum())
    ys, xs = np.nonzero(region)
    scene, covered = img.copy(), np.zeros(region.shape, np.uint8)
    centres, cover = [], 0.0
    for _ in range(LEAF_TRIES):
        if cover >= target:
            break
        pick = rng.integers(len(xs))
        outline = leaf_outline((xs[pick], ys[pick]),
                               rng.uniform(*LEAF_LENGTH_RANGE) * math.sqrt(area),
                               rng.uniform(*LEAF_ASPECT_RANGE), rng.uniform(0, math.pi))
        trial = covered.copy()
        cv2.fillPoly(trial, [outline], 1)
        trial_cover = float((trial.astype(bool) & region).sum() / area)
        if trial_cover > LEAF_COVER_RANGE[1]:
            continue
        covered, cover = trial, trial_cover
        centres.append([int(xs[pick]), int(ys[pick])])
        colour = rng.uniform(LEAF_GREEN_LOW, LEAF_GREEN_HIGH)
        cv2.fillPoly(scene, [outline], colour.tolist())
        cv2.line(scene, tuple(int(v) for v in outline[0]),
                 tuple(int(v) for v in outline[LEAF_POINTS - 1]), (colour * MIDRIB_DARKEN).tolist(),
                 1)
    return scene, {"occluder_region": region_name, "leaves": len(centres), "leaf_cover": cover,
                   "leaf_centres": centres}


def centre_region(shape):
    """The central ellipse leaves cover when a photo has no usable mask."""
    height, width = shape
    region = np.zeros(shape, np.uint8)
    axes = (max(1, int(width * CENTRE_REGION / 2)), max(1, int(height * CENTRE_REGION / 2)))
    cv2.ellipse(region, (width // 2, height // 2), axes, 0, 0, 360, 1, -1)
    return region.astype(bool)


def leaf_outline(centre, length, aspect, angle):
    """A leaf pointed at both ends: half-width (1 - x^2) / aspect for x in [-1, 1], then placed."""
    x = np.linspace(-1, 1, LEAF_POINTS)
    half = (1 - x ** 2) / aspect
    points = np.concatenate([np.stack([x, half], axis=1),
                             np.stack([x[::-1], -half[::-1]], axis=1)]) * length / 2
    turn = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return np.round(points @ turn.T + centre).astype(np.int32)


def vote_k_of_n(labels_seq, k=VOTE_K, n=VOTE_N, fallback=None):
    """Per frame, the label filling k of the last n frames, else `fallback` (not the last label)."""
    labels, shown = list(labels_seq), []
    for index in range(len(labels)):
        label, count = Counter(labels[max(0, index - n + 1):index + 1]).most_common(1)[0]
        shown.append(label if count >= k else fallback)
    return shown


def video_metrics(verdicts_seq, true_label, times=None, end=None):
    """Flips, time to the first stable label, and whether the last stable label is right."""
    labels = list(verdicts_seq)
    times = np.arange(len(labels)) / CLIP_FPS if times is None else np.asarray(times, dtype=float)
    end = len(labels) / CLIP_FPS if end is None else end
    end = max(end, float(times[-1])) if len(labels) else end
    runs = []
    for label, shown_at in zip(labels, times):
        if label is not None and (not runs or runs[-1][0] != label):
            runs.append((label, float(shown_at)))
    stops = [start for _, start in runs[1:]] + [end]
    stable = [(label, start) for (label, start), stop in zip(runs, stops)
              if stop - start >= STABLE_SECONDS - 1e-9]
    final = stable[-1][0] if stable else None
    return {"flips": max(0, len(runs) - 1), "time_to_stable": stable[0][1] if stable else None,
            "time_to_confirmed": stable[0][1] + STABLE_SECONDS if stable else None,
            "final_stable": final,
            "clip_correct": bool(final is not None and final == true_label),  # None is wrong
            "frames_correct": float(np.mean([label is not None and label == true_label
                                             for label in labels])) if labels else None}


def schedule(n_frames, fps, latency):
    """(processed frames, when each result shows) for a model busy `latency` s per frame."""
    # frame i arrives at i / fps; the free model takes the newest arrived frame
    seconds = np.asarray(latency, dtype=float)
    seconds = (np.full(n_frames, float(seconds)) if seconds.ndim == 0
               else np.resize(seconds, n_frames))
    processed, shown, free, last = [], [], 0.0, -1
    while True:
        newest = min(n_frames - 1, math.floor(free * fps + 1e-9))
        start = free
        if newest <= last:
            newest = last + 1
            if newest >= n_frames:
                break
            start = newest / fps
        free = start + float(seconds[newest])
        processed.append(newest)
        shown.append(free)
        last = newest
    return np.array(processed, dtype=int), np.array(shown)


def round_robin(frame, n, seed=SEED):
    """n photos taking the classes in turn, classes and photos in a seeded random order."""
    assert_no_test_paths(frame)
    rng = np.random.default_rng(seed)
    shuffled = frame.iloc[rng.permutation(len(frame))]
    species = shuffled["species"].unique()
    class_rank = dict(zip(species, rng.permutation(len(species))))
    shuffled = shuffled.assign(_turn=shuffled.groupby("species").cumcount(),
                               _rank=shuffled["species"].map(class_rank))
    picked = shuffled.sort_values(["_turn", "_rank"], kind="stable").head(n)
    return picked.drop(columns=["_turn", "_rank"]).reset_index(drop=True)
