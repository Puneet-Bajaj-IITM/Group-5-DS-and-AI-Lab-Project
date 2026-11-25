"""The fixed photo corruptions of the M5 robustness stress test, applied to the decoded photo before the pipeline.

Each corruption maps an RGB uint8 photo to an RGB uint8 photo of the same size (the front end's masks must match the
photo). Noise is seeded per photo from its path, so a rerun gives the same pixels. JPEG returns the encoded bytes,
since the pipeline decodes the file itself.
"""
import io
import zlib

import cv2
import numpy as np
from PIL import Image

SEED = 42
OCCLUDER_RGB = (128, 128, 128)
CONDITIONS = {  # name: (kind, severity)
    "clean": ("none", None),
    "blur_s1": ("blur", 0.005), "blur_s2": ("blur", 0.01), "blur_s3": ("blur", 0.02),
    "noise_s1": ("noise", 10), "noise_s2": ("noise", 25), "noise_s3": ("noise", 50),
    "jpeg_q30": ("jpeg", 30), "jpeg_q10": ("jpeg", 10),
    "dark_x0.6": ("brightness", 0.6), "dark_x0.3": ("brightness", 0.3),
    "bright_x1.5": ("brightness", 1.5), "bright_x2.0": ("brightness", 2.0),
    "rotate_15": ("rotate", 15), "rotate_30": ("rotate", 30),
    "occlude_10pct": ("occlude", 0.10), "occlude_25pct": ("occlude", 0.25),
    "lowres_x0.5": ("lowres", 0.5), "lowres_x0.25": ("lowres", 0.25),
}
SEVERITY_NOTE = {
    "blur": "Gaussian sigma as a fraction of the photo's short side (500 px photos: 1.7 / 3.3 / 6.7 px on a 334 px side)",
    "noise": "Gaussian noise, standard deviation in 0-255 grey levels, per-photo seed",
    "jpeg": "re-encoded as JPEG at this quality (PIL, 4:2:0)",
    "brightness": "every pixel multiplied, clipped to 0-255 (the pipeline's exposure fix still runs after)",
    "rotate": "rotated about the centre by this many degrees, same canvas, edges reflected",
    "occlude": "a grey (128) square centred on the photo covering this share of its area",
    "lowres": "downscaled by this factor (area) and back up to the original size (bilinear)",
}


def photo_rng(path):
    return np.random.default_rng([SEED, zlib.crc32(path.encode())])


def blur(rgb, fraction):
    sigma = fraction * min(rgb.shape[:2])
    return cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def noise(rgb, std, path):
    noisy = rgb.astype(np.float32) + photo_rng(path).normal(0.0, std, rgb.shape).astype(np.float32)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def brightness(rgb, factor):
    return np.clip(np.rint(rgb.astype(np.float32) * factor), 0, 255).astype(np.uint8)


def rotate(rgb, degrees):
    height, width = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), degrees, 1.0)
    return cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def occlude(rgb, share):
    height, width = rgb.shape[:2]
    side = int(round(np.sqrt(share * height * width)))
    side_y, side_x = min(side, height), min(side, width)
    top, left = (height - side_y) // 2, (width - side_x) // 2
    out = rgb.copy()
    out[top:top + side_y, left:left + side_x] = OCCLUDER_RGB
    return out


def lowres(rgb, factor):
    height, width = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def jpeg_bytes(rgb, quality):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def corrupted(rgb, condition, path):
    """The corrupted photo as an RGB array, or as JPEG bytes for the jpeg kind."""
    kind, severity = CONDITIONS[condition]
    if kind == "none":
        return rgb
    if kind == "jpeg":
        return jpeg_bytes(rgb, severity)
    if kind == "noise":
        return noise(rgb, severity, path)
    return {"blur": blur, "brightness": brightness, "rotate": rotate, "occlude": occlude, "lowres": lowres}[kind](
        rgb, severity)


def decode(file):
    """The photo exactly as every pipeline loader decodes it (PIL, RGB, no EXIF turn)."""
    with Image.open(file) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def write_corrupted(source, target, condition, path):
    """Decode source, corrupt, write to target (lossless PNG bytes, or the JPEG bytes) under the same file name."""
    out = corrupted(decode(source), condition, path)
    if isinstance(out, bytes):
        with open(target, "wb") as handle:
            handle.write(out)
        return
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    assert ok, f"could not encode {path}"
    with open(target, "wb") as handle:
        handle.write(encoded.tobytes())
