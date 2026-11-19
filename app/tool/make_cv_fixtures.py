"""cv2 outputs of each image step on the golden photos, for the Dart parity test.

Usage: python tool/make_cv_fixtures.py <model bundle> <golden photo folder> <out folder>
(opencv-python-headless==5.0.0.93, as pinned in requirements-m3.txt)

uint8 INTER_LINEAR is taken as cv2's INTER_LINEAR_EXACT: OpenCV's generic fixed-point
resize, which the Dart port reproduces bit for bit. ARM builds of cv2 route INTER_LINEAR
through the carotene NEON HAL and x86 builds through IPP, each rounding a little differently;
on the 5 golden photos the generic resize leaves every gate, top species and verdict unchanged
and moves probabilities by at most 0.0009.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
import reference_pipeline as rp

_resize = cv2.resize
def _generic_linear(src, dsize, interpolation=cv2.INTER_LINEAR, **kwargs):
    if interpolation == cv2.INTER_LINEAR and src.dtype == np.uint8:
        interpolation = cv2.INTER_LINEAR_EXACT
    return _resize(src, dsize, interpolation=interpolation, **kwargs)
cv2.resize = _generic_linear

photos, out = Path(sys.argv[2]), Path(sys.argv[3])
settings = json.loads((Path(sys.argv[1]) / 'models/pipeline.json').read_text())
index = {}
for p in sorted(photos.glob('*.jpg')):
    rgb = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
    h, w = rgb.shape[:2]
    stem = p.stem
    rgb.tofile(out / f'{stem}.rgb')
    lb = rp.letterbox(rgb, 320, settings['letterbox']['pad_rgb']); lb.tofile(out / f'{stem}.letterbox')
    det = (rp.detector_input(rgb, 320, 114)[0].transpose(1, 2, 0) * 255).round().astype(np.uint8); det.tofile(out / f'{stem}.detector')
    up = cv2.resize(rgb, (w * 3 // 2, h * 3 // 2), interpolation=cv2.INTER_LINEAR); up.tofile(out / f'{stem}.up')
    half = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA); half.tofile(out / f'{stem}.half')
    # a smooth synthetic probability map through unletterbox + cleanup + crop
    yy, xx = np.mgrid[0:320, 0:320].astype(np.float32)
    prob = (np.sin(xx / 23.0) * np.cos(yy / 17.0) * 0.5 + 0.5).astype(np.float32)
    prob.tofile(out / f'{stem}.prob')
    mask = (rp.unletterbox(prob, h, w, 320) > 0.5).astype(np.uint8); mask.tofile(out / f'{stem}.mask')
    clean = rp.clean_mask(mask, settings['mask_cleanup'])
    (np.zeros_like(mask) if clean is None else clean).tofile(out / f'{stem}.clean')
    window = rp.crop_window(clean, h, w, settings['crop']['padding'])
    dark = (rgb // 5).astype(np.uint8); dark.tofile(out / f'{stem}.dark')
    fixed = rp.fix_exposure(dark, settings['exposure_fix']); fixed.tofile(out / f'{stem}.fixed')
    lab = cv2.cvtColor(dark, cv2.COLOR_RGB2LAB); lab.tofile(out / f'{stem}.lab')
    cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.ascontiguousarray(lab[..., 0])).tofile(out / f'{stem}.clahe')
    index[stem] = {'w': w, 'h': h, 'window': window, 'clean_none': clean is None, 'mean': float(rgb.mean())}
(out / 'index.json').write_text(json.dumps(index))
print(json.dumps(index))
