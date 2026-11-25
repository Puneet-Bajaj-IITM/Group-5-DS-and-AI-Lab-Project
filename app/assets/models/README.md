# Model files (not in git)

The M5 app runs four ONNX models. `tool/m5/build_assets.py` copies them here from the frozen M5 outputs (checking
each source against `results/m5/test/FREEZE.md`) and writes `SHA256SUMS`:

```
classifier.onnx        20,826,597 bytes   M5_KD (MobileViT-S, distilled), fp32
detector.onnx           9,680,263 bytes   M4 YOLO26n (crop only)
segmenter.onnx         26,498,348 bytes   M4 U-Net (crop only)
snake_check_fp16.onnx   8,064,356 bytes   distil_oi snake check (EfficientNet-B0), fp16 weights
```

Without them the app still builds and runs, but every result is a clearly marked demo. `pipeline.json` and
`tables.json` live in `../model/` and are committed.
