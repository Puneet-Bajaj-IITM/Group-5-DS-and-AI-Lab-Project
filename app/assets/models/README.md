# Model files (not in git)

Copy the three fp32 ONNX models from the model bundle (`serpentguard-app-model.zip`, folder `models/`) here
before building:

```
classifier.onnx   20,826,597 bytes   A3, MobileViT-S
detector.onnx      9,680,263 bytes   M4 YOLO26n
segmenter.onnx    26,498,348 bytes   M4 U-Net
```

Check them with the bundle's `models/SHA256SUMS`. Without them the app still builds and runs, but every result is
a clearly marked demo. `pipeline.json` and `tables.json` live in `../model/` and are committed.
