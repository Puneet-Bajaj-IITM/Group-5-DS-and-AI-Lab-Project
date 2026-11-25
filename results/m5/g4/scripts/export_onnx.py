"""Export the kernel's model.pt to ONNX: images (N,3,S,S) -> probs (N,3: snake, lookalike, other), features (N,F).

Writes snake_check.onnx (fp32) and snake_check_fp16.onnx (fp16 weights, fp32 inputs and outputs, as the app's
classifier_fp16.onnx), then checks both against PyTorch on a sample of real photos.

Usage: python export_onnx.py --run <folder with model.pt> [--photos <photo> ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_inputs import model_input  # noqa: E402

OPSET = 17
OUTPUTS = ("probs", "features")


class Student(torch.nn.Module):
    """The kernel's model with a softmax on the three classes; the distillation projection is dropped."""

    def __init__(self, checkpoint):
        super().__init__()
        self.body = timm.create_model(checkpoint["backbone"], pretrained=False, num_classes=0)
        self.head = torch.nn.Linear(self.body.num_features, len(checkpoint["classes"]))
        state = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("project.")}
        self.load_state_dict(state)

    def forward(self, images):
        features = self.body(images)
        return torch.softmax(self.head(features), dim=1), features


def load(run):
    checkpoint = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=False)
    return Student(checkpoint).eval(), checkpoint


def export(model, size, path):
    sample = torch.zeros(1, 3, size, size)
    torch.onnx.export(model, (sample,), str(path), input_names=["images"], output_names=list(OUTPUTS),
                      dynamic_axes={"images": {0: "n"}, "probs": {0: "n"}, "features": {0: "n"}},
                      opset_version=OPSET, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    half = path.with_name(f"{path.stem}_fp16.onnx")
    onnx.save(convert_float_to_float16(onnx.load(str(path)), keep_io_types=True), str(half))
    return half


def main(args):
    run = Path(args.run)
    model, checkpoint = load(run)
    size = checkpoint["input_size"]
    fp32 = run / "snake_check.onnx"
    fp16 = export(model, size, fp32)
    report = {"input_size": size, "classes": list(checkpoint["classes"]),
              "bytes": {p.name: p.stat().st_size for p in (fp32, fp16)}, "parity": {}}
    batch = np.concatenate([model_input(p, size) for p in args.photos])
    with torch.no_grad():
        reference = model(torch.from_numpy(batch))[0].numpy()
    for path in (fp32, fp16):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = session.run(["probs"], {"images": batch})[0]
        report["parity"][path.name] = {"photos": len(batch), "max_abs_prob_diff": float(np.abs(got - reference).max()),
                                       "same_top_class": float((got.argmax(1) == reference.argmax(1)).mean())}
    (run / "export.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--photos", nargs="+", required=True)
    main(parser.parse_args())
