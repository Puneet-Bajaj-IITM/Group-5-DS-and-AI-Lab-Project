"""The Python reference of the M5 app path, for the Dart port's golden files.

The frozen pipeline (results/m5/test/FREEZE.md) as the app runs it: exposure fix -> snake check (distil_oi fp16,
probs[0] >= cutoff; it replaces the gate's decision) -> detector + U-Net for the crop only -> classifier on the crop
and its mirror image, every head averaged, prior added (results/m5/test/g5/reference_frozen.py) -> decision rules
(reference_pipeline.verdict) with the thresholds in the app's pipeline.json. Every setting is read from the app's
assets, every model is the app's own file.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
WORK = APP.parents[1]
_spec = importlib.util.spec_from_file_location("reference_frozen", WORK / "results/m5/test/g5/reference_frozen.py")
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)
ref = frozen._ref


def numbers(thresholds):
    return {k: float(v) for k, v in thresholds.items()}


class AppPipeline(frozen.Pipeline):
    """reference_frozen.Pipeline with the app's settings and tables and the snake check."""

    def __init__(self, threads=1):
        models = APP / "assets/models"
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.log_severity_level = 3

        def session(name):
            return ort.InferenceSession(str(models / name), options, providers=["CPUExecutionProvider"])

        self.settings = json.loads((APP / "assets/model/pipeline.json").read_text())
        self.tables = ref.read_tables(APP / "assets/model/tables.json")
        check = self.settings["snake_check"]
        self.classifier, self.segmenter = session("classifier.onnx"), session("segmenter.onnx")
        self.detector, self.snake_check = session("detector.onnx"), session(check["file"])
        self.output_names = [o.name for o in self.classifier.get_outputs()]
        classifier = self.settings["classifier"]
        assert classifier["prior_tau"] == frozen.TAU
        assert np.array_equal(np.asarray(classifier["log_counts"]), frozen.LOG_COUNTS)
        self.thresholds = numbers(self.settings["decision"]["thresholds"])
        self.frozen_thresholds = numbers(self.settings["decision"]["frozen_thresholds"])

    def snake_check_input(self, rgb):
        s, check = self.settings, self.settings["snake_check"]
        return ref.normalise(ref.letterbox(rgb, check["input_size"], s["letterbox"]["pad_rgb"]), s["normalise"])

    def snake_probs(self, rgb):
        return self.snake_check.run([self.settings["snake_check"]["output"]],
                                    {"images": self.snake_check_input(rgb)})[0][0]

    def is_snake(self, probs):
        check = self.settings["snake_check"]
        return bool(probs[check["snake_index"]] >= check["cutoff"])

    def answer_rgb(self, rgb):
        """One exposure-fixed photo to the app's record, with the frozen thresholds' verdict beside it."""
        probs = self.snake_probs(rgb)
        snake = self.is_snake(probs)
        check = {"snake": snake, "snake_probability": float(probs[self.settings["snake_check"]["snake_index"]])}
        if not snake:
            return {"snake_check": check, "front_end": None, "crop": None, **ref.no_snake_verdict(),
                    "frozen_verdict": ref.no_snake_verdict()["verdict"]}
        gate, mask = self.gate(rgb)
        front_end = {"box_conf": gate["box_conf"], "mask_area": gate["mask_area"],
                     "m4_gate_would_say_snake": gate["snake"]}
        height, width = rgb.shape[:2]
        window = ref.crop_window(ref.clean_mask(mask, self.settings["mask_cleanup"]), height, width,
                                 self.settings["crop"]["padding"])
        crop = rgb if window is None else rgb[window[0]:window[1], window[2]:window[3]]
        outputs = self.classify(crop)
        return {"snake_check": check, "front_end": front_end, "crop": "whole photo" if window is None else "mask box",
                "crop_window": window, **ref.verdict(outputs, self.thresholds, self.tables),
                "frozen_verdict": ref.verdict(outputs, self.frozen_thresholds, self.tables)["verdict"],
                "outputs": outputs}
