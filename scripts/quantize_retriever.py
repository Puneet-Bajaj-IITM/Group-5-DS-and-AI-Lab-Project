"""Build an ONNX Runtime int4 MiniLM candidate from its full-precision ONNX export."""

import argparse
import logging
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization.quant_utils import QuantFormat


def quantize(source, output):
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source)
    opset = next(item.version for item in model.opset_import if item.domain in ("", "ai.onnx"))
    if opset < 21:
        model = onnx.version_converter.convert_version(model, 21)
    quantizer = MatMulNBitsQuantizer(
        model, bits=4, block_size=32, is_symmetric=True,
        quant_format=QuantFormat.QOperator, op_types_to_quantize=("MatMul", "Gather"),
    )
    quantizer.process()
    model = quantizer.model.model
    initializers = {tensor.name: tensor.data_type for tensor in model.graph.initializer}
    bits_by_operator = Counter()
    for node in model.graph.node:
        if node.op_type == "MatMulNBits" and any(a.name == "bits" and a.i == 4 for a in node.attribute):
            bits_by_operator[node.op_type] += 1
        if node.op_type == "GatherBlockQuantized" and initializers.get(node.input[0]) == onnx.TensorProto.INT4:
            bits_by_operator[node.op_type] += 1
    if not bits_by_operator["MatMulNBits"] or not bits_by_operator["GatherBlockQuantized"]:
        raise ValueError(f"Expected int4 projection and embedding weights, got {dict(bits_by_operator)}")
    serialized = model.SerializeToString()
    session = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    del session
    with output.open("xb") as file:
        file.write(serialized)
    return {"operators": dict(bits_by_operator), "source_bytes": source.stat().st_size,
            "output_bytes": len(serialized)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    print(quantize(args.source, args.output))


if __name__ == "__main__":
    main()
