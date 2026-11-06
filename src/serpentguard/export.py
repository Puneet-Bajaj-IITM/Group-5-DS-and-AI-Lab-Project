"""ONNX Runtime sessions on the CPU, with telemetry off as the app runs offline; the pipeline and notebook 20 use them."""
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort

PROVIDERS = ["CPUExecutionProvider"]


def open_session(path, threads=None):
    """A CPU session; `threads` intra-op threads (one inter-op thread, sequential) or onnxruntime's default."""
    return ort.InferenceSession(str(path), _session_options(threads), providers=PROVIDERS)


def _session_options(threads=None):
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if threads:
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    return options


def runtime_graph(path):
    """The graph onnxruntime's CPU provider runs: the file after its load-time optimisations."""
    with tempfile.TemporaryDirectory(prefix="ort.runtime.") as folder:
        optimized = Path(folder) / "runtime.onnx"
        options = _session_options(1)
        options.optimized_model_filepath = str(optimized)
        ort.InferenceSession(str(path), options, providers=PROVIDERS)
        return onnx.load(str(optimized))
