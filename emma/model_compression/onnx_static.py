"""
Real static PTQ via ONNX Runtime -- the actual industry-standard path,
not the fp32 simulation in static_ptq.py.

Why this exists: PyTorch's eager-mode static-quant pipeline needs a
model's forward() written in a quantization-friendly style (explicit
QuantStub/DeQuantStub placement) that arbitrary HuggingFace/open_clip
code isn't, so it can't reliably find where to apply its (real, working)
int8 kernels. The fix used across the industry is to export the model to
a fixed computation graph first (here: ONNX), which flattens away all the
ambiguous Python control flow into an explicit op sequence -- then
quantize *that* graph using the target runtime's own tooling. ONNX
Runtime's quantizer inserts real int8 kernels (QLinearConv/QLinearMatMul
etc.), so unlike static_ptq.py's fake-quant, latency measured against the
resulting model reflects genuine int8 compute.

Scope: image encoders only. CLIP's text encoder hits an unrelated
transformers-internal tracing bug in its causal-mask code
(masking_utils.create_causal_mask indexes a traced tensor incorrectly)
that isn't a quantization problem -- see the report for that decision.
"""

import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType, quantize_static,
)


def export_image_encoder_to_onnx(encoder: torch.nn.Module, input_size: int,
                                 path: str) -> str:
    """
    Legacy TorchScript-based exporter (dynamo=False) -- the newer
    torch.export-based default exporter produced a malformed graph
    (duplicate output name) for these models; legacy tracing round-trips
    correctly (verified: max abs diff ~1e-5 vs the PyTorch output).
    """
    encoder = encoder.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        encoder, dummy, path,
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    return path


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds real calibration images (as preprocessed pixel_values) to
    ONNX Runtime's static quantizer, one batch element at a time."""

    def __init__(self, pixel_values: torch.Tensor):
        self._data = iter(pixel_values.numpy())

    def get_next(self):
        x = next(self._data, None)
        if x is None:
            return None
        return {"pixel_values": x[None, ...]}


def quantize_onnx_static(onnx_path: str, calibration_reader: CalibrationDataReader,
                         output_path: str, per_channel: bool = False,
                         op_types_to_quantize: list = None,
                         nodes_to_exclude: list = None) -> str:
    """
    per_channel controls WEIGHT quantization granularity only -- one
    scale/zero-point per output channel (filter) instead of one shared
    across the whole weight tensor. Doesn't touch activation
    quantization (still per-tensor, still calibration-based) or need any
    calibration data itself, since weights are static. Conv-heavy nets
    (MobileCLIP) are far more sensitive to this than Linear-heavy ones
    (CLIP) -- see the report for why. (Empirically insufficient alone
    for MobileCLIP -- see op_types_to_quantize below.)

    op_types_to_quantize restricts quantization to only the listed ONNX
    op types (e.g. ["Conv", "MatMul", "Gemm"]), leaving everything else
    -- crucially, elementwise/normalization math -- in fp32. ORT's
    default behavior already skips *named* ops like LayerNormalization/
    Softmax, but MobileCLIP's normalization is built from primitive ops
    (ReduceMean/Sub/Sqrt/Div) that the default exclusion list doesn't
    recognize, so those get quantized too unless restricted explicitly.
    """
    quantize_static(
        onnx_path, output_path, calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
    )
    return output_path


def summarize_onnx_graph(onnx_path: str) -> dict:
    """
    Op-type histogram + node count -- a text-summary stand-in for a full
    Netron-style diagram, cheap to compute and enough to see what a
    quantized graph actually contains (QuantizeLinear/DequantizeLinear/
    QLinearMatMul nodes appear here that don't exist in the fp32 graph).
    """
    model = onnx.load(onnx_path)
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    return {
        "n_nodes": len(model.graph.node),
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "file_size_mb": os.path.getsize(onnx_path) / (1024 ** 2),
    }


def build_ort_session(onnx_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
