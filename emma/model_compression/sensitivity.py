"""
Per-node quantization sensitivity sweep.

Quantizes ONE node at a time (everything else stays fp32), measures the
resulting output's cosine similarity against the true fp32 output.
Isolates which specific layer(s) are responsible for a catastrophic
accuracy collapse, rather than continuing to guess at global quantization
settings -- see the report: per-channel weights and restricting to
Conv/MatMul/Gemm each individually and combined were not enough to fix
MobileCLIP (cos_sim stuck around 0.05-0.11), so the problem is
concentrated in specific layer(s), not a general config choice.
"""

import os

import numpy as np
import onnx
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from .onnx_static import build_ort_session


def get_quantizable_nodes(onnx_path: str, op_types=("Conv", "MatMul", "Gemm")) -> list:
    """Returns [{"name": ..., "op_type": ...}, ...] in graph order."""
    model = onnx.load(onnx_path)
    return [{"name": n.name, "op_type": n.op_type}
            for n in model.graph.node if n.op_type in op_types and n.name]


def _cosine_sim_np(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return float((num / denom).mean())


def sweep_node_sensitivity(fp32_path: str, make_calibration_reader,
                           input_name: str, eval_input: np.ndarray,
                           fp32_output: np.ndarray, scratch_dir: str,
                           per_channel: bool = True) -> list:
    """
    make_calibration_reader: callable() -> fresh CalibrationDataReader.
    A new reader instance is required per quantize_static call, since
    readers are single-use iterators (get_next() exhausts them).

    Returns node sensitivity results sorted ascending by cos_sim (worst
    offenders -- the layers actually causing the collapse -- first).
    """
    os.makedirs(scratch_dir, exist_ok=True)
    nodes = get_quantizable_nodes(fp32_path)
    results = []

    for i, node in enumerate(nodes):
        out_path = os.path.join(scratch_dir, f"_sweep_{i}.onnx")
        try:
            quantize_static(
                fp32_path, out_path, make_calibration_reader(),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=per_channel, nodes_to_quantize=[node["name"]],
            )
            sess = build_ort_session(out_path)
            out = sess.run(None, {input_name: eval_input})[0]
            cos = _cosine_sim_np(fp32_output, out)
        except Exception as e:
            cos = float("nan")
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

        results.append({"name": node["name"], "op_type": node["op_type"], "cos_sim": cos})
        print(f"  [{i+1}/{len(nodes)}] {node['op_type']:6s} {node['name']:60s} cos_sim={cos:.4f}")

    return sorted(results, key=lambda r: (r["cos_sim"] if r["cos_sim"] == r["cos_sim"] else -1))
