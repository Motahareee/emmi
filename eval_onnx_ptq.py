"""
Third/real static PTQ baseline: genuine int8 kernels via ONNX Runtime,
contrasted against static_ptq.py's fp32 simulation. See
emma/model_compression/onnx_static.py for why ONNX export is the fix for
PyTorch eager mode's tracing limitations on these architectures.

Scope: image encoders only (CLIP + MobileCLIP) -- the text encoder hits
an unrelated transformers-internal export bug, see onnx_static.py.

Usage: python3 eval_onnx_ptq.py [--n-samples 16] [--n-calibration 8]
"""

import argparse
import os

import torch

from emma.data.coco import _stream_samples
from emma.model_compression import (
    export_image_encoder_to_onnx, ImageCalibrationReader, quantize_onnx_static,
    summarize_onnx_graph, build_ort_session,
)
from eval_ptq import build_clip, build_mobileclip, _cosine_sim, _latency_ms

ONNX_DIR = ".onnx_cache"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-calibration", type=int, default=8)
    p.add_argument("--per-channel", action="store_true",
                   help="per-channel weight quantization instead of per-tensor "
                        "(see report Finding 8 -- MobileCLIP's default per-tensor "
                        "run broke, cos_sim=0.045)")
    p.add_argument("--compute-ops-only", action="store_true",
                   help="restrict quantization to Conv/MatMul/Gemm, leaving "
                        "elementwise/normalization math in fp32 -- MobileCLIP's "
                        "normalization is unfused primitive ops that the "
                        "quantizer's default exclusion list doesn't recognize")
    p.add_argument("--exclude-sensitive-nodes", action="store_true",
                   help="exclude MobileCLIP's two catastrophically-sensitive stem.1 "
                        "nodes from quantization (see eval_sensitivity.py / report "
                        "Finding 8c-d -- recovers cos_sim 0.098 -> 0.754). No effect "
                        "on CLIP, which doesn't share this node.")
    return p.parse_args()


# Found via eval_sensitivity.py's per-node sweep (report Finding 8c): these two
# nodes, quantized alone, score cos_sim -0.021 and 0.014 -- both parallel
# branches of MobileCLIP's stem.1 structural-reparameterization block.
MOBILECLIP_SENSITIVE_NODES = [
    "/visual/trunk/stem/stem.1/conv_kxk.0/conv/Conv",
    "/visual/trunk/stem/stem.1/conv_scale/conv/Conv",
]


def benchmark_onnx(name: str, image_encoder, proc, images,
                   input_size: int, n_calibration: int, per_channel: bool = False,
                   compute_ops_only: bool = False, exclude_sensitive_nodes: bool = False) -> dict:
    tags = [t for t, on in [("per-channel", per_channel),
                            ("compute-ops-only", compute_ops_only),
                            ("exclude-sensitive", exclude_sensitive_nodes)] if on] or ["default"]
    print(f"\n=== {name} (real static PTQ via ONNX Runtime, {', '.join(tags)}) ===")
    os.makedirs(ONNX_DIR, exist_ok=True)
    suffix = ("_perchannel" if per_channel else "") + ("_computeonly" if compute_ops_only else "") \
             + ("_excl" if exclude_sensitive_nodes else "")
    fp32_path = os.path.join(ONNX_DIR, f"{name}_fp32.onnx")
    int8_path = os.path.join(ONNX_DIR, f"{name}_int8{suffix}.onnx")

    pixel_values = proc(images=images, return_tensors="pt")["pixel_values"]

    export_image_encoder_to_onnx(image_encoder, input_size, fp32_path)
    calib_reader = ImageCalibrationReader(pixel_values[:n_calibration])
    op_types = ["Conv", "MatMul", "Gemm"] if compute_ops_only else None
    exclude = MOBILECLIP_SENSITIVE_NODES if (exclude_sensitive_nodes and name == "MobileCLIP") else None
    quantize_onnx_static(fp32_path, calib_reader, int8_path,
                         per_channel=per_channel, op_types_to_quantize=op_types,
                         nodes_to_exclude=exclude)

    fp32_sess = build_ort_session(fp32_path)
    int8_sess = build_ort_session(int8_path)
    input_name = fp32_sess.get_inputs()[0].name

    def run_fp32():
        return fp32_sess.run(None, {input_name: pixel_values.numpy()})[0]

    def run_int8():
        return int8_sess.run(None, {input_name: pixel_values.numpy()})[0]

    out_fp32 = torch.from_numpy(run_fp32())
    out_int8 = torch.from_numpy(run_int8())
    cos_sim = _cosine_sim(out_fp32, out_int8)

    lat_fp32 = _latency_ms(run_fp32)
    lat_int8 = _latency_ms(run_int8)

    size_fp32 = os.path.getsize(fp32_path) / (1024 ** 2)
    size_int8 = os.path.getsize(int8_path) / (1024 ** 2)

    graph_fp32 = summarize_onnx_graph(fp32_path)
    graph_int8 = summarize_onnx_graph(int8_path)

    print(f"  size: {size_fp32:.1f} MB -> {size_int8:.1f} MB "
          f"({size_fp32/size_int8:.2f}x compression)")
    print(f"  cos_sim: {cos_sim:.4f}")
    print(f"  latency (mean/min/std, n=30): "
          f"fp32 {lat_fp32['mean']:.1f}/{lat_fp32['min']:.1f}/{lat_fp32['std']:.1f} ms -> "
          f"int8 {lat_int8['mean']:.1f}/{lat_int8['min']:.1f}/{lat_int8['std']:.1f} ms")
    print(f"  graph nodes: {graph_fp32['n_nodes']} -> {graph_int8['n_nodes']}")
    print(f"  fp32 op types: {graph_fp32['op_counts']}")
    print(f"  int8 op types: {graph_int8['op_counts']}")

    return {
        "name": name, "size_fp32_mb": size_fp32, "size_int8_mb": size_int8,
        "cos_sim": cos_sim, "latency_fp32": lat_fp32, "latency_int8": lat_int8,
        "graph_fp32": graph_fp32, "graph_int8": graph_int8,
    }


def main():
    args = _parse_args()
    torch.set_num_threads(4)

    print(f"Loading {args.n_samples} real COCO images ({args.n_calibration} for calibration)...")
    raw = _stream_samples(args.n_samples, offset=0)
    images = [s["image"] for s in raw]

    image_encoder, _, proc, _ = build_clip()
    benchmark_onnx("CLIP", image_encoder, proc, images, 224, args.n_calibration,
                   per_channel=args.per_channel, compute_ops_only=args.compute_ops_only,
                   exclude_sensitive_nodes=args.exclude_sensitive_nodes)

    image_encoder, _, proc, _ = build_mobileclip()
    benchmark_onnx("MobileCLIP", image_encoder, proc, images, 256, args.n_calibration,
                   per_channel=args.per_channel, compute_ops_only=args.compute_ops_only,
                   exclude_sensitive_nodes=args.exclude_sensitive_nodes)


if __name__ == "__main__":
    main()
