"""
Per-node quantization sensitivity sweep for MobileCLIP's image encoder --
finds which specific layer(s) are responsible for the catastrophic
cos_sim collapse (0.045-0.11 across every global config tried so far;
see the report). Reuses the already-exported .onnx_cache/MobileCLIP_fp32.onnx
from eval_onnx_ptq.py.

Usage: python3 eval_sensitivity.py [--n-samples 16] [--n-calibration 8] [--top-n 15]
"""

import argparse
import os

import torch

from emma.data.coco import _stream_samples
from emma.model_compression import (
    export_image_encoder_to_onnx, ImageCalibrationReader, sweep_node_sensitivity,
    build_ort_session,
)
from eval_ptq import build_mobileclip

ONNX_DIR = ".onnx_cache"
SCRATCH_DIR = ".onnx_cache/sweep"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-calibration", type=int, default=8)
    p.add_argument("--top-n", type=int, default=15,
                   help="how many worst-offender nodes to print in detail")
    return p.parse_args()


def main():
    args = _parse_args()
    torch.set_num_threads(4)

    print(f"Loading {args.n_samples} real COCO images ({args.n_calibration} for calibration)...")
    raw = _stream_samples(args.n_samples, offset=0)
    images = [s["image"] for s in raw]

    image_encoder, _, proc, _ = build_mobileclip()
    pixel_values = proc(images=images, return_tensors="pt")["pixel_values"]
    calib_pixel_values = pixel_values[:args.n_calibration]

    fp32_path = os.path.join(ONNX_DIR, "MobileCLIP_fp32.onnx")
    if not os.path.exists(fp32_path):
        print("Exporting MobileCLIP image encoder to ONNX...")
        export_image_encoder_to_onnx(image_encoder, 256, fp32_path)

    fp32_sess = build_ort_session(fp32_path)
    input_name = fp32_sess.get_inputs()[0].name
    fp32_output = fp32_sess.run(None, {input_name: pixel_values.numpy()})[0]

    def make_calibration_reader():
        return ImageCalibrationReader(calib_pixel_values)

    print("\nSweeping node-by-node sensitivity (this takes a while: "
          "one quantize_static + inference pass per candidate node)...")
    results = sweep_node_sensitivity(
        fp32_path, make_calibration_reader, input_name,
        pixel_values.numpy(), fp32_output, SCRATCH_DIR, per_channel=True,
    )

    print(f"\n=== {args.top_n} worst-offender nodes (quantizing just this one node "
          f"tanks cos_sim this much) ===")
    for r in results[:args.top_n]:
        print(f"  cos_sim={r['cos_sim']:.4f}  {r['op_type']:6s}  {r['name']}")

    print(f"\n=== {min(5, len(results))} least-sensitive nodes (safe to quantize alone) ===")
    for r in results[-5:]:
        print(f"  cos_sim={r['cos_sim']:.4f}  {r['op_type']:6s}  {r['name']}")


if __name__ == "__main__":
    main()
