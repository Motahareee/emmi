"""
First structured pruning baseline: prune CLIP ViT's MLP intermediate
width (see emma/model_compression/pruning.py for why that's the safe
starting point), sweep a range of prune ratios to get a real
compression-vs-accuracy trade-off curve rather than one config.

Reuses eval_ptq.py's model builder and metric helpers so results are
directly comparable to the PTQ/QAT numbers already in the report.

Usage: python3 eval_pruning.py [--n-samples 16] [--ratios 0.1,0.2,0.3,0.5]
"""

import argparse

import torch

from emma.data.coco import _stream_samples
from emma.model_compression import prune_clip_vit_mlps, model_size_mb
from eval_ptq import build_clip, _cosine_sim, _latency_ms


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.5",
                   help="comma-separated prune ratios to sweep")
    return p.parse_args()


def main():
    args = _parse_args()
    torch.set_num_threads(4)
    ratios = [float(r) for r in args.ratios.split(",")]

    print(f"Loading {args.n_samples} real COCO images...")
    raw = _stream_samples(args.n_samples, offset=0)
    images = [s["image"] for s in raw]

    image_encoder, _, proc, _ = build_clip()
    image_encoder.eval()
    pixel_values = proc(images=images, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        orig_out = image_encoder(pixel_values)
    orig_size = model_size_mb(image_encoder)
    orig_latency = _latency_ms(lambda: image_encoder(pixel_values))
    print(f"\nfp32 baseline: {orig_size:.1f} MB, "
          f"latency mean/min {orig_latency['mean']:.1f}/{orig_latency['min']:.1f} ms")

    print(f"\n{'ratio':>6} {'size (MB)':>10} {'compression':>12} "
          f"{'cos_sim':>8} {'latency mean/min (ms)':>24}")
    for ratio in ratios:
        pruned = prune_clip_vit_mlps(image_encoder, ratio)
        pruned.eval()
        with torch.no_grad():
            pruned_out = pruned(pixel_values)
        cos = _cosine_sim(orig_out, pruned_out)
        size = model_size_mb(pruned)
        lat = _latency_ms(lambda: pruned(pixel_values))
        print(f"{ratio:>6.1f} {size:>10.1f} {orig_size/size:>11.2f}x "
              f"{cos:>8.4f} {lat['mean']:>10.1f}/{lat['min']:<10.1f}")


if __name__ == "__main__":
    main()
