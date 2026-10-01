"""
Real Apple Neural Engine latency benchmark for CLIP vs MobileCLIP, run
locally on Apple Silicon (not the sandbox/cluster) -- the whole point of
this report's "motive" section is that MobileCLIP's published advantage
(Vasu et al., Table 7: 1.5ms vs CLIP's 5.9ms) is measured on Apple's
Neural Engine via Core ML, and every latency number elsewhere in this
investigation was measured on generic x86 CPU instead, where the
ranking inverts. This is the one piece of evidence that directly tests
the claim on real Apple hardware (a Mac's Neural Engine, not an
iPhone's, but the same ANE execution path via Core ML).

Converts each image encoder to Core ML (via torch.jit.trace, the
standard coremltools PyTorch path) and benchmarks with
compute_units=ALL -- Core ML's own scheduler picks ANE/GPU/CPU per op,
exactly how a real on-device deployment would run, same methodology the
paper itself uses.

Requires: pip install coremltools (torch/transformers/open_clip_torch
already needed for the rest of this repo).

Usage: python3 benchmark_coreml_mac.py [--n-repeats 30]
"""

import argparse
import time

import coremltools as ct
import torch

from eval_ptq import build_clip, build_mobileclip, model_size_mb


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-repeats", type=int, default=30)
    p.add_argument("--n-warmup", type=int, default=5)
    return p.parse_args()


def _latency_ms(fn, n_repeats, n_warmup):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    mean = sum(times) / len(times)
    return {"mean": mean, "min": min(times), "max": max(times)}


def convert_and_benchmark(name: str, image_encoder, pixel_values, n_repeats, n_warmup,
                          size_module=None) -> dict:
    """
    size_module: pass a different module to compute fp32 size from if
    image_encoder's own size is inflated by an unused shared backbone --
    MobileCLIP's wrapper holds the full shared text+image model
    (mc_img.model.visual is the actual visual-only size, ~43.7MB vs the
    ~286MB the whole wrapper reports). Doesn't affect the latency
    benchmark, which already only exercises the real image forward path.
    """
    print(f"\n--- {name}: tracing + converting to Core ML ---")
    image_encoder.eval()
    traced = torch.jit.trace(image_encoder, pixel_values)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=pixel_values.shape)],
        compute_units=ct.ComputeUnit.ALL,  # Core ML picks ANE/GPU/CPU per op -- same as real deployment
        convert_to="mlprogram",
    )

    np_input = {"pixel_values": pixel_values.numpy()}
    lat = _latency_ms(lambda: mlmodel.predict(np_input), n_repeats, n_warmup)
    size_fp32 = model_size_mb(size_module if size_module is not None else image_encoder)

    print(f"  fp32 PyTorch size: {size_fp32:.1f} MB")
    print(f"  Core ML (compute_units=ALL) latency mean/min: {lat['mean']:.2f}/{lat['min']:.2f} ms "
          f"(batch={pixel_values.shape[0]})")
    return {"size_mb": size_fp32, "latency": lat}


def main():
    args = _parse_args()
    torch.manual_seed(0)

    clip_img, _, clip_proc, _ = build_clip()
    clip_pv = clip_proc(images=[__import__("PIL.Image", fromlist=["Image"]).new("RGB", (224, 224))],
                        return_tensors="pt")["pixel_values"]
    clip_result = convert_and_benchmark("CLIP ViT-B/32", clip_img, clip_pv,
                                        args.n_repeats, args.n_warmup)

    mc_img, _, mc_proc, _ = build_mobileclip()
    mc_pv = mc_proc(images=[__import__("PIL.Image", fromlist=["Image"]).new("RGB", (256, 256))],
                    return_tensors="pt")["pixel_values"]
    mc_result = convert_and_benchmark("MobileCLIP-S0", mc_img, mc_pv,
                                      args.n_repeats, args.n_warmup,
                                      size_module=mc_img.model.visual)

    print("\n=== Summary: Core ML (real Apple Neural Engine path) on this Mac ===")
    print(f"  CLIP ViT-B/32:   {clip_result['size_mb']:.1f} MB, "
          f"latency min={clip_result['latency']['min']:.2f} ms")
    print(f"  MobileCLIP-S0:   {mc_result['size_mb']:.1f} MB, "
          f"latency min={mc_result['latency']['min']:.2f} ms")
    speedup = clip_result["latency"]["min"] / mc_result["latency"]["min"]
    print(f"  MobileCLIP speedup vs CLIP: {speedup:.2f}x "
          f"{'(faster, matches paper direction)' if speedup > 1 else '(slower, matches our CPU finding instead)'}")


if __name__ == "__main__":
    main()
