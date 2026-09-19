"""
Second PTQ baseline: static (calibration-based) quantization of CLIP and
MobileCLIP, compared against dynamic quantization (eval_ptq.py). See
emma/model_compression/static_ptq.py for what "static" means here and
why it's implemented as fake-quant rather than PyTorch's real eager-mode
static-quant pipeline.

Reuses eval_ptq.py's model builders and metric helpers so the two
scripts are directly comparable (same COCO batch, same latency
methodology, same cosine-similarity check).

Usage: python3 eval_static_ptq.py [--n-samples 16] [--n-calibration 8]
"""

import argparse

import torch

from emma.data.coco import _stream_samples
from emma.model_compression import calibrate_and_quantize, model_size_mb
from eval_ptq import build_clip, build_mobileclip, _cosine_sim, _latency_ms


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=16,
                   help="COCO images/captions for the eval batch")
    p.add_argument("--n-calibration", type=int, default=8,
                   help="subset of the batch used for calibration, kept "
                        "smaller than n-samples so calibration and eval "
                        "aren't drawing conclusions from identical data")
    return p.parse_args()


@torch.no_grad()
def benchmark_static(name: str, image_encoder, text_encoder, proc, tok,
                     images, captions, n_calibration: int,
                     shared_backbone: bool = False) -> dict:
    print(f"\n=== {name} (static PTQ) ===")
    pixel_values = proc(images=images, return_tensors="pt")["pixel_values"]
    text_inputs  = tok(captions, max_length=32, padding="max_length",
                       truncation=True, return_tensors="pt")
    calib_pixel_values = pixel_values[:n_calibration]
    calib_text_inputs  = {k: v[:n_calibration] for k, v in text_inputs.items()}

    if shared_backbone:
        def run_calibration(enc):
            enc(calib_pixel_values)
            enc.model.encode_text(calib_text_inputs["input_ids"], normalize=False)
        img_q = calibrate_and_quantize(image_encoder, run_calibration)
        txt_q = type(text_encoder)(img_q.model, freeze_base=False)
    else:
        img_q = calibrate_and_quantize(
            image_encoder, lambda enc: enc(calib_pixel_values))
        txt_q = calibrate_and_quantize(
            text_encoder, lambda enc: enc(**calib_text_inputs))

    v_orig = image_encoder(pixel_values)
    v_quant = img_q(pixel_values)
    t_orig = text_encoder(**text_inputs)
    t_quant = txt_q(**text_inputs)

    result = {
        "image_size_orig_mb":  model_size_mb(image_encoder),
        "image_size_quant_mb": model_size_mb(img_q),
        "text_size_orig_mb":   0.0 if shared_backbone else model_size_mb(text_encoder),
        "text_size_quant_mb":  0.0 if shared_backbone else model_size_mb(txt_q),
        "image_cos_sim":       _cosine_sim(v_orig, v_quant),
        "text_cos_sim":        _cosine_sim(t_orig, t_quant),
        "image_latency_orig_ms":  _latency_ms(lambda: image_encoder(pixel_values)),
        "image_latency_quant_ms": _latency_ms(lambda: img_q(pixel_values)),
    }
    total_orig  = result["image_size_orig_mb"] + result["text_size_orig_mb"]
    total_quant = result["image_size_quant_mb"] + result["text_size_quant_mb"]
    result["total_size_orig_mb"]  = total_orig
    result["total_size_quant_mb"] = total_quant
    result["compression_ratio"]   = total_orig / total_quant if total_quant else float("nan")

    label = "shared backbone" if shared_backbone else "image encoder"
    lat_o, lat_q = result["image_latency_orig_ms"], result["image_latency_quant_ms"]
    print(f"  {label}: {result['image_size_orig_mb']:.1f} MB -> "
          f"{result['image_size_quant_mb']:.1f} MB (cos_sim={result['image_cos_sim']:.4f})")
    print(f"    latency (mean/min/std, n=30): "
          f"orig {lat_o['mean']:.1f}/{lat_o['min']:.1f}/{lat_o['std']:.1f} ms -> "
          f"quant {lat_q['mean']:.1f}/{lat_q['min']:.1f}/{lat_q['std']:.1f} ms")
    if not shared_backbone:
        print(f"  text encoder:  {result['text_size_orig_mb']:.1f} MB -> "
              f"{result['text_size_quant_mb']:.1f} MB (cos_sim={result['text_cos_sim']:.4f})")
    else:
        print(f"  text cos_sim (same quantized backbone): {result['text_cos_sim']:.4f}")
    print(f"  total: {total_orig:.1f} MB -> {total_quant:.1f} MB "
          f"({result['compression_ratio']:.2f}x compression)")
    return result


def main():
    args = _parse_args()
    torch.set_num_threads(4)

    print(f"Loading {args.n_samples} real COCO images/captions "
          f"({args.n_calibration} used for calibration)...")
    raw = _stream_samples(args.n_samples, offset=0)
    images   = [s["image"] for s in raw]
    captions = [s["captions"][0] for s in raw]

    models = [("CLIP", build_clip, False), ("MobileCLIP", build_mobileclip, True)]
    for name, builder, shared_backbone in models:
        image_encoder, text_encoder, proc, tok = builder()
        benchmark_static(name, image_encoder, text_encoder, proc, tok,
                         images, captions, args.n_calibration,
                         shared_backbone=shared_backbone)


if __name__ == "__main__":
    main()
