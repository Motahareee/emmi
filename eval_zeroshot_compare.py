"""
Absolute accuracy comparison: fp32 CLIP vs. quantized CLIP vs. fp32
MobileCLIP, on a real task with real labels (zero-shot CIFAR-10
classification), not the self-referential cos_sim-vs-own-fp32-output
numbers everything else in this investigation has used.

This is what actually answers "does quantized CLIP perform as well as
MobileCLIP" -- cos_sim only ever told us how much a model's quantized
output drifted from its OWN original output, never whether either
model's embeddings are good for anything.

Scope note: CLIP's *text* encoder was never successfully quantized via
the real ONNX static-quant path (blocked by an unrelated transformers
export bug -- see emma/model_compression/onnx_static.py). So "quantized
CLIP" here means: quantized image encoder (best config found: per-channel
weights + Conv/MatMul/Gemm-only) + fp32 text encoder. This isolates
exactly the question this investigation has been asking -- does
image-encoder quantization hurt downstream accuracy -- without
conflating it with a different, unrelated quantization method for text.

CIFAR-10 (200 test images, 10 classes) is a small local proxy for the
field-standard OpenCLIP 38-dataset zero-shot suite -- same caveat as
eval_ptq.py's zero-shot check.

Usage: python3 eval_zeroshot_compare.py [--n-samples 200]
"""

import argparse
import os

import numpy as np
import torch
import torchvision

from emma.model_compression import (
    export_image_encoder_to_onnx, ImageCalibrationReader, quantize_onnx_static,
    build_ort_session,
)
from eval_ptq import build_clip, build_mobileclip, CIFAR10_CLASSES
from eval_onnx_ptq import MOBILECLIP_SENSITIVE_NODES
from emma.data.coco import _stream_samples

ONNX_DIR = ".onnx_cache"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-calibration", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def zero_shot_accuracy(image_embed_fn, text_encoder, tok, images, labels,
                       batch_size: int = 16) -> float:
    """
    image_embed_fn: callable(list[PIL.Image]) -> torch.Tensor [N, 512]
    (lets the image side be either a PyTorch encoder or an ONNX session
    without this function caring which).

    Images are embedded in chunks of batch_size rather than one giant
    forward call -- a 200-image single-batch forward through a Conv-heavy
    model (MobileCLIP) plus fake-quant's extra per-call tensor overhead
    (QATLinear) pushed memory high enough to get silently OOM-killed with
    n_zeroshot=200; chunking caps peak activation memory regardless of
    how many total images are evaluated.
    """
    prompts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    text_inputs = tok(prompts, max_length=32, padding="max_length",
                      truncation=True, return_tensors="pt")
    text_emb = text_encoder(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    img_embs = []
    for start in range(0, len(images), batch_size):
        chunk = image_embed_fn(images[start:start + batch_size])
        img_embs.append(chunk / chunk.norm(dim=-1, keepdim=True))
    img_emb = torch.cat(img_embs, dim=0)

    sims = img_emb @ text_emb.T
    preds = sims.argmax(dim=-1)
    correct = (preds == torch.tensor(labels)).sum().item()
    return correct / len(labels)


def main():
    args = _parse_args()
    torch.set_num_threads(4)

    print(f"Loading CIFAR-10 test set ({args.n_samples} images)...")
    ds = torchvision.datasets.CIFAR10(root=".cifar10_cache", train=False, download=True)
    images = [ds[i][0] for i in range(args.n_samples)]
    labels = [ds[i][1] for i in range(args.n_samples)]

    raw = _stream_samples(args.n_calibration, offset=0)
    calib_images = [s["image"] for s in raw]

    results = {}

    # --- fp32 CLIP ---
    print("\n=== fp32 CLIP ===")
    clip_img, clip_txt, clip_proc, clip_tok = build_clip()

    def clip_embed_fp32(imgs):
        pv = clip_proc(images=imgs, return_tensors="pt")["pixel_values"]
        return clip_img(pv)

    acc = zero_shot_accuracy(clip_embed_fp32, clip_txt, clip_tok, images, labels)
    results["fp32 CLIP"] = acc
    print(f"  zero-shot CIFAR-10 accuracy: {acc:.4f}")

    # --- quantized-image CLIP (fp32 text) ---
    print("\n=== quantized-image CLIP (ONNX, per-channel + compute-ops-only) + fp32 text ===")
    os.makedirs(ONNX_DIR, exist_ok=True)
    fp32_onnx = os.path.join(ONNX_DIR, "CLIP_fp32.onnx")
    int8_onnx = os.path.join(ONNX_DIR, "CLIP_int8_best.onnx")
    if not os.path.exists(fp32_onnx):
        export_image_encoder_to_onnx(clip_img, 224, fp32_onnx)
    calib_pv = clip_proc(images=calib_images, return_tensors="pt")["pixel_values"]
    quantize_onnx_static(fp32_onnx, ImageCalibrationReader(calib_pv), int8_onnx,
                         per_channel=True, op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    sess = build_ort_session(int8_onnx)
    input_name = sess.get_inputs()[0].name

    def clip_embed_quant(imgs):
        pv = clip_proc(images=imgs, return_tensors="pt")["pixel_values"]
        out = sess.run(None, {input_name: pv.numpy()})[0]
        return torch.from_numpy(out)

    acc = zero_shot_accuracy(clip_embed_quant, clip_txt, clip_tok, images, labels)
    results["quantized-image CLIP + fp32 text"] = acc
    print(f"  zero-shot CIFAR-10 accuracy: {acc:.4f}")

    # --- fp32 MobileCLIP ---
    print("\n=== fp32 MobileCLIP ===")
    mc_img, mc_txt, mc_proc, mc_tok = build_mobileclip()

    def mc_embed_fp32(imgs):
        pv = mc_proc(images=imgs, return_tensors="pt")["pixel_values"]
        return mc_img(pv)

    acc = zero_shot_accuracy(mc_embed_fp32, mc_txt, mc_tok, images, labels)
    results["fp32 MobileCLIP"] = acc
    print(f"  zero-shot CIFAR-10 accuracy: {acc:.4f}")

    print("\n=== Summary ===")
    for name, acc in results.items():
        print(f"  {name:45s} {acc:.4f}")


if __name__ == "__main__":
    main()
