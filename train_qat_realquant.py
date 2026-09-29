"""
Real-kernel quantization of QAT-trained weights.

QAT (train_qat.py) always trains with FAKE quantization -- QATLinear's
forward pass runs the full fp32 matmul under the hood, so it never
itself produces a real (smaller, faster) model. The 74.15%/43.45%
zero-shot numbers reported earlier are accuracy-only; no real int8
memory or latency numbers exist for a QAT-trained model, because
train_qat.py never saved anything and no export step was ever run.

This script closes that gap in one pipeline: run QAT training (reusing
qat_finetune from train_qat.py), unwrap the trained weights back to
plain fp32 (qat_encoder_to_fp32), then feed those weights through the
project's two EXISTING real-kernel PTQ paths --

  1. Dynamic PTQ (quantize_encoder_ptq) -- real int8, no calibration.
  2. Static PTQ via ONNX Runtime (export_image_encoder_to_onnx +
     quantize_onnx_static, per-channel + compute-ops-only, the best
     config found in the original PTQ investigation) -- genuine int8
     kernels, image encoder only (text blocked by a transformers ONNX
     export bug, same scope limit as the rest of this project).

-- and measures real size, real latency, and real zero-shot accuracy,
directly comparable to the plain-PTQ-without-QAT numbers already in the
report. This also tests a real hypothesis: does starting PTQ from
QAT-adapted weights beat starting from the original fp32 weights?

Usage: python3 train_qat_realquant.py [--n-train 4000] [--epochs 15] [--models clip,mobileclip]
"""

import argparse
import os

import torch
import torchvision

from emma.data.coco import _stream_samples
from emma.model_compression import (
    quantize_encoder_ptq, model_size_mb,
    export_image_encoder_to_onnx, ImageCalibrationReader, quantize_onnx_static,
    build_ort_session,
)
from emma.model_compression.qat import qat_encoder_to_fp32
from eval_ptq import build_clip, build_mobileclip, _cosine_sim, _latency_ms
from eval_onnx_ptq import MOBILECLIP_SENSITIVE_NODES
from eval_zeroshot_compare import zero_shot_accuracy
from train_qat import qat_finetune

ONNX_DIR = ".onnx_cache"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-zeroshot", type=int, default=2000)
    p.add_argument("--n-calibration", type=int, default=32,
                   help="calibration images for the ONNX static-quant path")
    p.add_argument("--models", type=str, default="clip,mobileclip",
                   help="comma-separated subset of clip,mobileclip to run")
    return p.parse_args()


def realquant_report(name: str, fp32_image, proc, calib_images,
                     zeroshot_images, zeroshot_labels, text_encoder, tok,
                     input_size: int, n_calibration: int) -> dict:
    print(f"\n--- Real-kernel quantization of {name}'s QAT-trained weights ---")
    result = {}

    size_fp32 = model_size_mb(fp32_image)

    def embed_fp32(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return fp32_image(pv)

    lat_fp32 = _latency_ms(lambda: embed_fp32(calib_images[:16]))
    acc_fp32 = zero_shot_accuracy(embed_fp32, text_encoder, tok, zeroshot_images, zeroshot_labels)
    print(f"  QAT fp32 (unwrapped, no further quant): {size_fp32:.1f} MB, "
          f"latency mean/min={lat_fp32['mean']:.1f}/{lat_fp32['min']:.1f} ms, "
          f"zero-shot acc={acc_fp32:.4f}")
    result["fp32"] = {"size_mb": size_fp32, "latency": lat_fp32, "acc": acc_fp32}

    # --- 1. Dynamic PTQ on QAT-trained weights ---
    img_dynq = quantize_encoder_ptq(fp32_image)
    size_dynq = model_size_mb(img_dynq)

    def embed_dynq(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return img_dynq(pv)

    lat_dynq = _latency_ms(lambda: embed_dynq(calib_images[:16]))
    cos_dynq = _cosine_sim(embed_fp32(calib_images), embed_dynq(calib_images))
    acc_dynq = zero_shot_accuracy(embed_dynq, text_encoder, tok, zeroshot_images, zeroshot_labels)
    print(f"  + dynamic PTQ (real int8): {size_dynq:.1f} MB ({size_fp32/size_dynq:.2f}x), "
          f"latency mean/min={lat_dynq['mean']:.1f}/{lat_dynq['min']:.1f} ms "
          f"({lat_fp32['mean']/lat_dynq['mean']:.2f}x), "
          f"cos_sim={cos_dynq:.4f}, zero-shot acc={acc_dynq:.4f}")
    result["dynamic_ptq"] = {"size_mb": size_dynq, "latency": lat_dynq,
                             "cos_sim": cos_dynq, "acc": acc_dynq}

    # --- 2. Real static PTQ via ONNX Runtime on QAT-trained weights ---
    os.makedirs(ONNX_DIR, exist_ok=True)
    fp32_onnx = os.path.join(ONNX_DIR, f"{name}_QAT_fp32.onnx")
    int8_onnx = os.path.join(ONNX_DIR, f"{name}_QAT_int8.onnx")
    export_image_encoder_to_onnx(fp32_image, input_size, fp32_onnx)
    calib_pv = proc(images=calib_images[:n_calibration], return_tensors="pt")["pixel_values"]
    exclude = MOBILECLIP_SENSITIVE_NODES if name == "MobileCLIP" else None
    quantize_onnx_static(fp32_onnx, ImageCalibrationReader(calib_pv), int8_onnx,
                         per_channel=True, op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                         nodes_to_exclude=exclude)
    sess = build_ort_session(int8_onnx)
    input_name = sess.get_inputs()[0].name

    def embed_onnx(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return torch.from_numpy(sess.run(None, {input_name: pv.numpy()})[0])

    size_onnx_fp32 = os.path.getsize(fp32_onnx) / (1024 ** 2)
    size_onnx_int8 = os.path.getsize(int8_onnx) / (1024 ** 2)
    lat_onnx = _latency_ms(lambda: embed_onnx(calib_images[:16]))
    cos_onnx = _cosine_sim(embed_fp32(calib_images), embed_onnx(calib_images))
    acc_onnx = zero_shot_accuracy(embed_onnx, text_encoder, tok, zeroshot_images, zeroshot_labels)
    print(f"  + static PTQ via ONNX (real int8 kernels): "
          f"{size_onnx_fp32:.1f} -> {size_onnx_int8:.1f} MB ({size_onnx_fp32/size_onnx_int8:.2f}x), "
          f"latency mean/min={lat_onnx['mean']:.1f}/{lat_onnx['min']:.1f} ms, "
          f"cos_sim={cos_onnx:.4f}, zero-shot acc={acc_onnx:.4f}")
    result["onnx_static_ptq"] = {"size_mb": size_onnx_int8, "latency": lat_onnx,
                                 "cos_sim": cos_onnx, "acc": acc_onnx}

    return result


def run_model(name, build_fn, input_size, train_images, train_captions,
             eval_images, eval_captions, zeroshot_images, zeroshot_labels,
             epochs, lr, batch_size, n_calibration):
    teacher_image, teacher_text, proc, tok = build_fn()
    qat_result = qat_finetune(name, teacher_image, teacher_text, proc, tok,
                              train_images, train_captions, eval_images, eval_captions,
                              epochs, lr, batch_size, zeroshot_images, zeroshot_labels)
    # Real quantization (dynamic PTQ's int8 CPU kernels, and this
    # project's whole PTQ/ONNX pipeline) targets CPU deployment, same as
    # every other PTQ script here -- move off DEVICE (cuda during
    # training) before quantizing/measuring latency.
    fp32_image = qat_encoder_to_fp32(qat_result["student_image"]).to("cpu").eval()
    fp32_text = qat_result["student_text"].to("cpu").eval()

    # realquant_report's embed_* calls run each cos_sim check as ONE
    # unbatched forward pass -- fine for a small slice, but eval_images
    # here can be 1000+ (matching train_qat.py's held-out eval set),
    # which OOMs exactly like the unbatched-full-tensor-forward bug
    # _batched_image_forward was built to fix elsewhere in this project.
    # These calls only need enough images for a stable cos_sim estimate,
    # not the whole eval set.
    cos_sim_images = eval_images[:min(32, len(eval_images))]

    return realquant_report(name, fp32_image, proc, cos_sim_images,
                            zeroshot_images, zeroshot_labels, fp32_text, tok,
                            input_size, n_calibration)


def main():
    args = _parse_args()
    torch.manual_seed(0)
    models = [m.strip() for m in args.models.split(",")]

    n_total = args.n_train + args.n_eval
    print(f"Loading {n_total} real COCO images/captions...")
    raw = _stream_samples(n_total, offset=0)
    images = [s["image"] for s in raw]
    captions = [s["captions"][0] for s in raw]
    train_images, eval_images = images[:args.n_train], images[args.n_train:]
    train_captions, eval_captions = captions[:args.n_train], captions[args.n_train:]

    print(f"Loading CIFAR-10 test set ({args.n_zeroshot} images)...")
    ds = torchvision.datasets.CIFAR10(root=".cifar10_cache", train=False, download=True)
    zeroshot_images = [ds[i][0] for i in range(args.n_zeroshot)]
    zeroshot_labels = [ds[i][1] for i in range(args.n_zeroshot)]

    all_results = {}
    if "clip" in models:
        all_results["CLIP"] = run_model("CLIP", build_clip, 224,
                                        train_images, train_captions, eval_images, eval_captions,
                                        zeroshot_images, zeroshot_labels,
                                        args.epochs, args.lr, args.batch_size, args.n_calibration)
    if "mobileclip" in models:
        all_results["MobileCLIP"] = run_model("MobileCLIP", build_mobileclip, 256,
                                              train_images, train_captions, eval_images, eval_captions,
                                              zeroshot_images, zeroshot_labels,
                                              args.epochs, args.lr, args.batch_size, args.n_calibration)

    print("\n=== Summary: real-kernel quantization of QAT-trained weights ===")
    for name, r in all_results.items():
        print(f"\n{name}:")
        for stage, m in r.items():
            acc = m.get("acc")
            size = m.get("size_mb")
            lat = m.get("latency", {}).get("min")
            print(f"  {stage:18s} size={size:7.1f} MB  latency(min)={lat:6.1f} ms  acc={acc:.4f}")


if __name__ == "__main__":
    main()
