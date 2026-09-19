"""
First PTQ baseline: dynamic int8 quantization of CLIP and MobileCLIP's
Linear layers (see emma/model_compression/ptq.py), evaluated three ways:

  1. Model size (state_dict MB) -- the actual compression signal.
  2. Embedding cosine similarity, original vs quantized, on real COCO
     images/captions -- fast intrinsic sanity check.
  3. Latency (CPU wall-clock forward pass) -- the project's headline
     metric throughout.
  4. Zero-shot CIFAR-10 classification accuracy, original vs quantized --
     a small local proxy for the field-standard evaluation (OpenCLIP's
     38-dataset zero-shot suite, see conversation history). CIFAR-10 was
     chosen because it's small enough to download/run in a CPU sandbox;
     the full 38-dataset suite is a cluster-scale follow-up, not
     something this script attempts.

Usage: python3 eval_ptq.py [--n-samples 32] [--skip-zeroshot]
"""

import argparse
import time

import torch

from emma.data.coco import _stream_samples
from emma.model_compression import quantize_encoder_ptq, model_size_mb

N_ZEROSHOT_SAMPLES = 200
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=32,
                   help="COCO images/captions for the intrinsic + latency checks")
    p.add_argument("--skip-zeroshot", action="store_true",
                   help="skip the CIFAR-10 zero-shot proxy (no torchvision download)")
    return p.parse_args()


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def _latency_ms(fn, n_repeats: int = 30, n_warmup: int = 5) -> dict:
    """
    Returns {mean, min, std} in ms over n_repeats *individually timed*
    calls (not one big loop divided by N -- that hides variance). min is
    reported alongside mean because it's the more robust "perf floor"
    signal in a shared/noisy sandbox: scheduler interruptions inflate
    individual calls but can't make one artificially fast, so min is
    harder to fake than mean when there's background jitter.
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    return {"mean": mean, "min": min(times), "max": max(times), "std": variance ** 0.5}


def build_clip():
    from emma.encoders.image_encoder import ImageEncoder
    from emma.encoders.text_encoder import TextEncoder
    from transformers import CLIPImageProcessor, CLIPTokenizer

    image_encoder = ImageEncoder().eval()
    text_encoder  = TextEncoder().eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tok  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return image_encoder, text_encoder, proc, tok


def build_mobileclip():
    from emma.encoders.mobileclip_encoder import (
        build_mobileclip_encoders, build_mobileclip_processors,
    )
    text_encoder, image_encoder = build_mobileclip_encoders(freeze_base=True)
    image_encoder, text_encoder = image_encoder.eval(), text_encoder.eval()
    proc, tok = build_mobileclip_processors()
    return image_encoder, text_encoder, proc, tok


@torch.no_grad()
def benchmark(name: str, image_encoder, text_encoder, proc, tok,
             images, captions, shared_backbone: bool = False) -> dict:
    """
    shared_backbone=True: image_encoder/text_encoder wrap the exact same
    underlying model object (MobileCLIP's build_mobileclip_encoders loads
    ONE open_clip model and hands out two thin wrappers around it -- see
    emma/encoders/mobileclip_encoder.py). Quantizing each wrapper
    independently would deep-copy and quantize the *entire* shared model
    twice, double-counting size and wasting compute. Quantize once via
    image_encoder (which recursively quantizes every Linear in the shared
    model, including the text tower), then just re-wrap that same
    quantized model for the text side -- matches how this would actually
    be deployed (one quantized model on-device, not two).
    """
    print(f"\n=== {name} ===")
    pixel_values = proc(images=images, return_tensors="pt")["pixel_values"]
    text_inputs  = tok(captions, max_length=32, padding="max_length",
                       truncation=True, return_tensors="pt")

    img_q = quantize_encoder_ptq(image_encoder)
    if shared_backbone:
        txt_q = type(text_encoder)(img_q.model, freeze_base=False)
    else:
        txt_q = quantize_encoder_ptq(text_encoder)

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
    result["compression_ratio"]   = total_orig / total_quant

    label = "shared backbone" if shared_backbone else "image encoder"
    lat_o, lat_q = result["image_latency_orig_ms"], result["image_latency_quant_ms"]
    print(f"  {label}: {result['image_size_orig_mb']:.1f} MB -> "
          f"{result['image_size_quant_mb']:.1f} MB "
          f"(cos_sim={result['image_cos_sim']:.4f})")
    print(f"    latency (mean/min/std, n=30): "
          f"orig {lat_o['mean']:.1f}/{lat_o['min']:.1f}/{lat_o['std']:.1f} ms -> "
          f"quant {lat_q['mean']:.1f}/{lat_q['min']:.1f}/{lat_q['std']:.1f} ms")
    if shared_backbone:
        print(f"  text cos_sim (same quantized backbone): {result['text_cos_sim']:.4f}")
    else:
        print(f"  text encoder:  {result['text_size_orig_mb']:.1f} MB -> "
              f"{result['text_size_quant_mb']:.1f} MB (cos_sim={result['text_cos_sim']:.4f})")
    print(f"  total: {total_orig:.1f} MB -> {total_quant:.1f} MB "
          f"({result['compression_ratio']:.2f}x compression)")

    return result, img_q, txt_q


@torch.no_grad()
def zero_shot_cifar10_accuracy(image_encoder, text_encoder, proc, tok,
                               n_samples: int = N_ZEROSHOT_SAMPLES) -> float:
    """
    Standard CLIP zero-shot protocol at CIFAR-10 scale: embed
    "a photo of a {class}" for all 10 classes, classify each image by
    nearest text embedding (cosine similarity), report top-1 accuracy.
    """
    import torchvision

    ds = torchvision.datasets.CIFAR10(root=".cifar10_cache", train=False, download=True)
    prompts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    text_inputs = tok(prompts, max_length=32, padding="max_length",
                     truncation=True, return_tensors="pt")
    text_emb = text_encoder(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    correct, total = 0, 0
    batch_images, batch_labels = [], []
    for i in range(min(n_samples, len(ds))):
        img, label = ds[i]
        batch_images.append(img)
        batch_labels.append(label)

    pixel_values = proc(images=batch_images, return_tensors="pt")["pixel_values"]
    img_emb = image_encoder(pixel_values)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    sims = img_emb @ text_emb.T                       # [N, 10]
    preds = sims.argmax(dim=-1)
    correct = (preds == torch.tensor(batch_labels)).sum().item()
    total = len(batch_labels)
    return correct / total


def main():
    args = _parse_args()
    # Pin thread count for reproducible timing -- an unpinned default can
    # vary run-to-run in a shared sandbox depending on what else is
    # scheduled, which was likely a real contributor to the latency
    # numbers flipping direction between runs earlier.
    torch.set_num_threads(4)

    print(f"Loading {args.n_samples} real COCO images/captions...")
    raw = _stream_samples(args.n_samples, offset=0)
    images   = [s["image"] for s in raw]
    captions = [s["captions"][0] for s in raw]

    models = [("CLIP", build_clip, False), ("MobileCLIP", build_mobileclip, True)]
    for name, builder, shared_backbone in models:
        image_encoder, text_encoder, proc, tok = builder()
        result, img_q, txt_q = benchmark(name, image_encoder, text_encoder,
                                         proc, tok, images, captions,
                                         shared_backbone=shared_backbone)

        if not args.skip_zeroshot:
            try:
                acc_orig  = zero_shot_cifar10_accuracy(image_encoder, text_encoder, proc, tok)
                acc_quant = zero_shot_cifar10_accuracy(img_q, txt_q, proc, tok)
                print(f"  zero-shot CIFAR-10 acc: {acc_orig:.4f} -> {acc_quant:.4f} "
                      f"(proxy for the full OpenCLIP 38-dataset suite)")
            except Exception as e:
                print(f"  [skipped zero-shot check: {e}]")


if __name__ == "__main__":
    main()
