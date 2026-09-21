"""
Structured pruning + recovery fine-tuning for CLIP's image encoder.

eval_pruning.py showed naive pruning (prune once, no fine-tuning) is far
more destructive per unit of compression than any quantization variant --
e.g. 10% pruning already dropped cos_sim to 0.842, worse than PTQ's best
config at 2.57-3.97x compression. This is expected: pruning permanently
removes channels with no chance to compensate, unlike quantization which
only reduces precision. Real pruning pipelines always include a recovery
step -- fine-tune the survivors to compensate for what was removed.

Reuses QAT's distillation infrastructure (emma/model_compression/qat.py)
unchanged -- prune is a one-time structural edit (unlike QAT's per-call
fake-quant), so recovery fine-tuning here is just normal gradient descent
against a frozen fp32 teacher, no STE needed.

Learned from the QAT investigation: cos_sim on the fine-tuning domain
(COCO) does NOT reliably predict real accuracy on a different domain
(CIFAR-10 zero-shot classification dropped to 37% despite 0.824 cos_sim)
-- so this script reports BOTH, not just cos_sim, for the pruned+recovered
model, the no-recovery baseline, and the fp32 original.

Usage: python3 train_pruning_recovery.py [--ratio 0.3] [--n-train 128] [--epochs 20]
"""

import argparse

import torch
import torchvision
from torch.optim import AdamW

from emma.data.coco import _stream_samples
from emma.model_compression import prune_clip_vit_mlps, distillation_loss, model_size_mb
from eval_ptq import build_clip, _cosine_sim
from eval_zeroshot_compare import zero_shot_accuracy


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.3,
                   help="prune ratio -- 0.3 gave cos_sim=0.55 with no recovery")
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-zeroshot", type=int, default=200)
    return p.parse_args()


def main():
    args = _parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)

    n_total = args.n_train + args.n_eval
    print(f"Loading {n_total} real COCO images/captions "
          f"({args.n_train} train / {args.n_eval} held-out eval)...")
    raw = _stream_samples(n_total, offset=0)
    images = [s["image"] for s in raw]
    train_images, eval_images = images[:args.n_train], images[args.n_train:]

    print(f"Loading CIFAR-10 test set ({args.n_zeroshot} images) for real zero-shot accuracy...")
    ds = torchvision.datasets.CIFAR10(root=".cifar10_cache", train=False, download=True)
    zeroshot_images = [ds[i][0] for i in range(args.n_zeroshot)]
    zeroshot_labels = [ds[i][1] for i in range(args.n_zeroshot)]

    teacher_image, text_encoder, proc, tok = build_clip()
    teacher_image.eval()
    text_encoder.eval()
    for p in teacher_image.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    train_pv = proc(images=train_images, return_tensors="pt")["pixel_values"]
    eval_pv = proc(images=eval_images, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        teacher_train_out = teacher_image(train_pv)
        teacher_eval_out = teacher_image(eval_pv)

    # --- No-recovery baseline (matches eval_pruning.py) ---
    pruned_only = prune_clip_vit_mlps(teacher_image, args.ratio)
    pruned_only.eval()
    with torch.no_grad():
        pruned_only_eval_out = pruned_only(eval_pv)
    no_recovery_cos = _cosine_sim(teacher_eval_out, pruned_only_eval_out)

    def pruned_only_embed(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return pruned_only(pv)

    no_recovery_acc = zero_shot_accuracy(pruned_only_embed, text_encoder, tok,
                                         zeroshot_images, zeroshot_labels)
    print(f"\nNo recovery (ratio={args.ratio}): "
          f"held-out cos_sim={no_recovery_cos:.4f}  zero-shot acc={no_recovery_acc:.4f}")

    # --- Prune + recovery fine-tune ---
    print(f"\n=== Pruning (ratio={args.ratio}) + recovery fine-tuning ===")
    student = prune_clip_vit_mlps(teacher_image, args.ratio)
    for p in student.parameters():
        p.requires_grad = True
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    BATCH_SIZE = 8
    n_train = train_pv.size(0)
    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            optimizer.zero_grad()
            student_out = student(train_pv[start:end])
            loss = distillation_loss(student_out, teacher_train_out[start:end])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:02d}/{args.epochs}  distillation loss={epoch_loss / n_batches:.4f}")

    student.eval()
    with torch.no_grad():
        student_eval_out = student(eval_pv)
    recovered_cos = _cosine_sim(teacher_eval_out, student_eval_out)

    def student_embed(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return student(pv)

    recovered_acc = zero_shot_accuracy(student_embed, text_encoder, tok,
                                       zeroshot_images, zeroshot_labels)

    print("\n=== Summary ===")
    print(f"  size: {model_size_mb(teacher_image):.1f} -> {model_size_mb(student):.1f} MB")
    print(f"  {'':25s} {'held-out cos_sim':>18} {'zero-shot acc':>15}")
    print(f"  {'no recovery':25s} {no_recovery_cos:>18.4f} {no_recovery_acc:>15.4f}")
    print(f"  {'pruned + recovered':25s} {recovered_cos:>18.4f} {recovered_acc:>15.4f}")
    print(f"  {'fp32 baseline (for ref)':25s} {'1.0000':>18} {'0.9200':>15}  (from earlier zero-shot comparison)")


if __name__ == "__main__":
    main()
