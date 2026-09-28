"""
Iterative structured pruning: prune a little, fine-tune, prune a little
more, repeat -- rather than train_pruning_recovery.py's one-shot "remove
the full target ratio, then fine-tune once."

Motivation (report Findings 17-18): one-shot pruning + recovery plateaued
around 60-64% real accuracy regardless of ratio (0.1-0.2), and more
epochs made it *worse* (overfitting), not better -- ruling out "just
train longer" as the fix. The pruning literature's standard answer to
"one-shot pruning damages too much at once" is iterative pruning: each
step removes less, so the model never has to recover from a single large
shock, and fine-tuning between steps lets the remaining weights adapt
before the next cut.

No changes needed to emma/model_compression/pruning.py -- prune_clip_vit_mlps
operates on whatever the model's current fc1/fc2 sizes are, so calling it
repeatedly on an already-pruned model composes naturally. To reach a
target final ratio in n equal-fraction steps, each step removes fraction
f where (1-f)^n = (1-target_ratio).

Usage: python3 train_iterative_pruning.py [--target-ratio 0.3] [--n-steps 3]
"""

import argparse

import torch
import torchvision
from torch.optim import AdamW

from emma.data.coco import _stream_samples
from emma.model_compression import prune_clip_vit_mlps, distillation_loss, model_size_mb
from eval_ptq import build_clip, _cosine_sim
from eval_zeroshot_compare import zero_shot_accuracy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _batched_image_forward(model, pv, batch_size):
    outs = [model(pv[i:i + batch_size]) for i in range(0, pv.size(0), batch_size)]
    return torch.cat(outs, dim=0)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-ratio", type=float, default=0.3,
                   help="final compression target -- 0.3 is train_pruning_recovery.py's "
                        "one-shot baseline (49.65% accuracy), for direct comparison")
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--epochs-per-step", type=int, default=5,
                   help="fine-tuning epochs after each intermediate prune step")
    p.add_argument("--final-epochs", type=int, default=15,
                   help="fine-tuning epochs after the last prune step -- matches "
                        "train_pruning_recovery.py's one-shot recovery length")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--n-zeroshot", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def _finetune(student, teacher_out_train, train_pv, epochs, lr, batch_size):
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    n_train = train_pv.size(0)
    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            optimizer.zero_grad()
            student_out = student(train_pv[start:end])
            loss = distillation_loss(student_out, teacher_out_train[start:end])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"    epoch {epoch:02d}/{epochs}  distillation loss={epoch_loss / n_batches:.4f}")
    student.eval()


def main():
    args = _parse_args()
    torch.manual_seed(0)
    print(f"Using device: {DEVICE}")

    per_step_ratio = 1 - (1 - args.target_ratio) ** (1 / args.n_steps)
    print(f"Target final ratio {args.target_ratio} over {args.n_steps} steps "
          f"-> {per_step_ratio:.4f} removed per step")

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
    teacher_image = teacher_image.to(DEVICE).eval()
    text_encoder = text_encoder.to(DEVICE).eval()
    for p in teacher_image.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    train_pv = proc(images=train_images, return_tensors="pt")["pixel_values"].to(DEVICE)
    eval_pv = proc(images=eval_images, return_tensors="pt")["pixel_values"].to(DEVICE)

    teacher_train_out = _batched_image_forward(teacher_image, train_pv, args.batch_size)
    teacher_eval_out = _batched_image_forward(teacher_image, eval_pv, args.batch_size)

    student = teacher_image
    for step in range(1, args.n_steps + 1):
        print(f"\n=== Iterative pruning step {step}/{args.n_steps} "
              f"(remove {per_step_ratio:.4f} of current channels) ===")
        student = prune_clip_vit_mlps(student, per_step_ratio)
        for p in student.parameters():
            p.requires_grad = True

        epochs = args.final_epochs if step == args.n_steps else args.epochs_per_step
        print(f"  fine-tuning for {epochs} epochs...")
        _finetune(student, teacher_train_out, train_pv, epochs, args.lr, args.batch_size)

        eval_out = _batched_image_forward(student, eval_pv, args.batch_size)
        cos = _cosine_sim(teacher_eval_out, eval_out)
        size = model_size_mb(student)
        print(f"  after step {step}: size={size:.1f} MB  held-out cos_sim={cos:.4f}")

    def student_embed(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
        return student(pv)

    final_acc = zero_shot_accuracy(student_embed, text_encoder, tok,
                                   zeroshot_images, zeroshot_labels, device=DEVICE)
    final_cos = _cosine_sim(teacher_eval_out, _batched_image_forward(student, eval_pv, args.batch_size))
    final_size = model_size_mb(student)
    orig_size = model_size_mb(teacher_image)

    print("\n=== Summary ===")
    print(f"  size: {orig_size:.1f} -> {final_size:.1f} MB ({orig_size/final_size:.2f}x compression)")
    print(f"  iterative pruning ({args.n_steps} steps): "
          f"cos_sim={final_cos:.4f}  zero-shot acc={final_acc:.4f}")
    print(f"  (compare: one-shot ratio={args.target_ratio} + recovery from "
          f"train_pruning_recovery.py)")


if __name__ == "__main__":
    main()
