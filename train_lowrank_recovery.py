"""
Low-rank approximation + recovery fine-tuning for CLIP's image encoder --
the fourth and final compression technique from this project's original
research question (after PTQ, QAT, structured pruning).

Unlike pruning (delete channels) or quantization (reduce precision),
this factors each MLP weight matrix into a product of two smaller
matrices via truncated SVD (emma/model_compression/lowrank.py) -- the
mathematically optimal rank-r approximation, before any fine-tuning.
Like pruning, this is lossy and benefits from recovery fine-tuning to
let the factored weights compensate for the approximation error.

Mirrors train_pruning_recovery.py's structure and reporting (same
--eval-every/--early-stop trajectory tracking from Finding 26, same
latency+size+accuracy summary table) for direct comparability against
the pruning results at the same ratio.

Usage: python3 train_lowrank_recovery.py [--ratio 0.3] [--n-train 128] [--epochs 20] [--eval-every 5] [--early-stop]
"""

import argparse
import copy

import torch
import torchvision
from torch.optim import AdamW

from emma.data.coco import _stream_samples
from emma.model_compression import lowrank_clip_vit_mlps, distillation_loss, model_size_mb
from eval_ptq import build_clip, _cosine_sim, _latency_ms
from eval_zeroshot_compare import zero_shot_accuracy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _batched_image_forward(model, pv, batch_size):
    outs = [model(pv[i:i + batch_size]) for i in range(0, pv.size(0), batch_size)]
    return torch.cat(outs, dim=0)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.3,
                   help="fraction of full rank removed -- 0.3 keeps rank=round(0.7*768)=538 "
                        "for CLIP ViT-B/32's 768<->3072 MLP (see lowrank.py breakeven math)")
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-zeroshot", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=0,
                   help="see train_pruning_recovery.py's identical flag (Finding 26)")
    p.add_argument("--early-stop", action="store_true",
                   help="see train_pruning_recovery.py's identical flag (Finding 26)")
    return p.parse_args()


def main():
    args = _parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    print(f"Using device: {DEVICE}")

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

    # --- No-recovery baseline ---
    factored_only = lowrank_clip_vit_mlps(teacher_image, args.ratio).to(DEVICE)
    factored_only.eval()
    factored_only_eval_out = _batched_image_forward(factored_only, eval_pv, args.batch_size)
    no_recovery_cos = _cosine_sim(teacher_eval_out, factored_only_eval_out)

    def factored_only_embed(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
        return factored_only(pv)

    no_recovery_acc = zero_shot_accuracy(factored_only_embed, text_encoder, tok,
                                         zeroshot_images, zeroshot_labels, device=DEVICE)
    print(f"\nNo recovery (ratio={args.ratio}): "
          f"held-out cos_sim={no_recovery_cos:.4f}  zero-shot acc={no_recovery_acc:.4f}")

    # --- Low-rank factorize + recovery fine-tune ---
    print(f"\n=== Low-rank factorization (ratio={args.ratio}) + recovery fine-tuning ===")
    student = lowrank_clip_vit_mlps(teacher_image, args.ratio).to(DEVICE)
    for p in student.parameters():
        p.requires_grad = True
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    BATCH_SIZE = args.batch_size
    n_train = train_pv.size(0)
    best_acc, best_epoch, best_state = -1.0, None, None
    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss, n_batches = 0.0, 0
        perm = torch.randperm(n_train, device=DEVICE)
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            idx = perm[start:end]
            optimizer.zero_grad()
            student_out = student(train_pv[idx])
            loss = distillation_loss(student_out, teacher_train_out[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:02d}/{args.epochs}  cosine loss={epoch_loss / n_batches:.4f}")

        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            student.eval()
            trk_eval_out = _batched_image_forward(student, eval_pv, args.batch_size)
            trk_cos = _cosine_sim(teacher_eval_out, trk_eval_out)

            def trk_embed(imgs):
                pv = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
                return student(pv)

            trk_acc = zero_shot_accuracy(trk_embed, text_encoder, tok,
                                         zeroshot_images, zeroshot_labels, device=DEVICE)
            is_best = args.early_stop and trk_acc > best_acc
            print(f"    [epoch {epoch:02d}] held-out cos_sim={trk_cos:.4f}  zero-shot acc={trk_acc:.4f}"
                  f"{'  (new best)' if is_best else ''}")
            if is_best:
                best_acc, best_epoch = trk_acc, epoch
                best_state = copy.deepcopy(student.state_dict())
            student.train()

    def _embed(imgs):
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
        return student(pv)

    if args.early_stop and best_state is not None:
        student.eval()
        final_epoch_acc = zero_shot_accuracy(_embed, text_encoder, tok,
                                             zeroshot_images, zeroshot_labels, device=DEVICE)
        print(f"\nFinal epoch ({args.epochs}) zero-shot acc={final_epoch_acc:.4f} -- "
              f"restoring best checkpoint instead (epoch {best_epoch}, acc={best_acc:.4f})")
        student.load_state_dict(best_state)

    student.eval()
    student_eval_out = _batched_image_forward(student, eval_pv, args.batch_size)
    recovered_cos = _cosine_sim(teacher_eval_out, student_eval_out)
    recovered_acc = zero_shot_accuracy(_embed, text_encoder, tok,
                                       zeroshot_images, zeroshot_labels, device=DEVICE)

    lat_batch = proc(images=eval_images[:16], return_tensors="pt")["pixel_values"].to("cpu")
    teacher_cpu = teacher_image.to("cpu").eval()
    factored_only_cpu = factored_only.to("cpu").eval()
    student_cpu = student.to("cpu").eval()
    with torch.no_grad():
        lat_fp32 = _latency_ms(lambda: teacher_cpu(lat_batch))
        lat_no_recovery = _latency_ms(lambda: factored_only_cpu(lat_batch))
        lat_recovered = _latency_ms(lambda: student_cpu(lat_batch))

    print("\n=== Summary ===")
    size_fp32 = model_size_mb(teacher_image)
    size_lowrank = model_size_mb(student)
    print(f"  size: {size_fp32:.1f} -> {size_lowrank:.1f} MB ({size_fp32/size_lowrank:.2f}x)")
    if args.early_stop and best_state is not None:
        print(f"  early-stopped at epoch {best_epoch}/{args.epochs} (best held-out zero-shot acc)")
    print(f"  {'':25s} {'held-out cos_sim':>18} {'zero-shot acc':>15} {'latency min (ms)':>18}")
    print(f"  {'no recovery':25s} {no_recovery_cos:>18.4f} {no_recovery_acc:>15.4f} "
          f"{lat_no_recovery['min']:>18.1f}")
    print(f"  {'low-rank + recovered':25s} {recovered_cos:>18.4f} {recovered_acc:>15.4f} "
          f"{lat_recovered['min']:>18.1f}")
    print(f"  {'fp32 baseline (for ref)':25s} {'1.0000':>18} {'0.9200':>15} {lat_fp32['min']:>18.1f}")
    print(f"  latency speedup (min, low-rank+recovered vs fp32): {lat_fp32['min']/lat_recovered['min']:.2f}x")


if __name__ == "__main__":
    main()
