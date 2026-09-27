"""
QAT (Quantization-Aware Training) for CLIP and MobileCLIP's image+text
encoders -- fourth quantization variant after dynamic/simulated-static/
real-ONNX-static PTQ (see eval_ptq.py / eval_static_ptq.py / eval_onnx_ptq.py).

Fine-tunes with fake-quant nodes in the loop (emma/model_compression/qat.py)
via distillation against the frozen fp32 model -- no labels or original
training data needed, just real COCO images/captions as fine-tuning input.

Unlike the PTQ scripts, train/eval data here are disjoint (fine-tune on
one set of images, evaluate on a separate held-out set) -- static PTQ's
calibration/eval overlap was flagged as a real methodological gap in the
report; this closes it for QAT.

Usage: python3 train_qat.py [--n-train 64] [--n-eval 32] [--epochs 15]
"""

import argparse

import torch
from torch.optim import AdamW

from emma.data.coco import _stream_samples
from emma.model_compression import convert_to_qat, distillation_loss, model_size_mb
from eval_ptq import build_clip, build_mobileclip, _cosine_sim
from eval_zeroshot_compare import zero_shot_accuracy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _batched_image_forward(model, pv, batch_size):
    """
    A full-tensor forward call (no mini-batching) was fine at the CPU
    investigation's scale (n_train=128) but OOMs on GPU at real scale
    (n_train=4000+) -- MobileCLIP's Conv-heavy stem produces activation
    tensors that scale directly with batch size, and a single 4000-image
    batch tried to allocate 15+ GiB in one shot. Chunking caps peak
    memory the same way the training loop already does.
    """
    outs = [model(pv[i:i + batch_size]) for i in range(0, pv.size(0), batch_size)]
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _batched_text_forward(model, txt, batch_size):
    n = txt["input_ids"].size(0)
    outs = [model(input_ids=txt["input_ids"][i:i + batch_size],
                  attention_mask=txt["attention_mask"][i:i + batch_size])
           for i in range(0, n, batch_size)]
    return torch.cat(outs, dim=0)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8,
                   help="8 was chosen for CPU memory limits during the "
                        "initial investigation -- raise substantially on GPU "
                        "(e.g. 64-128) where activation memory is not the "
                        "binding constraint")
    p.add_argument("--n-zeroshot", type=int, default=200,
                   help="CIFAR-10 test images for real zero-shot accuracy "
                        "(not just cos_sim vs the model's own fp32 output)")
    return p.parse_args()


def qat_finetune(name: str, teacher_image, teacher_text, proc, tok,
                 train_images, train_captions, eval_images, eval_captions,
                 epochs: int, lr: float, batch_size: int,
                 zeroshot_images=None, zeroshot_labels=None) -> dict:
    print(f"\n=== {name} QAT fine-tuning (device={DEVICE}) ===")
    teacher_image = teacher_image.to(DEVICE).eval()
    teacher_text = teacher_text.to(DEVICE).eval()

    # Convert to QAT (which deepcopies) BEFORE freezing the teacher --
    # deepcopy preserves requires_grad, so freezing first would have left
    # the student's copied weights frozen too.
    student_image = convert_to_qat(teacher_image).to(DEVICE)
    student_text = convert_to_qat(teacher_text).to(DEVICE)

    for p in teacher_image.parameters():
        p.requires_grad = False
    for p in teacher_text.parameters():
        p.requires_grad = False

    train_pv = proc(images=train_images, return_tensors="pt")["pixel_values"].to(DEVICE)
    train_txt = tok(train_captions, max_length=32, padding="max_length",
                    truncation=True, return_tensors="pt")
    train_txt = {k: v.to(DEVICE) for k, v in train_txt.items()}
    eval_pv = proc(images=eval_images, return_tensors="pt")["pixel_values"].to(DEVICE)
    eval_txt = tok(eval_captions, max_length=32, padding="max_length",
                   truncation=True, return_tensors="pt")
    eval_txt = {k: v.to(DEVICE) for k, v in eval_txt.items()}

    params = list(student_image.parameters()) + list(student_text.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)

    # QATLinear's fake-quant recomputes several full-size temporary
    # tensors per weight (round/clamp/etc.), which autograd then has to
    # retain for backward through STE -- memory scales with both model
    # size AND batch size. Mini-batching caps the activation-side peak
    # regardless of how large train_images is (the weight-side cost is
    # batch-independent, but this is still the lever that's actually
    # under our control here).
    BATCH_SIZE = batch_size
    n_train = train_pv.size(0)

    teacher_img_out = _batched_image_forward(teacher_image, train_pv, BATCH_SIZE)
    teacher_txt_out = _batched_text_forward(teacher_text, train_txt, BATCH_SIZE)

    for epoch in range(1, epochs + 1):
        student_image.train()
        student_text.train()
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            optimizer.zero_grad()

            student_img_out = student_image(train_pv[start:end])
            student_txt_out = student_text(input_ids=train_txt["input_ids"][start:end],
                                           attention_mask=train_txt["attention_mask"][start:end])
            loss = (distillation_loss(student_img_out, teacher_img_out[start:end]) +
                   distillation_loss(student_txt_out, teacher_txt_out[start:end]))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:02d}/{epochs}  distillation loss={epoch_loss / n_batches:.4f}")

    student_image.eval()
    student_text.eval()
    eval_teacher_img = _batched_image_forward(teacher_image, eval_pv, BATCH_SIZE)
    eval_teacher_txt = _batched_text_forward(teacher_text, eval_txt, BATCH_SIZE)
    eval_student_img = _batched_image_forward(student_image, eval_pv, BATCH_SIZE)
    eval_student_txt = _batched_text_forward(student_text, eval_txt, BATCH_SIZE)

    img_cos = _cosine_sim(eval_teacher_img, eval_student_img)
    txt_cos = _cosine_sim(eval_teacher_txt, eval_student_txt)
    print(f"  held-out eval cos_sim: image={img_cos:.4f}  text={txt_cos:.4f}")

    result = {"image_cos_sim": img_cos, "text_cos_sim": txt_cos, "final_loss": loss.item()}

    if zeroshot_images is not None:
        def qat_embed(imgs):
            pv = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
            return student_image(pv)

        acc = zero_shot_accuracy(qat_embed, student_text, tok, zeroshot_images, zeroshot_labels,
                                 device=DEVICE)
        result["zeroshot_accuracy"] = acc
        print(f"  zero-shot CIFAR-10 accuracy (real task accuracy, not cos_sim): {acc:.4f}")

    return result


def main():
    args = _parse_args()
    torch.manual_seed(0)

    n_total = args.n_train + args.n_eval
    print(f"Loading {n_total} real COCO images/captions "
          f"({args.n_train} train / {args.n_eval} held-out eval)...")
    raw = _stream_samples(n_total, offset=0)
    images = [s["image"] for s in raw]
    captions = [s["captions"][0] for s in raw]
    train_images, eval_images = images[:args.n_train], images[args.n_train:]
    train_captions, eval_captions = captions[:args.n_train], captions[args.n_train:]

    import torchvision
    print(f"Loading CIFAR-10 test set ({args.n_zeroshot} images) for real "
          f"zero-shot accuracy, not just cos_sim...")
    ds = torchvision.datasets.CIFAR10(root=".cifar10_cache", train=False, download=True)
    zeroshot_images = [ds[i][0] for i in range(args.n_zeroshot)]
    zeroshot_labels = [ds[i][1] for i in range(args.n_zeroshot)]

    import gc

    clip_img, clip_txt, clip_proc, clip_tok = build_clip()
    qat_finetune("CLIP", clip_img, clip_txt, clip_proc, clip_tok,
                train_images, train_captions, eval_images, eval_captions,
                args.epochs, args.lr, args.batch_size, zeroshot_images, zeroshot_labels)
    # Each qat_finetune call holds teacher + student + AdamW momentum/
    # variance buffers in memory (student is fully trainable, unlike PTQ's
    # frozen-encoder scripts) -- free CLIP's before building MobileCLIP's
    # rather than letting both stack up in main()'s scope simultaneously.
    del clip_img, clip_txt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mc_img, mc_txt, mc_proc, mc_tok = build_mobileclip()
    qat_finetune("MobileCLIP", mc_img, mc_txt, mc_proc, mc_tok,
                train_images, train_captions, eval_images, eval_captions,
                args.epochs, args.lr, args.batch_size, zeroshot_images, zeroshot_labels)


if __name__ == "__main__":
    main()
