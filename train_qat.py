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


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def qat_finetune(name: str, teacher_image, teacher_text, proc, tok,
                 train_images, train_captions, eval_images, eval_captions,
                 epochs: int, lr: float) -> dict:
    print(f"\n=== {name} QAT fine-tuning ===")
    teacher_image.eval()
    teacher_text.eval()

    # Convert to QAT (which deepcopies) BEFORE freezing the teacher --
    # deepcopy preserves requires_grad, so freezing first would have left
    # the student's copied weights frozen too.
    student_image = convert_to_qat(teacher_image)
    student_text = convert_to_qat(teacher_text)

    for p in teacher_image.parameters():
        p.requires_grad = False
    for p in teacher_text.parameters():
        p.requires_grad = False

    train_pv = proc(images=train_images, return_tensors="pt")["pixel_values"]
    train_txt = tok(train_captions, max_length=32, padding="max_length",
                    truncation=True, return_tensors="pt")
    eval_pv = proc(images=eval_images, return_tensors="pt")["pixel_values"]
    eval_txt = tok(eval_captions, max_length=32, padding="max_length",
                   truncation=True, return_tensors="pt")

    params = list(student_image.parameters()) + list(student_text.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)

    # QATLinear's fake-quant recomputes several full-size temporary
    # tensors per weight (round/clamp/etc.), which autograd then has to
    # retain for backward through STE -- memory scales with both model
    # size AND batch size. Mini-batching caps the activation-side peak
    # regardless of how large train_images is (the weight-side cost is
    # batch-independent, but this is still the lever that's actually
    # under our control here).
    BATCH_SIZE = 8
    n_train = train_pv.size(0)

    with torch.no_grad():
        teacher_img_out = teacher_image(train_pv)
        teacher_txt_out = teacher_text(**train_txt)

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
    with torch.no_grad():
        eval_teacher_img = teacher_image(eval_pv)
        eval_teacher_txt = teacher_text(**eval_txt)
        eval_student_img = student_image(eval_pv)
        eval_student_txt = student_text(**eval_txt)

    img_cos = _cosine_sim(eval_teacher_img, eval_student_img)
    txt_cos = _cosine_sim(eval_teacher_txt, eval_student_txt)
    print(f"  held-out eval cos_sim: image={img_cos:.4f}  text={txt_cos:.4f}")
    return {"image_cos_sim": img_cos, "text_cos_sim": txt_cos, "final_loss": loss.item()}


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

    import gc

    clip_img, clip_txt, clip_proc, clip_tok = build_clip()
    qat_finetune("CLIP", clip_img, clip_txt, clip_proc, clip_tok,
                train_images, train_captions, eval_images, eval_captions,
                args.epochs, args.lr)
    # Each qat_finetune call holds teacher + student + AdamW momentum/
    # variance buffers in memory (student is fully trainable, unlike PTQ's
    # frozen-encoder scripts) -- free CLIP's before building MobileCLIP's
    # rather than letting both stack up in main()'s scope simultaneously.
    del clip_img, clip_txt
    gc.collect()

    mc_img, mc_txt, mc_proc, mc_tok = build_mobileclip()
    qat_finetune("MobileCLIP", mc_img, mc_txt, mc_proc, mc_tok,
                train_images, train_captions, eval_images, eval_captions,
                args.epochs, args.lr)


if __name__ == "__main__":
    main()
