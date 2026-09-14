"""
EMMI training script -- open-ended VQA (visual question answering).

Unlike train.py/train_llava.py (binary image-text matching), this task
keeps the pipeline genuinely multimodal at the input (image + question
text) while requiring real language generation that can't be solved by
embedding similarity alone -- addressing the critique that binary
matching is nearly solved by CLIP's own embedding geometry and doesn't
exercise the server LLM's reasoning capacity.

Scope of this first version: --compression none only. VQA needs
image-only compression (there's no caption to fuse with at inference --
generating the answer is the task), and none of the project's 7 existing
compressors were trained on image-only embeddings (they were all trained
on fused image+text match-fusion vectors). Extending compression to this
setting is a real follow-up decision, not something to bundle in here.

Server LLM: --scenario llava (default) or gpt2 (--debug, fast iteration,
no GPU needed) via ServerPipelineVQA / GenerativeLLM in
emma/server/pipeline.py. Only "llava" and "causal_lm"-loader scenarios
are implemented for generation -- "qwen_audio" is registered in
SCENARIO_REGISTRY but has never been validated in this project.

Evaluation: the official VQA accuracy metric (Antol et al. 2015) --
leave-one-out over the 10 human answers, min(matches/3, 1), averaged.
Answer text is normalized (lowercase, strip punctuation/articles) before
comparison -- a simplified version of the official VQA eval script's
normalization, not byte-for-byte identical to it.
"""

import argparse
import os
import re
import string
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM

from emma.config import EMMAConfig, EncoderConfig
from emma.data.vqa import get_loaders
from emma.pipeline import EdgePipeline
from emma.server.pipeline import (
    ServerPipelineVQA, _build_projection, _load_llm_for_generation,
)

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["clip", "mobileclip"], default="mobileclip",
                   help="edge image encoder family (default: mobileclip)")
    p.add_argument("--scenario", choices=["llava", "gpt2"], default="llava",
                   help="server LLM backbone: 'llava' (real MLLM) or 'gpt2' "
                        "(fast debug iteration, no GPU needed)")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="load the llava backbone with bitsandbytes 8-bit quantization")
    p.add_argument("--llm-dtype", choices=["float16", "float32"], default="float16",
                   help="precision for the llava backbone when --load-in-8bit is not set")
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--epochs",         type=int, default=20)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--n-train", type=int, default=5_000)
    p.add_argument("--n-valid", type=int, default=500)
    p.add_argument("--n-test",  type=int, default=500)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--max-question-len", type=int, default=32)
    p.add_argument("--max-answer-len",   type=int, default=8)
    p.add_argument("--patience",   type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=None)
    return p.parse_args()

_args = _parse_args()

# ── Hyperparameters ───────────────────────────────────────────────────────────

ENCODER        = _args.encoder
SCENARIO       = _args.scenario
LOAD_IN_8BIT   = _args.load_in_8bit
LLM_DTYPE      = _args.llm_dtype
BATCH_SIZE     = _args.batch_size
LR             = _args.lr
N_TRAIN        = _args.n_train
N_VALID        = _args.n_valid
N_TEST         = _args.n_test
SEED           = _args.seed
torch.manual_seed(SEED)
GRAD_CLIP      = 1.0
N_SOFT_TOKENS  = 8
CHECKPOINT_DIR = _args.checkpoint_dir
MAX_Q_LEN      = _args.max_question_len
MAX_A_LEN      = _args.max_answer_len
PATIENCE       = _args.patience
MAX_EPOCHS     = _args.max_epochs or (_args.epochs * 10)

_enc  = "" if ENCODER == "clip" else f"_{ENCODER}"
_seed = f"_s{SEED}" if SEED != 0 else ""
CACHE_DIR = os.path.join(CHECKPOINT_DIR, f"vqa_embed_cache{_enc}_{N_TRAIN}_{N_VALID}_{N_TEST}")
CKPT_TAG  = f"vqa_{SCENARIO}{_enc}{_seed}"

D_IMG = 512  # both CLIP and MobileCLIP produce 512-dim image embeddings

cfg = EMMAConfig(encoder=EncoderConfig(family=ENCODER, image_freeze_base=True))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model builders ────────────────────────────────────────────────────────────

def build_image_encoder():
    """Only the frozen image encoder is needed -- the question is handled
    directly by the LLM side, not via CLIP's text encoder / fusion."""
    return EdgePipeline(config=cfg).image_encoder


def build_server():
    if SCENARIO == "gpt2":
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        d_llm = model.config.hidden_size
        from emma.server.pipeline import GenerativeLLM
        llm = GenerativeLLM(model, lm_head=None)
    else:
        llm, d_llm = _load_llm_for_generation(
            "llava", load_in_8bit=LOAD_IN_8BIT, dtype=LLM_DTYPE)
        if not LOAD_IN_8BIT:
            llm = llm.to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token

    projection = _build_projection(D_IMG, N_SOFT_TOKENS, d_llm).to(DEVICE)
    server = ServerPipelineVQA(
        llm=llm, d_llm=d_llm, projection=projection,
        n_soft_tokens=N_SOFT_TOKENS, freeze_llm=True, vae_decoder=None,
    )
    return server, tokenizer


def configure_server_stage(server: ServerPipelineVQA) -> AdamW:
    for p in server.parameters():
        p.requires_grad = False
    for p in server.projection.parameters():
        p.requires_grad = True
    trainable = [p for p in server.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"\n── Stage 'server' (VQA, {SCENARIO}): training projection only ({n_params:,} params) ──")
    return AdamW(trainable, lr=LR, weight_decay=1e-2)


# ── VQA answer normalization + official accuracy metric ──────────────────────

_ARTICLES = {"a", "an", "the"}

def normalize_answer(s: str) -> str:
    """
    Simplified version of the official VQA eval script's normalization:
    lowercase, strip punctuation, collapse whitespace, drop leading
    articles. Not byte-for-byte identical to the official script (which
    also converts number words to digits and expands contractions) --
    good enough for a first working version, not a literature-exact score.
    """
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in s.split() if w not in _ARTICLES]
    return " ".join(words)


def vqa_accuracy(pred: str, human_answers: list) -> float:
    """
    Official VQA accuracy (Antol et al. 2015): leave-one-out over the up
    to 10 human answers -- for each of the N ways of holding one out,
    score min(matches-among-the-rest / 3, 1), then average.
    """
    pred = normalize_answer(pred)
    answers = [normalize_answer(a) for a in human_answers]
    n = len(answers)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0 if pred == answers[0] else 0.0

    accs = []
    for i in range(n):
        others  = answers[:i] + answers[i+1:]
        matches = sum(1 for a in others if a == pred)
        accs.append(min(matches / 3, 1.0))
    return sum(accs) / len(accs)


# ── Embedding cache (image-only) ──────────────────────────────────────────────

def cache_image_embeddings(image_encoder, loaders: dict, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    image_encoder.eval()

    for split, loader in loaders.items():
        embs, q_ids, q_mask, a_ids, a_mask, answers_all = [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                pv = batch["pixel_values"].to(DEVICE)
                emb = image_encoder(pv)
                embs.append(emb.cpu())
                q_ids.append(batch["question_input_ids"])
                q_mask.append(batch["question_attn_mask"])
                a_ids.append(batch["answer_input_ids"])
                a_mask.append(batch["answer_attn_mask"])
                answers_all.extend(batch["answers"])

        torch.save(torch.cat(embs),   os.path.join(cache_dir, f"{split}_image_emb.pt"))
        torch.save(torch.cat(q_ids),  os.path.join(cache_dir, f"{split}_q_ids.pt"))
        torch.save(torch.cat(q_mask), os.path.join(cache_dir, f"{split}_q_mask.pt"))
        torch.save(torch.cat(a_ids),  os.path.join(cache_dir, f"{split}_a_ids.pt"))
        torch.save(torch.cat(a_mask), os.path.join(cache_dir, f"{split}_a_mask.pt"))
        torch.save(answers_all,       os.path.join(cache_dir, f"{split}_answers.pt"))
        n = sum(len(e) for e in embs)
        print(f"  cached {split}: {n} questions")

    print(f"  embeddings written to {cache_dir}/")


class CachedVQADataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, split: str):
        self.emb    = torch.load(os.path.join(cache_dir, f"{split}_image_emb.pt"), weights_only=True)
        self.q_ids  = torch.load(os.path.join(cache_dir, f"{split}_q_ids.pt"),      weights_only=True)
        self.q_mask = torch.load(os.path.join(cache_dir, f"{split}_q_mask.pt"),     weights_only=True)
        self.a_ids  = torch.load(os.path.join(cache_dir, f"{split}_a_ids.pt"),      weights_only=True)
        self.a_mask = torch.load(os.path.join(cache_dir, f"{split}_a_mask.pt"),     weights_only=True)
        self.answers = torch.load(os.path.join(cache_dir, f"{split}_answers.pt"), weights_only=False)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return {
            "image_emb":  self.emb[idx],
            "q_ids":      self.q_ids[idx],
            "q_mask":     self.q_mask[idx],
            "a_ids":      self.a_ids[idx],
            "a_mask":     self.a_mask[idx],
            "answers":    self.answers[idx],
        }


def _collate(batch):
    out = {}
    for key in batch[0]:
        if key == "answers":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


def get_cached_loaders(cache_dir: str, batch_size: int) -> dict:
    loaders = {}
    for split in ("train", "valid", "test"):
        ds = CachedVQADataset(cache_dir, split)
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=0, collate_fn=_collate,
        )
    return loaders


# ── Train / eval epochs ───────────────────────────────────────────────────────

def run_train_epoch(server, loader, optimizer, scheduler) -> float:
    server.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        z      = batch["image_emb"].to(DEVICE)
        q_ids  = batch["q_ids"].to(DEVICE)
        q_mask = batch["q_mask"].to(DEVICE)
        a_ids  = batch["a_ids"].to(DEVICE)
        a_mask = batch["a_mask"].to(DEVICE)

        out  = server(z, q_ids, q_mask, answer_ids=a_ids, answer_mask=a_mask)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        trainable = [p for p in server.parameters() if p.requires_grad]
        grad_norm = nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:
            print(f"    [warn] non-finite grad norm ({float(grad_norm):.3g}) -- skipping optimizer step")
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def run_eval_epoch(server, loader, tokenizer) -> dict:
    server.eval()
    total_loss, n_batches = 0.0, 0
    accs = []
    for batch in loader:
        z      = batch["image_emb"].to(DEVICE)
        q_ids  = batch["q_ids"].to(DEVICE)
        q_mask = batch["q_mask"].to(DEVICE)
        a_ids  = batch["a_ids"].to(DEVICE)
        a_mask = batch["a_mask"].to(DEVICE)

        loss = server(z, q_ids, q_mask, answer_ids=a_ids, answer_mask=a_mask)["loss"]
        total_loss += loss.item()
        n_batches  += 1

        generated = server.generate_answer(
            z, q_ids, q_mask,
            eos_token_id=tokenizer.eos_token_id, max_new_tokens=MAX_A_LEN,
        )
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for pred, human_answers in zip(preds, batch["answers"]):
            accs.append(vqa_accuracy(pred, human_answers))

    return {
        "loss": total_loss / n_batches,
        "vqa_accuracy": sum(accs) / len(accs) if accs else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nBuilding models (scenario={SCENARIO}, encoder={ENCODER})...")
    server, tokenizer = build_server()

    embed_cache = os.path.join(CACHE_DIR, "train_image_emb.pt")
    if os.path.exists(embed_cache):
        print(f"\nFound embedding cache at {CACHE_DIR} — skipping raw data loading and encoding.")
    else:
        print("Loading VQAv2 data...")
        loaders = get_loaders(
            tokenizer,
            batch_size=BATCH_SIZE, max_question_len=MAX_Q_LEN, max_answer_len=MAX_A_LEN,
            num_workers=0, n_train=N_TRAIN, n_valid=N_VALID, n_test=N_TEST, encoder=ENCODER,
        )
        print(f"  train={len(loaders['train'].dataset)}  "
              f"valid={len(loaders['valid'].dataset)}  test={len(loaders['test'].dataset)}")

        print("\nEncoding images with frozen encoder...")
        image_encoder = build_image_encoder().to(DEVICE)
        cache_image_embeddings(image_encoder, loaders, CACHE_DIR)
        del image_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cached_loaders = get_cached_loaders(CACHE_DIR, batch_size=BATCH_SIZE)

    optimizer = configure_server_stage(server)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * len(cached_loaders["train"]))
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_train_epoch(server, cached_loaders["train"], optimizer, scheduler)
        val_m      = run_eval_epoch(server, cached_loaders["valid"], tokenizer)

        print(f"  [server] Epoch {epoch:02d}/{MAX_EPOCHS} | "
              f"train loss {train_loss:.4f} | "
              f"val loss {val_m['loss']:.4f} val VQA-acc {val_m['vqa_accuracy']:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            no_improve    = 0
            torch.save({
                "stage": "server", "epoch": epoch,
                "server": server.state_dict(),
                "val_loss": best_val_loss, "val_metrics": val_m,
            }, os.path.join(CHECKPOINT_DIR, f"best_server_{CKPT_TAG}.pt"))
            print(f"    ✓ saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f"  [server] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print("\nFinal test evaluation...")
    test_m = run_eval_epoch(server, cached_loaders["test"], tokenizer)
    print(f"Test | loss {test_m['loss']:.4f} | VQA accuracy {test_m['vqa_accuracy']:.4f}")


if __name__ == "__main__":
    main()
