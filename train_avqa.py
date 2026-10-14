"""
EMMI training script -- audio-visual QA (MUSIC-AVQA).

Extends train_vqa.py's open-ended-generation approach (ServerPipelineVQA /
GenerativeLLM, unchanged) to a task where edge-side compression must fuse
two real modalities -- audio + video -- rather than carry one modality
through untouched. See emma/pipeline.py's AVEdgePipeline and
emma/data/avqa.py for why: many MUSIC-AVQA questions ("how many
instruments are sounding") aren't answerable from audio or video alone.
The question text itself still bypasses compression, same as in
train_vqa.py -- only audio+video are fused/compressed.

Scope of this first version: same as train_vqa.py, --compression none
only (AVEdgePipeline supports use_compression=True but this script doesn't
train a VAE for the audio-visual fused space yet -- that's the natural
next step, not bundled here). Dataset scale is prototype-sized (see
emma/data/avqa.py's N_TRAIN_VIDEOS etc. -- real per-video downloads, not
free streaming) -- meant to prove the pipeline works, not to reproduce
MUSIC-AVQA benchmark numbers.

Evaluation: plain normalized exact-match accuracy (MUSIC-AVQA's answer
vocabulary is small and closed -- yes/no, counts, left/right/middle --
unlike VQAv2 there's a single ground-truth "anser" per question, not 10
human annotations, so the leave-one-out VQA accuracy metric doesn't apply).
"""

import argparse
import os
import string
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM

from emma.config import EMMAConfig, EncoderConfig
from emma.data.avqa import get_loaders
from emma.pipeline import AVEdgePipeline
from emma.alignment import CrossModalAlignment
from emma.encoders.audio_encoder import AudioEncoder
from emma.server.pipeline import (
    ServerPipelineVQA, GenerativeLLM, _build_projection, _load_llm_for_generation,
)

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["clip", "mobileclip"], default="clip",
                   help="edge image (frame) encoder family (default: clip)")
    p.add_argument("--scenario", choices=["llava", "gpt2"], default="gpt2",
                   help="server LLM backbone: 'llava' (real MLLM) or 'gpt2' "
                        "(fast debug iteration, no GPU needed; default here "
                        "since this script is prototype-scale)")
    p.add_argument("--ablation", choices=["fused", "audio_only", "video_only"], default="fused",
                   help="which modalities feed the server: 'fused' (default, "
                        "the actual proposal), or single-modality ablations "
                        "to test whether fusion is actually earning its keep")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--llm-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--batch-size",     type=int, default=8)
    p.add_argument("--epochs",         type=int, default=10)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--n-train-videos", type=int, default=60)
    p.add_argument("--n-valid-videos", type=int, default=15)
    p.add_argument("--n-test-videos",  type=int, default=15)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--max-question-len", type=int, default=32)
    p.add_argument("--max-answer-len",   type=int, default=4)
    p.add_argument("--patience",   type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=None)
    return p.parse_args()

_args = _parse_args()

# ── Hyperparameters ───────────────────────────────────────────────────────────

ENCODER        = _args.encoder
SCENARIO       = _args.scenario
ABLATION       = _args.ablation
LOAD_IN_8BIT   = _args.load_in_8bit
LLM_DTYPE      = _args.llm_dtype
BATCH_SIZE     = _args.batch_size
LR             = _args.lr
N_TRAIN_VIDEOS = _args.n_train_videos
N_VALID_VIDEOS = _args.n_valid_videos
N_TEST_VIDEOS  = _args.n_test_videos
SEED           = _args.seed
torch.manual_seed(SEED)
GRAD_CLIP      = 1.0
N_SOFT_TOKENS  = 8
CHECKPOINT_DIR = _args.checkpoint_dir
MAX_Q_LEN      = _args.max_question_len
MAX_A_LEN      = _args.max_answer_len
PATIENCE       = _args.patience
MAX_EPOCHS     = _args.max_epochs or (_args.epochs * 10)

_seed = f"_s{SEED}" if SEED != 0 else ""
CACHE_DIR = os.path.join(
    CHECKPOINT_DIR,
    f"avqa_embed_cache_{ENCODER}_{N_TRAIN_VIDEOS}_{N_VALID_VIDEOS}_{N_TEST_VIDEOS}",
)
CKPT_TAG  = f"avqa_{SCENARIO}_{ENCODER}_{ABLATION}{_seed}"

cfg = EMMAConfig(encoder=EncoderConfig(family=ENCODER, image_freeze_base=True),
                 use_compression=False)  # see module docstring -- no AV VAE trained yet
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model builders ────────────────────────────────────────────────────────────

def build_av_encoder():
    return AVEdgePipeline(config=cfg)


def build_server():
    if SCENARIO == "gpt2":
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        d_llm = model.config.hidden_size
        llm = GenerativeLLM(model, lm_head=None)
    else:
        llm, d_llm = _load_llm_for_generation(
            "llava", load_in_8bit=LOAD_IN_8BIT, dtype=LLM_DTYPE)
        if not LOAD_IN_8BIT:
            llm = llm.to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token

    d_shared = cfg.alignment.d_shared * {"mean": 1, "concat": 2, "match": 4}[cfg.alignment.fusion]
    projection = _build_projection(d_shared, N_SOFT_TOKENS, d_llm).to(DEVICE)
    server = ServerPipelineVQA(
        llm=llm, d_llm=d_llm, projection=projection,
        n_soft_tokens=N_SOFT_TOKENS, freeze_llm=True, vae_decoder=None,
    )
    return server, tokenizer


def configure_trainable(server: ServerPipelineVQA, audio_encoder: AudioEncoder) -> AdamW:
    """
    Trains projection + audio_encoder jointly. Unlike VQA's image-only setup
    (frozen pretrained CLIP -- caching its output once is lossless), the
    audio encoder has no pretrained weights: caching its output once would
    freeze it at a random init forever, so it has to stay in the gradient
    path (see cache_frame_embeddings -- only the *frame* embedding is
    cached, audio runs live every step).

    ABLATION == "video_only" never calls audio_encoder at all (see
    run_train_epoch), so its params are left out of the optimizer/trainable
    set entirely rather than trained on a disconnected graph.
    """
    for p in server.parameters():
        p.requires_grad = False
    for p in server.projection.parameters():
        p.requires_grad = True

    trainable = [p for p in server.parameters() if p.requires_grad]
    if ABLATION != "video_only":
        for p in audio_encoder.parameters():
            p.requires_grad = True
        trainable += list(audio_encoder.parameters())
    else:
        for p in audio_encoder.parameters():
            p.requires_grad = False

    n_params  = sum(p.numel() for p in trainable)
    print(f"\n── Stage 'server' (AVQA, {SCENARIO}, ablation={ABLATION}): "
          f"training {n_params:,} params ──")
    return AdamW(trainable, lr=LR, weight_decay=1e-2)


# ── Answer normalization + exact-match accuracy ───────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def avqa_accuracy(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


# ── Embedding cache (frozen frame encoder only -- audio runs live) ────────────

def cache_frame_embeddings(av_encoder, loaders: dict, cache_dir: str):
    """
    Caches only the frame (image) embedding -- the frozen, pretrained half
    of the edge pipeline, safe to precompute once like VQA's image cache.
    Raw waveform is cached too (not audio *embeddings*) so the trainable
    audio encoder can still run forward+backward on it every step; see
    configure_trainable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    av_encoder.eval()

    for split, loader in loaders.items():
        frame_embs, waveforms, q_ids, q_mask, a_ids, a_mask, answers_all = \
            [], [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                frames = batch["frames"].to(DEVICE)
                v = av_encoder._encode_frames(frames)
                frame_embs.append(v.cpu())
                waveforms.append(batch["waveform"])
                q_ids.append(batch["question_input_ids"])
                q_mask.append(batch["question_attn_mask"])
                a_ids.append(batch["answer_input_ids"])
                a_mask.append(batch["answer_attn_mask"])
                answers_all.extend(batch["answer"])

        assert frame_embs, (f"{split} split has 0 QA pairs -- every requested video "
                            f"failed to download/extract. Re-run with more videos "
                            f"(YouTube data rot means some are always dead).")
        torch.save(torch.cat(frame_embs), os.path.join(cache_dir, f"{split}_frame_emb.pt"))
        torch.save(torch.cat(waveforms),  os.path.join(cache_dir, f"{split}_waveform.pt"))
        torch.save(torch.cat(q_ids),      os.path.join(cache_dir, f"{split}_q_ids.pt"))
        torch.save(torch.cat(q_mask),     os.path.join(cache_dir, f"{split}_q_mask.pt"))
        torch.save(torch.cat(a_ids),      os.path.join(cache_dir, f"{split}_a_ids.pt"))
        torch.save(torch.cat(a_mask),     os.path.join(cache_dir, f"{split}_a_mask.pt"))
        torch.save(answers_all,           os.path.join(cache_dir, f"{split}_answers.pt"))
        n = sum(len(e) for e in frame_embs)
        print(f"  cached {split}: {n} QA pairs")

    print(f"  embeddings written to {cache_dir}/")


class CachedAVQADataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, split: str):
        self.frame_emb = torch.load(os.path.join(cache_dir, f"{split}_frame_emb.pt"), weights_only=True)
        self.waveform  = torch.load(os.path.join(cache_dir, f"{split}_waveform.pt"),  weights_only=True)
        self.q_ids  = torch.load(os.path.join(cache_dir, f"{split}_q_ids.pt"),  weights_only=True)
        self.q_mask = torch.load(os.path.join(cache_dir, f"{split}_q_mask.pt"), weights_only=True)
        self.a_ids  = torch.load(os.path.join(cache_dir, f"{split}_a_ids.pt"),  weights_only=True)
        self.a_mask = torch.load(os.path.join(cache_dir, f"{split}_a_mask.pt"), weights_only=True)
        self.answers = torch.load(os.path.join(cache_dir, f"{split}_answers.pt"), weights_only=False)

    def __len__(self):
        return len(self.frame_emb)

    def __getitem__(self, idx):
        return {
            "frame_emb": self.frame_emb[idx],
            "waveform":  self.waveform[idx],
            "q_ids":     self.q_ids[idx],
            "q_mask":    self.q_mask[idx],
            "a_ids":     self.a_ids[idx],
            "a_mask":    self.a_mask[idx],
            "answer":    self.answers[idx],
        }


def _collate(batch):
    out = {}
    for key in batch[0]:
        if key == "answer":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


def get_cached_loaders(cache_dir: str, batch_size: int) -> dict:
    loaders = {}
    for split in ("train", "valid", "test"):
        ds = CachedAVQADataset(cache_dir, split)
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=0, collate_fn=_collate,
        )
    return loaders


# ── Train / eval epochs ───────────────────────────────────────────────────────

def _compute_z(audio_encoder, alignment, v, waveform):
    """
    ABLATION selects what reaches the server: the actual proposal (fused
    audio+video) or a single-modality baseline. video_only never calls
    audio_encoder -- see configure_trainable for why that matters for
    which params end up in the optimizer.
    """
    if ABLATION == "video_only":
        return v
    a = audio_encoder(waveform)
    if ABLATION == "audio_only":
        return a
    return alignment(a, v)


def run_train_epoch(server, audio_encoder, alignment, loader, optimizer, scheduler) -> float:
    server.train()
    audio_encoder.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        v        = batch["frame_emb"].to(DEVICE)
        waveform = batch["waveform"].to(DEVICE)
        q_ids  = batch["q_ids"].to(DEVICE)
        q_mask = batch["q_mask"].to(DEVICE)
        a_ids  = batch["a_ids"].to(DEVICE)
        a_mask = batch["a_mask"].to(DEVICE)

        z = _compute_z(audio_encoder, alignment, v, waveform)

        out  = server(z, q_ids, q_mask, answer_ids=a_ids, answer_mask=a_mask)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        trainable = [p for p in server.parameters() if p.requires_grad]
        if ABLATION != "video_only":
            trainable += list(audio_encoder.parameters())
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
def run_eval_epoch(server, audio_encoder, alignment, loader, tokenizer) -> dict:
    server.eval()
    audio_encoder.eval()
    total_loss, n_batches = 0.0, 0
    accs = []
    for batch in loader:
        v        = batch["frame_emb"].to(DEVICE)
        waveform = batch["waveform"].to(DEVICE)
        q_ids  = batch["q_ids"].to(DEVICE)
        q_mask = batch["q_mask"].to(DEVICE)
        a_ids  = batch["a_ids"].to(DEVICE)
        a_mask = batch["a_mask"].to(DEVICE)

        z = _compute_z(audio_encoder, alignment, v, waveform)

        loss = server(z, q_ids, q_mask, answer_ids=a_ids, answer_mask=a_mask)["loss"]
        total_loss += loss.item()
        n_batches  += 1

        generated = server.generate_answer(
            z, q_ids, q_mask,
            eos_token_id=tokenizer.eos_token_id, max_new_tokens=MAX_A_LEN,
        )
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for pred, gold in zip(preds, batch["answer"]):
            accs.append(avqa_accuracy(pred, gold))

    return {
        "loss": total_loss / n_batches,
        "avqa_accuracy": sum(accs) / len(accs) if accs else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nBuilding models (scenario={SCENARIO}, encoder={ENCODER})...")
    server, tokenizer = build_server()

    embed_cache = os.path.join(CACHE_DIR, "train_frame_emb.pt")
    if os.path.exists(embed_cache):
        print(f"\nFound embedding cache at {CACHE_DIR} — skipping raw data loading and encoding.")
    else:
        print("Loading MUSIC-AVQA data (downloading clips -- this can take a while)...")
        loaders = get_loaders(
            tokenizer, batch_size=BATCH_SIZE,
            max_question_len=MAX_Q_LEN, max_answer_len=MAX_A_LEN,
            num_workers=0, n_train_videos=N_TRAIN_VIDEOS,
            n_valid_videos=N_VALID_VIDEOS, n_test_videos=N_TEST_VIDEOS,
            encoder=ENCODER, seed=SEED,
        )
        print(f"  train={len(loaders['train'].dataset)}  "
              f"valid={len(loaders['valid'].dataset)}  test={len(loaders['test'].dataset)}")

        print("\nEncoding frames with the frozen image encoder (audio stays live -- see configure_trainable)...")
        av_encoder = build_av_encoder().to(DEVICE)
        cache_frame_embeddings(av_encoder, loaders, CACHE_DIR)
        del av_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cached_loaders = get_cached_loaders(CACHE_DIR, batch_size=BATCH_SIZE)

    audio_encoder = AudioEncoder().to(DEVICE)
    alignment     = CrossModalAlignment(d_shared=cfg.alignment.d_shared, fusion=cfg.alignment.fusion)

    optimizer = configure_trainable(server, audio_encoder)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * max(1, len(cached_loaders["train"])))
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_train_epoch(server, audio_encoder, alignment,
                                     cached_loaders["train"], optimizer, scheduler)
        val_m      = run_eval_epoch(server, audio_encoder, alignment,
                                    cached_loaders["valid"], tokenizer)

        print(f"  [server] Epoch {epoch:02d}/{MAX_EPOCHS} | "
              f"train loss {train_loss:.4f} | "
              f"val loss {val_m['loss']:.4f} val AVQA-acc {val_m['avqa_accuracy']:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            no_improve    = 0
            torch.save({
                "stage": "server", "epoch": epoch,
                "server": server.state_dict(),
                "audio_encoder": audio_encoder.state_dict(),
                "val_loss": best_val_loss, "val_metrics": val_m,
            }, os.path.join(CHECKPOINT_DIR, f"best_server_{CKPT_TAG}.pt"))
            print(f"    ✓ saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f"  [server] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    best_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"best_server_{CKPT_TAG}.pt"), weights_only=False)
    server.load_state_dict(best_ckpt["server"])
    audio_encoder.load_state_dict(best_ckpt["audio_encoder"])
    print(f"\nReloaded best checkpoint from epoch {best_ckpt['epoch']} "
          f"(val_loss={best_ckpt['val_loss']:.4f})")

    print("\nFinal test evaluation...")
    test_m = run_eval_epoch(server, audio_encoder, alignment, cached_loaders["test"], tokenizer)
    print(f"Test | loss {test_m['loss']:.4f} | AVQA accuracy {test_m['avqa_accuracy']:.4f}")


if __name__ == "__main__":
    main()
