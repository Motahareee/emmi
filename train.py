"""
EMMA training script — staged training, no VAE compression, multi-task.

Two training stages:
  edge   — Projection heads (text, audio, vision) + cross-modal alignment
            trained jointly with the task loss. Backbone weights stay frozen;
            only the projection linears and the alignment module are updated.
            Training projections and alignment together gives the projections
            a meaningful gradient signal from the start.

  server — Server projection MLP + task heads only.
           Trained on fused embeddings cached after the edge stage so the
           server is completely decoupled from the edge pipeline.

DEBUG=True  uses GPT-2 as the server LLM (small, no GPU needed)
DEBUG=False uses the scenario set in ServerConfig
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from emma.config import EMMAConfig, EncoderConfig, AlignmentConfig, VAEConfig, ServerConfig
from emma.data import get_loaders
from emma.pipeline import EdgePipeline
from emma.server.pipeline import ServerPipeline, _build_projection

# ── Hyperparameters ───────────────────────────────────────────────────────────

DEBUG             = True
BATCH_SIZE        = 16
LR                = 1e-3
GRAD_CLIP         = 1.0
MAX_TEXT_LEN      = 128
MAX_AUDIO_SAMPLES = 132300      # 6 s @ 22050 Hz
N_SOFT_TOKENS     = 8
CHECKPOINT_DIR    = "checkpoints"
CACHE_DIR         = "checkpoints/embed_cache"
INSTRUCTION       = "Analyze sentiment and emotions:"

# Epochs per stage
STAGE_EPOCHS = {
    "edge":   10,   # projections + alignment jointly
    "server": 10,   # server projection + task heads on cached embeddings
}

# ── Config ────────────────────────────────────────────────────────────────────

cfg = EMMAConfig(
    encoder=EncoderConfig(text_freeze_base=True, audio_freeze_base=True),
    alignment=AlignmentConfig(d_shared=256),
    vae=VAEConfig(d_latent=64),
    server=ServerConfig(
        scenario="plain_llm",
        n_soft_tokens=N_SOFT_TOKENS,
        freeze_llm=True,
        sentiment_weight=1.0,
        emotion_weight=1.0,
    ),
    use_compression=False,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model builders ────────────────────────────────────────────────────────────

def build_edge(cfg: EMMAConfig) -> EdgePipeline:
    return EdgePipeline(
        config=cfg,
        text_encoder=None,   # DistilBERT from config
        audio_encoder=None,  # Whisper-small from config
    )


def build_server(cfg: EMMAConfig, debug: bool) -> tuple[ServerPipeline, AutoTokenizer]:
    sc = cfg.server
    ac = cfg.alignment

    if debug:
        llm       = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        d_llm = llm.config.hidden_size
    else:
        from emma.server.pipeline import _load_llm
        llm, d_llm = _load_llm(sc.scenario, sc.load_in_8bit)
        tokenizer  = AutoTokenizer.from_pretrained(
            {"llava": "llava-hf/llava-1.5-7b-hf",
             "qwen_audio": "Qwen/Qwen2-Audio-7B-Instruct",
             "plain_llm": "mistralai/Mistral-7B-v0.1"}[sc.scenario]
        )
        tokenizer.pad_token = tokenizer.eos_token

    server = ServerPipeline(
        llm=llm,
        d_llm=d_llm,
        projection=_build_projection(ac.d_shared, sc.n_soft_tokens, d_llm),
        sentiment_head=nn.Linear(d_llm, 1),
        emotion_head=nn.Linear(d_llm, 6),
        n_soft_tokens=sc.n_soft_tokens,
        freeze_llm=sc.freeze_llm,
        vae_decoder=None,
    )
    return server, tokenizer


# ── Staged training helpers ───────────────────────────────────────────────────

def freeze_all(edge: EdgePipeline, server: ServerPipeline):
    """Freeze every parameter in both pipelines."""
    for p in list(edge.parameters()) + list(server.parameters()):
        p.requires_grad = False


def configure_stage(stage: str, edge: EdgePipeline,
                    server: ServerPipeline) -> AdamW:
    """
    Freeze everything, then unfreeze only the components for this stage.
    Returns a fresh AdamW optimizer over the newly active parameters.

    "edge"   — projection heads (text, audio, vision) + alignment jointly.
               Backbone weights stay frozen; only projections and alignment
               are updated. The task loss gives projections a real gradient.
    "server" — server projection MLP + sentiment head + emotion head.
               (edge not used; call only when edge is available for reference)
    """
    freeze_all(edge, server)

    stage_map = {
        "edge": (
            [edge.text_encoder.proj, edge.audio_encoder.proj, edge.alignment],
            "projection heads + alignment",
        ),
        "server": (
            [server.projection, server.sentiment_head, server.emotion_head],
            "server projection + task heads",
        ),
    }

    groups, label = stage_map[stage]
    for module in groups:
        for p in module.parameters():
            p.requires_grad = True

    trainable = [p for p in list(edge.parameters()) + list(server.parameters())
                 if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"\n── Stage '{stage}': training {label} ({n_params:,} params) ──")

    return AdamW(trainable, lr=LR, weight_decay=1e-2)


# ── Loss ──────────────────────────────────────────────────────────────────────

def multitask_loss(preds: dict, batch: dict) -> dict:
    sc     = cfg.server
    s_loss = F.mse_loss(preds["sentiment"].squeeze(-1), batch["sentiment"])
    e_loss = F.binary_cross_entropy_with_logits(preds["emotions"], batch["emotions"])
    total  = sc.sentiment_weight * s_loss + sc.emotion_weight * e_loss
    return {"loss": total, "sentiment_loss": s_loss, "emotion_loss": e_loss}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(sent_preds, sent_targets, emo_preds, emo_targets) -> dict:
    sent_preds   = torch.cat(sent_preds).squeeze(-1)
    sent_targets = torch.cat(sent_targets)
    emo_preds    = torch.cat(emo_preds)
    emo_targets  = torch.cat(emo_targets)

    mae        = (sent_preds - sent_targets).abs().mean().item()
    binary_acc = ((sent_preds > 0) == (sent_targets > 0)).float().mean().item()
    emo_acc    = ((torch.sigmoid(emo_preds) > 0.5) == emo_targets.bool()).float().mean().item()

    return {"sentiment_mae": mae, "sentiment_acc": binary_acc, "emotion_acc": emo_acc}


# ── Train / eval loop (Stages 1–2, full edge pipeline) ───────────────────────

def run_epoch(edge, server, loader, optimizer, scheduler,
              instr_ids, instr_mask, training: bool) -> dict:
    edge.train(training)
    server.train(training)

    total_loss = s_loss_sum = e_loss_sum = 0.0
    sent_preds, sent_tgts, emo_preds, emo_tgts = [], [], [], []
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch        = {k: v.to(DEVICE) for k, v in batch.items()}
            B            = batch["sentiment"].size(0)
            instr_ids_b  = instr_ids.expand(B, -1).to(DEVICE)
            instr_mask_b = instr_mask.expand(B, -1).to(DEVICE)

            fused = edge(
                text_inputs  = {"input_ids":      batch["input_ids"],
                                "attention_mask": batch["attention_mask"]},
                audio_inputs = {"waveform":       batch["waveform"]},
                training=False,
            )

            preds  = server(fused, instr_ids_b, instr_mask_b)
            losses = multitask_loss(preds, batch)

            if training:
                optimizer.zero_grad()
                losses["loss"].backward()
                active = [p for p in list(edge.parameters()) + list(server.parameters())
                          if p.requires_grad]
                nn.utils.clip_grad_norm_(active, GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            total_loss += losses["loss"].item()
            s_loss_sum += losses["sentiment_loss"].item()
            e_loss_sum += losses["emotion_loss"].item()
            sent_preds.append(preds["sentiment"].detach().cpu())
            sent_tgts.append(batch["sentiment"].cpu())
            emo_preds.append(preds["emotions"].detach().cpu())
            emo_tgts.append(batch["emotions"].cpu())
            n_batches += 1

    metrics = compute_metrics(sent_preds, sent_tgts, emo_preds, emo_tgts)
    metrics.update({
        "loss":           total_loss / n_batches,
        "sentiment_loss": s_loss_sum / n_batches,
        "emotion_loss":   e_loss_sum / n_batches,
    })
    return metrics


# ── Embedding cache (decouples Stage 3 from edge pipeline) ───────────────────

def cache_embeddings(edge: EdgePipeline, loaders: dict, cache_dir: str):
    """
    Run the frozen edge pipeline over every split and save fused embeddings
    plus labels to disk.  Stage 3 loads these files instead of re-running
    the (expensive) edge pipeline every batch.

    Saves per split:
        {cache_dir}/{split}_fused.pt      — FloatTensor [N, d_shared]
        {cache_dir}/{split}_sentiment.pt  — FloatTensor [N]
        {cache_dir}/{split}_emotions.pt   — FloatTensor [N, 6]
    """
    os.makedirs(cache_dir, exist_ok=True)
    edge.eval()

    for split, loader in loaders.items():
        fused_list, sent_list, emo_list = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                fused = edge(
                    text_inputs  = {"input_ids":      batch["input_ids"],
                                    "attention_mask": batch["attention_mask"]},
                    audio_inputs = {"waveform":       batch["waveform"]},
                    training=False,
                )
                fused_list.append(fused.cpu())
                sent_list.append(batch["sentiment"].cpu())
                emo_list.append(batch["emotions"].cpu())

        torch.save(torch.cat(fused_list),  os.path.join(cache_dir, f"{split}_fused.pt"))
        torch.save(torch.cat(sent_list),   os.path.join(cache_dir, f"{split}_sentiment.pt"))
        torch.save(torch.cat(emo_list),    os.path.join(cache_dir, f"{split}_emotions.pt"))
        print(f"  cached {split}: {len(fused_list[0] if len(fused_list)==1 else torch.cat(fused_list))} samples")  # noqa: E501 (best-effort)

    print(f"  embeddings written to {cache_dir}/")


class CachedEmbeddingDataset(Dataset):
    """Loads pre-cached fused embeddings and labels for Stage 3 training."""

    def __init__(self, cache_dir: str, split: str):
        self.fused     = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"),
                                    weights_only=True)
        self.sentiment = torch.load(os.path.join(cache_dir, f"{split}_sentiment.pt"),
                                    weights_only=True)
        self.emotions  = torch.load(os.path.join(cache_dir, f"{split}_emotions.pt"),
                                    weights_only=True)

    def __len__(self):
        return len(self.fused)

    def __getitem__(self, idx):
        return {
            "fused":     self.fused[idx],
            "sentiment": self.sentiment[idx],
            "emotions":  self.emotions[idx],
        }


def get_cached_loaders(cache_dir: str, batch_size: int) -> dict:
    loaders = {}
    for split in ("train", "valid", "test"):
        ds = CachedEmbeddingDataset(cache_dir, split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
    return loaders


# ── Stage 3 train / eval loop (cached embeddings only) ───────────────────────

def run_server_epoch(server, loader, optimizer, scheduler,
                     instr_ids, instr_mask, training: bool) -> dict:
    """Like run_epoch but works from cached embeddings; no edge pipeline."""
    server.train(training)

    total_loss = s_loss_sum = e_loss_sum = 0.0
    sent_preds, sent_tgts, emo_preds, emo_tgts = [], [], [], []
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch        = {k: v.to(DEVICE) for k, v in batch.items()}
            B            = batch["sentiment"].size(0)
            instr_ids_b  = instr_ids.expand(B, -1).to(DEVICE)
            instr_mask_b = instr_mask.expand(B, -1).to(DEVICE)

            preds  = server(batch["fused"], instr_ids_b, instr_mask_b)
            losses = multitask_loss(preds, batch)

            if training:
                optimizer.zero_grad()
                losses["loss"].backward()
                trainable = [p for p in server.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            total_loss += losses["loss"].item()
            s_loss_sum += losses["sentiment_loss"].item()
            e_loss_sum += losses["emotion_loss"].item()
            sent_preds.append(preds["sentiment"].detach().cpu())
            sent_tgts.append(batch["sentiment"].cpu())
            emo_preds.append(preds["emotions"].detach().cpu())
            emo_tgts.append(batch["emotions"].cpu())
            n_batches += 1

    metrics = compute_metrics(sent_preds, sent_tgts, emo_preds, emo_tgts)
    metrics.update({
        "loss":           total_loss / n_batches,
        "sentiment_loss": s_loss_sum / n_batches,
        "emotion_loss":   e_loss_sum / n_batches,
    })
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading data...")
    loaders = get_loaders(batch_size=BATCH_SIZE, max_text_len=MAX_TEXT_LEN,
                          max_audio_samples=MAX_AUDIO_SAMPLES, num_workers=0)
    print(f"  train={len(loaders['train'].dataset)}  "
          f"valid={len(loaders['valid'].dataset)}  "
          f"test={len(loaders['test'].dataset)}")

    print(f"Building models (debug={DEBUG})...")
    edge              = build_edge(cfg).to(DEVICE)
    server, tokenizer = build_server(cfg, debug=DEBUG)
    server            = server.to(DEVICE)

    instr_enc  = tokenizer(INSTRUCTION, return_tensors="pt",
                           padding=True, truncation=True, max_length=32)
    instr_ids  = instr_enc["input_ids"]
    instr_mask = instr_enc["attention_mask"]

    # ── Edge stage: projections + alignment trained jointly ───────────────────
    edge_ckpt = os.path.join(CHECKPOINT_DIR, "best_edge.pt")
    if os.path.exists(edge_ckpt):
        print(f"\nFound {edge_ckpt} — skipping edge training, loading checkpoint.")
        ckpt = torch.load(edge_ckpt, map_location=DEVICE, weights_only=False)
        edge.load_state_dict(ckpt["edge"])
        stages_to_run = []
    else:
        stages_to_run = ["edge"]

    for stage in stages_to_run:
        optimizer     = configure_stage(stage, edge, server)
        n_epochs      = STAGE_EPOCHS[stage]
        scheduler     = CosineAnnealingLR(optimizer,
                                          T_max=n_epochs * len(loaders["train"]))
        best_val_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            train_m = run_epoch(edge, server, loaders["train"], optimizer, scheduler,
                                instr_ids, instr_mask, training=True)
            val_m   = run_epoch(edge, server, loaders["valid"], optimizer, scheduler,
                                instr_ids, instr_mask, training=False)

            print(f"  [{stage}] Epoch {epoch:02d}/{n_epochs} | "
                  f"train {train_m['loss']:.4f} "
                  f"(sent {train_m['sentiment_loss']:.4f} "
                  f"emo {train_m['emotion_loss']:.4f}) | "
                  f"val {val_m['loss']:.4f} | "
                  f"MAE {val_m['sentiment_mae']:.3f} | "
                  f"emo_acc {val_m['emotion_acc']:.3f}")

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                torch.save({
                    "stage":       stage,
                    "epoch":       epoch,
                    "edge":        edge.state_dict(),
                    "server":      server.state_dict(),
                    "val_loss":    best_val_loss,
                    "val_metrics": val_m,
                }, os.path.join(CHECKPOINT_DIR, "best_edge.pt"))
                print(f"    ✓ saved (val_loss={best_val_loss:.4f})")

    # ── Cache fused embeddings after alignment stage ─────────────────────────
    print("\nCaching fused embeddings (alignment stage output)...")
    cache_embeddings(edge, loaders, CACHE_DIR)

    # Free edge pipeline memory before Stage 3
    del edge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cached_loaders = get_cached_loaders(CACHE_DIR, batch_size=BATCH_SIZE)

    # ── Server stage: trained from cached embeddings only ────────────────────
    #    Edge is deleted; manage optimizer directly over server params.
    for p in server.parameters():
        p.requires_grad = False
    for module in [server.projection, server.sentiment_head, server.emotion_head]:
        for p in module.parameters():
            p.requires_grad = True

    trainable = [p for p in server.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"\n── Stage 'server': training server projection + task heads ({n_params:,} params) ──")
    optimizer = AdamW(trainable, lr=LR, weight_decay=1e-2)

    n_epochs  = STAGE_EPOCHS["server"]
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=n_epochs * len(cached_loaders["train"]))
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_m = run_server_epoch(server, cached_loaders["train"], optimizer, scheduler,
                                   instr_ids, instr_mask, training=True)
        val_m   = run_server_epoch(server, cached_loaders["valid"], optimizer, scheduler,
                                   instr_ids, instr_mask, training=False)

        print(f"  [server] Epoch {epoch:02d}/{n_epochs} | "
              f"train {train_m['loss']:.4f} "
              f"(sent {train_m['sentiment_loss']:.4f} "
              f"emo {train_m['emotion_loss']:.4f}) | "
              f"val {val_m['loss']:.4f} | "
              f"MAE {val_m['sentiment_mae']:.3f} | "
              f"emo_acc {val_m['emotion_acc']:.3f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "stage":       "server",
                "epoch":       epoch,
                "server":      server.state_dict(),
                "val_loss":    best_val_loss,
                "val_metrics": val_m,
            }, os.path.join(CHECKPOINT_DIR, "best_server.pt"))
            print(f"    ✓ saved (val_loss={best_val_loss:.4f})")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nFinal test evaluation...")
    test_m = run_server_epoch(server, cached_loaders["test"], optimizer, scheduler,
                              instr_ids, instr_mask, training=False)
    print(f"Test | loss {test_m['loss']:.4f} | "
          f"MAE {test_m['sentiment_mae']:.3f} | "
          f"sent_acc {test_m['sentiment_acc']:.3f} | "
          f"emo_acc {test_m['emotion_acc']:.3f}")


if __name__ == "__main__":
    main()
