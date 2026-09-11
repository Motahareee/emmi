"""
EMMA training script — COCO image-text matching, binary classification.

Two independent training stages (edge CLIP encoders are fully frozen):

  Stage 1 — compression: VAE or PCA  (no server, no encoder training)
    Frozen CLIP encoders produce fused embeddings [B, 512].
    VAE: β-VAE loss (reconstruction MSE + KL divergence)
    PCA: closed-form fit, no gradient training needed
    Produces cached latents [B, 64] for Stage 2.
    Skipped when COMPRESSION="none".

  Stage 2 — server (no edge involved)
    Server projection MLP + binary match head trained with BCE.
    Receives cached latents [B, 64] (compression on) or
    cached fused embeddings [B, 512] (compression off).

DEBUG=True  uses GPT-2 as the server LLM (small, no GPU needed)
DEBUG=False uses the scenario set in ServerConfig
"""

import argparse
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
from emma.compression.vae import VAE, vae_loss
from emma.compression.autoencoder import AutoEncoder, distilae_loss, distvarae_loss, supcon_loss, cross_modal_infonce_loss
from emma.compression.pca import PCACompressor
from emma.compression.block_pca import BlockPCACompressor
from emma.compression.lda import LDACompressor

# ── CLI args (override defaults below) ───────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compression", choices=["none", "ae", "vae", "pca", "lda", "distilae", "distvarae", "contrastiveae", "blockpca", "crossmodalae"], default=None)
    p.add_argument("--latent-dim",  type=int,   default=None, help="compression bottleneck size (default: 64)")
    p.add_argument("--batch-size",  type=int,   default=None)
    p.add_argument("--epochs",      type=int,   default=None, help="server training epochs")
    p.add_argument("--comp-epochs", type=int,   default=None, help="compression training epochs")
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--no-debug",    action="store_true", help="use full LLM instead of GPT-2")
    p.add_argument("--n-train",     type=int,   default=None, help="number of training images (default: 5000)")
    p.add_argument("--n-valid",     type=int,   default=None, help="number of validation images (default: 500)")
    p.add_argument("--n-test",      type=int,   default=None, help="number of test images (default: 500)")
    p.add_argument("--seed",        type=int,   default=0,    help="random seed for training (default: 0)")
    # DistilAE variant flags (all optional; defaults reproduce the baseline DistilAE)
    p.add_argument("--lambda-distil",   type=float, default=0.1,
                   help="weight of the similarity distillation term (default: 0.1)")
    p.add_argument("--distil-mask-diag", action="store_true",
                   help="exclude the diagonal from the similarity MSE")
    p.add_argument("--distil-only",     action="store_true",
                   help="train compression on the distillation term only (no recon MSE)")
    p.add_argument("--comp-batch-size", type=int, default=None,
                   help="batch size for compression training (default: --batch-size)")
    p.add_argument("--lambda-con",    type=float, default=1.0,
                   help="weight of contrastive loss in ContrastiveAE (default: 1.0)")
    p.add_argument("--con-temp",      type=float, default=0.07,
                   help="temperature for SupCon loss (default: 0.07)")
    # Encoder / fusion variants (defaults reproduce the baseline pipeline)
    p.add_argument("--encoder", choices=["clip", "mobileclip"], default="clip",
                   help="edge encoder family (default: clip = ViT-B/32)")
    p.add_argument("--fusion",  choices=["mean", "concat", "match"], default="mean",
                   help="cross-modal fusion: mean (512), concat (1024), "
                        "match [t;v;|t-v|;t*v] (2048)")
    p.add_argument("--decode-on-server", action="store_true",
                   help="decode AE latents back to original dim before server training "
                        "(tests the full edge-encode → server-decode pipeline)")
    # Early stopping for server training
    p.add_argument("--patience",   type=int, default=10,
                   help="early-stopping patience in epochs (default: 10)")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="hard cap on server epochs; overrides --epochs if set")
    return p.parse_args()

_args = _parse_args()

# ── Hyperparameters ───────────────────────────────────────────────────────────

DEBUG          = not _args.no_debug
BATCH_SIZE     = _args.batch_size  or 64
LR             = _args.lr          or 1e-4
N_TRAIN        = _args.n_train     or 5_000
N_VALID        = _args.n_valid     or 500
N_TEST         = _args.n_test      or 500
SEED           = _args.seed
torch.manual_seed(SEED)
GRAD_CLIP      = 1.0
MAX_TEXT_LEN   = 77    # CLIP tokenizer max length
N_SOFT_TOKENS  = 8
CHECKPOINT_DIR = _args.checkpoint_dir or "checkpoints"
INSTRUCTION    = "Does the image match the description?"

# Encoder family and fusion mode (defaults = baseline: clip + mean)
ENCODER = _args.encoder
FUSION  = _args.fusion
D_FUSED = {"mean": 512, "concat": 1024, "match": 2048}[FUSION]
_enc    = "" if ENCODER == "clip" else f"_{ENCODER}"
_fus    = "" if FUSION  == "mean" else f"_{FUSION}"

# Embed cache depends on dataset size, encoder, and fusion mode
CACHE_DIR = os.path.join(
    CHECKPOINT_DIR, f"embed_cache{_enc}{_fus}_{N_TRAIN}_{N_VALID}_{N_TEST}")

# MobileCLIP embeddings (even unnormalized) have different scale from CLIP —
# standardize per-feature before AE training so the AE's initialization and lr
# are correct regardless of encoder family.
NORMALIZE_FUSED = False   # raw embeddings work better; PCA shows MC is already compact

# AE hidden dims scale with input dimension so each halving step is reasonable.
# For mean (512):  (512, 256, 128) — identical to the original baseline.
# For concat (1024): (1024, 512, 256, 128)
# For match (2048): (2048, 1024, 512, 256, 128)
def _hidden_for(d_in: int) -> tuple:
    dims, h = [], d_in
    while h > 128:
        dims.append(h); h //= 2
    dims.append(128)
    return tuple(dims)
AE_HIDDEN = _hidden_for(D_FUSED)

# Compression: "none" | "ae" | "vae" | "pca"
COMPRESSION    = _args.compression or "ae"
D_LATENT       = _args.latent_dim  or 64

# DistilAE variant settings (defaults leave behavior identical to baseline)
LAMBDA_DISTIL   = _args.lambda_distil
DISTIL_MASK_DIAG = _args.distil_mask_diag
DISTIL_ONLY     = _args.distil_only
COMP_BATCH_SIZE = _args.comp_batch_size or BATCH_SIZE
LAMBDA_CON      = _args.lambda_con
CON_TEMP        = _args.con_temp
DECODE_ON_SERVER = _args.decode_on_server

# Variant tag: empty for baseline settings, so existing checkpoint names are unchanged
_variant = ""
if LAMBDA_DISTIL != 0.1:
    _variant += f"_ld{LAMBDA_DISTIL:g}"
if DISTIL_MASK_DIAG:
    _variant += "_md"
if DISTIL_ONLY:
    _variant += "_do"
if _args.comp_batch_size:
    _variant += f"_cb{COMP_BATCH_SIZE}"

# Checkpoint tag includes compression ratio and seed so runs don't overwrite each other
# e.g. "ae_64", "ae_128_80k", "vae_32_80k_s3", "distilae_64_72k_md_cb512_s2",
#      "ae_64_72k_mobileclip_match_s1"
_scale   = f"_{N_TRAIN//1000}k" if N_TRAIN != 5_000 else ""
_seed    = f"_s{SEED}" if SEED != 0 else ""
_dos_tag = "_decoded" if DECODE_ON_SERVER else ""
CKPT_TAG = (f"{COMPRESSION}{_enc}{_fus}{_seed}" if COMPRESSION == "none"
            else f"{COMPRESSION}_{D_LATENT}{_scale}{_enc}{_fus}{_variant}{_dos_tag}{_seed}")

# Epochs per stage
STAGE_EPOCHS = {
    "vae":    _args.comp_epochs or 50,
    "server": _args.epochs      or 20,
}
PATIENCE   = _args.patience
MAX_EPOCHS = _args.max_epochs or (STAGE_EPOCHS["server"] * 10)  # generous hard cap

# ── Config ────────────────────────────────────────────────────────────────────

cfg = EMMAConfig(
    encoder=EncoderConfig(
        family=ENCODER,
        text_freeze_base=True,
        image_freeze_base=True,
    ),
    alignment=AlignmentConfig(d_shared=512, fusion=FUSION),  # per-modality dim; fused dim is D_FUSED
    vae=VAEConfig(d_latent=D_LATENT, encoder_hidden_dims=(512, 256, 128), beta=0.001),
    server=ServerConfig(
        scenario="plain_llm",
        n_soft_tokens=N_SOFT_TOKENS,
        freeze_llm=True,
    ),
    use_compression=False,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model builders ────────────────────────────────────────────────────────────

def build_edge(cfg: EMMAConfig) -> EdgePipeline:
    return EdgePipeline(config=cfg)


def build_server(cfg: EMMAConfig, debug: bool) -> tuple[ServerPipeline, AutoTokenizer]:
    sc = cfg.server
    ac = cfg.alignment

    if debug:
        llm       = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token  = tokenizer.eos_token
        d_llm = llm.config.hidden_size
    else:
        from emma.server.pipeline import _load_llm
        llm, d_llm = _load_llm(sc.scenario, sc.load_in_8bit)
        tokenizer  = AutoTokenizer.from_pretrained(
            {"llava":      "llava-hf/llava-1.5-7b-hf",
             "qwen_audio": "Qwen/Qwen2-Audio-7B-Instruct",
             "plain_llm":  "mistralai/Mistral-7B-v0.1"}[sc.scenario]
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Input dim to server projection depends on compression mode and decode-on-server flag.
    # When DECODE_ON_SERVER=True the AE decoder reconstructs back to D_FUSED before the server.
    if COMPRESSION == "none":
        d_server_in = D_FUSED
    elif DECODE_ON_SERVER:
        d_server_in = D_FUSED
    else:
        d_server_in = cfg.vae.d_latent

    server = ServerPipeline(
        llm=llm,
        d_llm=d_llm,
        projection=_build_projection(d_server_in, sc.n_soft_tokens, d_llm),
        match_head=nn.Linear(d_llm, 1),
        n_soft_tokens=sc.n_soft_tokens,
        freeze_llm=sc.freeze_llm,
        vae_decoder=None,   # decoding already done in compression stage
    )
    return server, tokenizer


# ── Stage configuration ───────────────────────────────────────────────────────

def freeze_all(edge: EdgePipeline, server: ServerPipeline):
    for p in list(edge.parameters()) + list(server.parameters()):
        p.requires_grad = False


def configure_edge_stage(edge: EdgePipeline, server: ServerPipeline) -> AdamW:
    """
    Prepare Stage 1: freeze everything, then unfreeze the CLIP projection
    layers (text_projection and visual_projection) for lightweight fine-tuning.
    The backbone transformers and the server stay frozen.
    Returns a fresh AdamW optimizer.
    """
    freeze_all(edge, server)

    for module in [edge.text_encoder.text_projection,
                   edge.image_encoder.visual_projection]:
        for p in module.parameters():
            p.requires_grad = True

    trainable = [p for p in edge.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"\n── Stage 'edge': CLIP projection layers ({n_params:,} params) ──")

    return AdamW(trainable, lr=LR, weight_decay=1e-2)


def configure_server_stage(server: ServerPipeline) -> AdamW:
    """
    Prepare Stage 2: freeze the entire server, then unfreeze only the
    projection MLP and match head.  The edge model is not involved.
    Returns a fresh AdamW optimizer over the active server parameters.
    """
    for p in server.parameters():
        p.requires_grad = False

    for module in [server.projection, server.match_head]:
        for p in module.parameters():
            p.requires_grad = True

    trainable = [p for p in server.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"\n── Stage 'server': training projection + match head ({n_params:,} params) ──")

    return AdamW(trainable, lr=LR, weight_decay=1e-2)


# ── Loss & metrics ────────────────────────────────────────────────────────────

def info_nce_loss(t: torch.Tensor, v: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style) between text and image projections.

    Positive pairs are (t_i, v_i); all other within-batch combinations
    are negatives.  Both inputs are L2-normalised before the dot product.

    Args:
        t           : [B, d_shared]  text projections
        v           : [B, d_shared]  image projections
        temperature : softmax temperature (lower = sharper)
    Returns:
        scalar loss
    """
    t = F.normalize(t, dim=-1)
    v = F.normalize(v, dim=-1)
    logits = (t @ v.T) / temperature          # [B, B]
    labels = torch.arange(len(t), device=t.device)
    loss_t = F.cross_entropy(logits,   labels)
    loss_v = F.cross_entropy(logits.T, labels)
    return (loss_t + loss_v) / 2


def contrastive_accuracy(t: torch.Tensor, v: torch.Tensor) -> float:
    """Fraction of samples where the correct pair has the highest similarity."""
    t = F.normalize(t, dim=-1)
    v = F.normalize(v, dim=-1)
    sims   = t @ v.T                          # [B, B]
    labels = torch.arange(len(t), device=t.device)
    acc_t  = (sims.argmax(dim=1) == labels).float().mean().item()
    acc_v  = (sims.argmax(dim=0) == labels).float().mean().item()
    return (acc_t + acc_v) / 2


def match_loss(preds: dict, batch: dict) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        preds["match"].squeeze(-1), batch["label"]
    )


def match_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return ((logits.squeeze(-1) > 0) == (labels > 0.5)).float().mean().item()


# ── Stage 1: contrastive edge training (no server) ───────────────────────────

def run_contrastive_epoch(edge, loader, optimizer, scheduler,
                          training: bool) -> dict:
    """
    Train/eval the edge pipeline with InfoNCE loss.
    No server involved — edge is fully self-contained here.
    """
    edge.train(training)

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            t_proj, v_proj = edge.encode_contrastive(
                text_inputs  = {"input_ids":      batch["input_ids"],
                                "attention_mask": batch["attention_mask"]},
                image_inputs = {"pixel_values":   batch["pixel_values"]},
            )

            loss = info_nce_loss(t_proj, v_proj)

            if training:
                optimizer.zero_grad()
                loss.backward()
                active = [p for p in edge.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(active, GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            total_acc  += contrastive_accuracy(t_proj.detach(), v_proj.detach())
            n_batches  += 1

    return {
        "loss":     total_loss / n_batches,
        "accuracy": total_acc  / n_batches,
    }


# ── Embedding cache ───────────────────────────────────────────────────────────

def cache_embeddings(edge: EdgePipeline, loaders: dict, cache_dir: str):
    """Run frozen edge pipeline over every split; save fused embeddings + labels.
    Also saves individual text and image embeddings for cross-modal compression."""
    os.makedirs(cache_dir, exist_ok=True)
    edge.eval()

    for split, loader in loaders.items():
        fused_list, label_list, text_list, img_list = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                t_emb, v_emb = edge.encode_contrastive(
                    text_inputs  = {"input_ids":      batch["input_ids"],
                                    "attention_mask": batch["attention_mask"]},
                    image_inputs = {"pixel_values":   batch["pixel_values"]},
                )
                fused = edge.alignment(t_emb, v_emb)
                fused_list.append(fused.cpu())
                label_list.append(batch["label"].cpu())
                text_list.append(t_emb.cpu())
                img_list.append(v_emb.cpu())

        torch.save(torch.cat(fused_list),  os.path.join(cache_dir, f"{split}_fused.pt"))
        torch.save(torch.cat(label_list),  os.path.join(cache_dir, f"{split}_label.pt"))
        torch.save(torch.cat(text_list),   os.path.join(cache_dir, f"{split}_text.pt"))
        torch.save(torch.cat(img_list),    os.path.join(cache_dir, f"{split}_img.pt"))
        n = sum(len(f) for f in fused_list)
        print(f"  cached {split}: {n} samples")

    print(f"  embeddings written to {cache_dir}/")


class CachedEmbeddingDataset(Dataset):
    def __init__(self, cache_dir: str, split: str):
        self.fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"),
                                  weights_only=True)
        self.labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"),
                                  weights_only=True)

    def __len__(self):
        return len(self.fused)

    def __getitem__(self, idx):
        return {"fused": self.fused[idx], "label": self.labels[idx]}


def get_cached_loaders(cache_dir: str, batch_size: int) -> dict:
    loaders = {}
    for split in ("train", "valid", "test"):
        ds = CachedEmbeddingDataset(cache_dir, split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0,
        )
    return loaders


# ── Server epoch (cached embeddings only) ────────────────────────────────────

def run_server_epoch(server, loader, optimizer, scheduler,
                     instr_ids, instr_mask, training: bool) -> dict:
    server.train(training)

    total_loss = 0.0
    all_logits, all_labels = [], []
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch        = {k: v.to(DEVICE) for k, v in batch.items()}
            B            = batch["label"].size(0)
            instr_ids_b  = instr_ids.expand(B, -1).to(DEVICE)
            instr_mask_b = instr_mask.expand(B, -1).to(DEVICE)

            preds = server(batch["fused"], instr_ids_b, instr_mask_b)
            loss  = match_loss(preds, batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                trainable = [p for p in server.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            all_logits.append(preds["match"].detach().cpu())
            all_labels.append(batch["label"].cpu())
            n_batches += 1

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {
        "loss":     total_loss / n_batches,
        "accuracy": match_accuracy(logits, labels),
    }


# ── Compression stage ────────────────────────────────────────────────────────

def train_ae_compression(cache_dir: str, cfg: EMMAConfig,
                         ae_ckpt: str, epochs: int = 50) -> str:
    """
    Train a plain autoencoder on cached CLIP embeddings.
    No KL term — pure MSE reconstruction loss.
    """
    ac = cfg.alignment
    vc = cfg.vae

    train_fused = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)
    valid_fused = torch.load(os.path.join(cache_dir, "valid_fused.pt"), weights_only=True)

    # Per-feature standardization for MobileCLIP: its pre-normalization features have
    # different scale/distribution than CLIP.  Standardizing each feature to mean=0
    # std=1 preserves relative differences between samples (better than L2-norm which
    # collapses informative high-variance features).  Stats from train applied to all splits.
    if NORMALIZE_FUSED:
        emb_mean = train_fused.mean(dim=0, keepdim=True)
        emb_std  = train_fused.std(dim=0, keepdim=True).clamp(min=1e-8)
        train_fused = (train_fused - emb_mean) / emb_std
        valid_fused = (valid_fused - emb_mean) / emb_std
        print(f"  Standardizing embeddings: mean_norm={emb_mean.norm():.3f}, "
              f"avg_std={emb_std.mean():.3f}")
    else:
        emb_mean = emb_std = None

    train_loader = DataLoader(torch.utils.data.TensorDataset(train_fused),
                              batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(torch.utils.data.TensorDataset(valid_fused),
                              batch_size=BATCH_SIZE)

    ae        = AutoEncoder(d_in=D_FUSED, d_latent=vc.d_latent,
                            hidden_dims=AE_HIDDEN).to(DEVICE)
    optimizer = AdamW(ae.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    best_val_loss = float("inf")
    print(f"\n── AutoEncoder compression training ({epochs} epochs, "
          f"d_in={D_FUSED}, hidden={AE_HIDDEN}) ──")

    for epoch in range(1, epochs + 1):
        ae.train()
        tl, nb = 0., 0
        for (x,) in train_loader:
            x     = x.to(DEVICE)
            recon = ae(x)
            loss  = nn.functional.mse_loss(recon, x)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step()
            tl += loss.item(); nb += 1

        ae.eval()
        vl, vb = 0., 0
        with torch.no_grad():
            for (x,) in valid_loader:
                x = x.to(DEVICE)
                vl += nn.functional.mse_loss(ae(x), x).item(); vb += 1

        # Cosine similarity
        with torch.no_grad():
            recon_v = ae(valid_fused.to(DEVICE))
            cos = nn.functional.cosine_similarity(
                valid_fused.to(DEVICE), recon_v).mean().item()

        print(f"  Epoch {epoch:03d}/{epochs} | "
              f"train MSE {tl/nb:.4f} | val MSE {vl/vb:.4f} | cos_sim {cos:.4f}")

        if vl / vb < best_val_loss:
            best_val_loss = vl / vb
            ae.save(ae_ckpt)

    print(f"  AE saved → {ae_ckpt}")

    # Cache latents (or decoded reconstructions if --decode-on-server)
    _dos = "_decoded" if DECODE_ON_SERVER else ""
    latent_cache_dir = os.path.join(cache_dir, f"ae_latents_{vc.d_latent}{_seed}{_dos}")
    os.makedirs(latent_cache_dir, exist_ok=True)
    ae = AutoEncoder.load(ae_ckpt).to(DEVICE)
    ae.eval()
    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        if emb_mean is not None:
            fused = (fused - emb_mean) / emb_std
        with torch.no_grad():
            z   = ae.encoder(fused.to(DEVICE))
            out = ae.decoder(z).cpu() if DECODE_ON_SERVER else z.cpu()
        torch.save(out,    os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels, os.path.join(latent_cache_dir, f"{split}_label.pt"))
        mode = "decoded" if DECODE_ON_SERVER else "latent"
        print(f"  cached {split} ae {mode}: {out.shape}")

    return latent_cache_dir


def train_distilae_compression(cache_dir: str, cfg: EMMAConfig,
                               ae_ckpt: str, epochs: int = 50,
                               use_var: bool = False) -> str:
    """
    Train DistilAE (or DistilVarAE if use_var=True).
    Loss = MSE + λ_distil * ||S_latent - S_original||²  [+ λ_var * -var(latents)]
    """
    ac = cfg.alignment
    vc = cfg.vae

    train_fused = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)
    valid_fused = torch.load(os.path.join(cache_dir, "valid_fused.pt"), weights_only=True)

    # Per-feature standardization (same reasoning as train_ae_compression)
    if NORMALIZE_FUSED:
        emb_mean = train_fused.mean(dim=0, keepdim=True)
        emb_std  = train_fused.std(dim=0, keepdim=True).clamp(min=1e-8)
        train_fused = (train_fused - emb_mean) / emb_std
        valid_fused = (valid_fused - emb_mean) / emb_std
        print(f"  Standardizing embeddings: mean_norm={emb_mean.norm():.3f}, "
              f"avg_std={emb_std.mean():.3f}")
    else:
        emb_mean = emb_std = None

    train_loader = DataLoader(torch.utils.data.TensorDataset(train_fused),
                              batch_size=COMP_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(torch.utils.data.TensorDataset(valid_fused),
                              batch_size=COMP_BATCH_SIZE)

    ae        = AutoEncoder(d_in=D_FUSED, d_latent=vc.d_latent,
                            hidden_dims=AE_HIDDEN).to(DEVICE)
    optimizer = AdamW(ae.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    tag = "DistilVarAE" if use_var else "DistilAE"
    best_val_loss = float("inf")
    print(f"\n── {tag} compression training ({epochs} epochs, "
          f"d_in={D_FUSED}, hidden={AE_HIDDEN}, "
          f"batch={COMP_BATCH_SIZE}, λ_distil={LAMBDA_DISTIL}) ──")

    for epoch in range(1, epochs + 1):
        ae.train()
        tl, trl, tdl, tvl_sum, nb = 0., 0., 0., 0., 0
        for (x,) in train_loader:
            x       = x.to(DEVICE)
            latents = ae.encoder(x)
            recon   = ae.decoder(latents)
            if use_var:
                losses = distvarae_loss(recon, x, latents)
            else:
                losses = distilae_loss(recon, x, latents,
                                       lambda_distil=LAMBDA_DISTIL,
                                       mask_diag=DISTIL_MASK_DIAG,
                                       distil_only=DISTIL_ONLY)
            loss = losses["loss"]
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step()
            tl += loss.item(); trl += losses["recon_loss"].item()
            tdl += losses["distil_loss"].item(); nb += 1

        ae.eval()
        vl, vb = 0., 0
        with torch.no_grad():
            for (x,) in valid_loader:
                x       = x.to(DEVICE)
                latents = ae.encoder(x)
                recon   = ae.decoder(latents)
                losses  = distvarae_loss(recon, x, latents) if use_var \
                          else distilae_loss(recon, x, latents,
                                             lambda_distil=LAMBDA_DISTIL,
                                             mask_diag=DISTIL_MASK_DIAG,
                                             distil_only=DISTIL_ONLY)
                vl += losses["loss"].item(); vb += 1

        with torch.no_grad():
            recon_v = ae(valid_fused.to(DEVICE))
            cos = nn.functional.cosine_similarity(
                valid_fused.to(DEVICE), recon_v).mean().item()

        print(f"  Epoch {epoch:03d}/{epochs} | loss {tl/nb:.4f} | "
              f"recon {trl/nb:.4f} | distil {tdl/nb:.4f} | cos_sim {cos:.4f}")

        if vl / vb < best_val_loss:
            best_val_loss = vl / vb
            ae.save(ae_ckpt)

    print(f"  {tag} saved → {ae_ckpt}")

    # Cache latents
    tag_dir = f"distvarae_latents_{vc.d_latent}{_seed}" if use_var \
              else f"distilae_latents_{vc.d_latent}{_variant}{_seed}"
    latent_cache_dir = os.path.join(cache_dir, tag_dir)
    os.makedirs(latent_cache_dir, exist_ok=True)
    ae = AutoEncoder.load(ae_ckpt).to(DEVICE)
    ae.eval()
    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        if emb_mean is not None:
            fused = (fused - emb_mean) / emb_std
        with torch.no_grad():
            latents = ae.encoder(fused.to(DEVICE)).cpu()
        torch.save(latents, os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels,  os.path.join(latent_cache_dir, f"{split}_label.pt"))
        print(f"  cached {split} latents: {latents.shape}")

    return latent_cache_dir


def train_vae_compression(cache_dir: str, cfg: EMMAConfig,
                          vae_ckpt: str, epochs: int = 50) -> str:
    """
    Train a β-VAE on cached fused embeddings [B, 256] → latent [B, 64].
    Saves the trained VAE checkpoint and re-caches latents for the server.

    Args:
        cache_dir : directory with {split}_fused.pt files
        cfg       : EMMAConfig (uses cfg.vae for dimensions and beta)
        vae_ckpt  : path to save the trained VAE
        epochs    : number of training epochs

    Returns:
        latent_cache_dir — directory containing {split}_latent.pt files
    """
    vc  = cfg.vae
    ac  = cfg.alignment

    # Load cached fused embeddings
    train_fused = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)
    valid_fused = torch.load(os.path.join(cache_dir, "valid_fused.pt"), weights_only=True)

    train_ds = torch.utils.data.TensorDataset(train_fused)
    valid_ds = torch.utils.data.TensorDataset(valid_fused)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

    vae       = VAE(d_in=D_FUSED, d_latent=vc.d_latent,
                    encoder_hidden_dims=vc.encoder_hidden_dims,
                    decoder_hidden_dims=tuple(reversed(vc.encoder_hidden_dims))).to(DEVICE)
    optimizer = AdamW(vae.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    best_val_loss = float("inf")
    # β warmup: start at 0, linearly ramp to vc.beta over first 20 epochs
    WARMUP_EPOCHS = min(20, epochs // 2)
    FREE_BITS     = 0.5   # nats per latent dim, prevents posterior collapse
    print(f"\n── VAE compression training ({epochs} epochs, β={vc.beta}, "
          f"warmup={WARMUP_EPOCHS} epochs, free_bits={FREE_BITS}) ──")

    for epoch in range(1, epochs + 1):
        beta_now = vc.beta * min(1.0, (epoch - 1) / max(WARMUP_EPOCHS, 1))

        vae.train()
        tl, nb = 0., 0
        for (x,) in train_loader:
            x = x.to(DEVICE)
            recon, mu, log_var = vae(x)
            losses = vae_loss(recon, x, mu, log_var, beta=beta_now,
                              free_bits=FREE_BITS)
            optimizer.zero_grad(); losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step()
            tl += losses["loss"].item(); nb += 1

        vae.eval()
        vl, vr, vk, vb = 0., 0., 0., 0
        with torch.no_grad():
            for (x,) in valid_loader:
                x = x.to(DEVICE)
                recon, mu, log_var = vae(x)
                losses = vae_loss(recon, x, mu, log_var, beta=beta_now,
                                  free_bits=FREE_BITS)
                vl += losses["loss"].item()
                vr += losses["recon_loss"].item()
                vk += losses["kl_loss"].item()
                vb += 1

        print(f"  Epoch {epoch:03d}/{epochs} | β={beta_now:.3f} | "
              f"train loss {tl/nb:.4f} | "
              f"val loss {vl/vb:.4f}  recon {vr/vb:.4f}  kl {vk/vb:.4f}")

        if vl / vb < best_val_loss:
            best_val_loss = vl / vb
            vae.save(vae_ckpt)

    print(f"  VAE saved → {vae_ckpt}")

    # Re-cache latents (or decoded reconstructions if --decode-on-server)
    _dos = "_decoded" if DECODE_ON_SERVER else ""
    latent_cache_dir = os.path.join(cache_dir, f"vae_latents_{vc.d_latent}{_seed}{_dos}")
    os.makedirs(latent_cache_dir, exist_ok=True)
    vae = VAE.load(vae_ckpt).to(DEVICE)
    vae.eval()

    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        with torch.no_grad():
            z   = vae.encoder.encode(fused.to(DEVICE))
            out = vae.decoder(z).cpu() if DECODE_ON_SERVER else z.cpu()
        torch.save(out,    os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels, os.path.join(latent_cache_dir, f"{split}_label.pt"))
        mode = "decoded" if DECODE_ON_SERVER else "latent"
        print(f"  cached {split} vae {mode}: {out.shape}")

    return latent_cache_dir


def apply_pca_compression(cache_dir: str, cfg: EMMAConfig,
                          pca_ckpt: str) -> str:
    """
    Fit PCA on training fused embeddings and project all splits.
    No gradient training — closed-form solution.

    Returns:
        latent_cache_dir — directory containing {split}_fused.pt files
    """
    vc = cfg.vae

    train_fused = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)

    print(f"\n── PCA compression (n_components={vc.d_latent}) ──")
    pca = PCACompressor(n_components=vc.d_latent)
    pca.fit(train_fused)
    pca.save(pca_ckpt)

    latent_cache_dir = os.path.join(cache_dir, f"pca_latents_{vc.d_latent}{_seed}")
    os.makedirs(latent_cache_dir, exist_ok=True)

    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        latents = pca.encode(fused)
        torch.save(latents, os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels,  os.path.join(latent_cache_dir, f"{split}_label.pt"))
        print(f"  cached {split} PCA latents: {latents.shape}")

    return latent_cache_dir


def train_contrastiveae_compression(cache_dir: str, cfg: EMMAConfig,
                                    ae_ckpt: str, epochs: int = 50) -> str:
    """
    Train ContrastiveAE: MSE reconstruction + SupCon loss on latents.

    The encoder is forced to cluster matched pairs together and non-matched
    apart in 64-dim latent space — using labels only during training.
    At deployment: edge runs encoder only, server runs decoder only (no labels).

    Loss = MSE(recon, x) + λ_con * SupCon(latents, labels)
    """
    vc = cfg.vae

    train_fused  = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)
    train_labels = torch.load(os.path.join(cache_dir, "train_label.pt"), weights_only=True)
    valid_fused  = torch.load(os.path.join(cache_dir, "valid_fused.pt"), weights_only=True)
    valid_labels = torch.load(os.path.join(cache_dir, "valid_label.pt"), weights_only=True)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_fused, train_labels),
        batch_size=COMP_BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(
        torch.utils.data.TensorDataset(valid_fused, valid_labels),
        batch_size=COMP_BATCH_SIZE, drop_last=False)

    ae        = AutoEncoder(d_in=D_FUSED, d_latent=vc.d_latent,
                            hidden_dims=AE_HIDDEN).to(DEVICE)
    optimizer = AdamW(ae.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    print(f"\n── ContrastiveAE compression training ({epochs} epochs, "
          f"d_in={D_FUSED}, hidden={AE_HIDDEN}, "
          f"batch={COMP_BATCH_SIZE}, λ_con={LAMBDA_CON}, τ={CON_TEMP}) ──")

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        ae.train()
        tl, trl, tcl, nb = 0., 0., 0., 0
        for x, y in train_loader:
            x, y    = x.to(DEVICE), y.to(DEVICE)
            latents = ae.encoder(x)
            recon   = ae.decoder(latents)
            recon_l = F.mse_loss(recon, x)
            con_l   = supcon_loss(latents, y, temperature=CON_TEMP)
            loss    = recon_l + LAMBDA_CON * con_l
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step()
            tl += loss.item(); trl += recon_l.item()
            tcl += con_l.item(); nb += 1

        ae.eval()
        vl, vb = 0., 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y    = x.to(DEVICE), y.to(DEVICE)
                latents = ae.encoder(x)
                recon   = ae.decoder(latents)
                recon_l = F.mse_loss(recon, x)
                con_l   = supcon_loss(latents, y, temperature=CON_TEMP)
                vl += (recon_l + LAMBDA_CON * con_l).item(); vb += 1

        print(f"  Epoch {epoch:03d}/{epochs} | loss {tl/nb:.4f} | "
              f"recon {trl/nb:.4f} | contrastive {tcl/nb:.4f} | "
              f"val {vl/vb:.4f}")

        if vl / vb < best_val_loss:
            best_val_loss = vl / vb
            ae.save(ae_ckpt)

    print(f"  ContrastiveAE saved → {ae_ckpt}")

    # Cache latents (or decoded reconstructions if --decode-on-server)
    _dos     = "_decoded" if DECODE_ON_SERVER else ""
    latent_cache_dir = os.path.join(cache_dir, f"contrastiveae_latents_{vc.d_latent}{_seed}{_dos}")
    os.makedirs(latent_cache_dir, exist_ok=True)
    ae = AutoEncoder.load(ae_ckpt).to(DEVICE)
    ae.eval()
    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        with torch.no_grad():
            z = ae.encoder(fused.to(DEVICE))
            out = ae.decoder(z).cpu() if DECODE_ON_SERVER else z.cpu()
        torch.save(out,    os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels, os.path.join(latent_cache_dir, f"{split}_label.pt"))
        mode = "decoded" if DECODE_ON_SERVER else "latent"
        print(f"  cached {split} contrastiveae {mode}: {out.shape}")

    return latent_cache_dir


def apply_lda_compression(cache_dir: str, cfg: EMMAConfig,
                          lda_ckpt: str) -> str:
    """
    Fit LDA-PCA hybrid on training fused embeddings + labels, project all splits.
    Closed-form — no gradient training needed.
    The LDA direction captures the most class-discriminative signal first;
    the remaining components are PCA in the orthogonal complement.

    Returns:
        latent_cache_dir — directory containing {split}_fused.pt files
    """
    vc = cfg.vae

    train_fused  = torch.load(os.path.join(cache_dir, "train_fused.pt"),  weights_only=True)
    train_labels = torch.load(os.path.join(cache_dir, "train_label.pt"),  weights_only=True)

    print(f"\n── LDA-PCA compression (n_components={vc.d_latent}) ──")
    lda = LDACompressor(n_components=vc.d_latent)
    lda.fit(train_fused, train_labels)
    lda.save(lda_ckpt)

    latent_cache_dir = os.path.join(cache_dir, f"lda_latents_{vc.d_latent}{_seed}")
    os.makedirs(latent_cache_dir, exist_ok=True)

    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"),  weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"),  weights_only=True)
        latents = lda.encode(fused)
        torch.save(latents, os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels,  os.path.join(latent_cache_dir, f"{split}_label.pt"))
        print(f"  cached {split} LDA latents: {latents.shape}")

    return latent_cache_dir


def apply_block_pca_compression(cache_dir: str, cfg: EMMAConfig,
                                blockpca_ckpt: str) -> str:
    """
    Fit BlockPCA on training match-fused embeddings [B, 2048] and project all splits.
    Each of the 4 structural blocks (t, v, |t-v|, t⊙v) gets n_components//4 PCA dims.
    Closed-form — no gradient training needed.

    Only valid with match fusion (D_FUSED == 2048, d_block == 512).
    """
    vc = cfg.vae

    if FUSION != "match":
        raise ValueError("BlockPCA is only supported with match fusion "
                         "(fusion='match'). Got fusion='{FUSION}'.")

    train_fused = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)

    print(f"\n── BlockPCA compression (n_components={vc.d_latent}, 4 blocks of "
          f"{vc.d_latent//4} each) ──")
    bpca = BlockPCACompressor(n_components=vc.d_latent, d_block=512)
    bpca.fit(train_fused)
    bpca.save(blockpca_ckpt)

    latent_cache_dir = os.path.join(cache_dir, f"blockpca_latents_{vc.d_latent}{_seed}")
    os.makedirs(latent_cache_dir, exist_ok=True)

    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        latents = bpca.encode(fused)
        torch.save(latents, os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels,  os.path.join(latent_cache_dir, f"{split}_label.pt"))
        print(f"  cached {split} BlockPCA latents: {latents.shape}")

    return latent_cache_dir


def train_crossmodalae_compression(cache_dir: str, cfg: EMMAConfig,
                                   ae_ckpt: str, epochs: int = 50) -> str:
    """
    Train CrossModalAE: MSE reconstruction + cross-modal InfoNCE loss.

    Uses the natural image-text pairing as free supervision (no binary match labels).
    InfoNCE encourages the 64-dim latent to be a sufficient statistic for
    cross-modal alignment — the exact signal that determines binary matching.

    Loss = MSE(recon, fused) + λ_con * InfoNCE(z, text_emb, img_emb)
    """
    vc = cfg.vae

    train_fused  = torch.load(os.path.join(cache_dir, "train_fused.pt"),  weights_only=True)
    train_text   = torch.load(os.path.join(cache_dir, "train_text.pt"),   weights_only=True)
    train_img    = torch.load(os.path.join(cache_dir, "train_img.pt"),    weights_only=True)
    valid_fused  = torch.load(os.path.join(cache_dir, "valid_fused.pt"),  weights_only=True)
    valid_text   = torch.load(os.path.join(cache_dir, "valid_text.pt"),   weights_only=True)
    valid_img    = torch.load(os.path.join(cache_dir, "valid_img.pt"),    weights_only=True)
    valid_labels = torch.load(os.path.join(cache_dir, "valid_label.pt"),  weights_only=True)

    D_ENC = train_text.shape[1]   # 512 for both CLIP and MobileCLIP

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_fused, train_text, train_img),
        batch_size=COMP_BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(
        torch.utils.data.TensorDataset(valid_fused, valid_text, valid_img, valid_labels),
        batch_size=COMP_BATCH_SIZE, drop_last=False)

    ae        = AutoEncoder(d_in=D_FUSED, d_latent=vc.d_latent,
                            hidden_dims=AE_HIDDEN).to(DEVICE)
    # Learnable linear projection heads: map encoder dim → latent dim for InfoNCE
    text_proj = nn.Linear(D_ENC, vc.d_latent, bias=False).to(DEVICE)
    img_proj  = nn.Linear(D_ENC, vc.d_latent, bias=False).to(DEVICE)

    optimizer = AdamW(list(ae.parameters()) +
                      list(text_proj.parameters()) +
                      list(img_proj.parameters()),
                      lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    print(f"\n── CrossModalAE compression training ({epochs} epochs, "
          f"d_in={D_FUSED}, hidden={AE_HIDDEN}, "
          f"batch={COMP_BATCH_SIZE}, λ_con={LAMBDA_CON}, τ={CON_TEMP}) ──")

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        ae.train(); text_proj.train(); img_proj.train()
        tl, trl, tcl, nb = 0., 0., 0., 0
        for x, t_emb, v_emb in train_loader:
            x, t_emb, v_emb = x.to(DEVICE), t_emb.to(DEVICE), v_emb.to(DEVICE)
            latents = ae.encoder(x)
            recon   = ae.decoder(latents)
            recon_l = F.mse_loss(recon, x)
            con_l   = cross_modal_infonce_loss(
                latents, t_emb, v_emb, text_proj, img_proj, temperature=CON_TEMP)
            loss = recon_l + LAMBDA_CON * con_l
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ae.parameters()) + list(text_proj.parameters()) +
                list(img_proj.parameters()), GRAD_CLIP)
            optimizer.step(); scheduler.step()
            tl += loss.item(); trl += recon_l.item()
            tcl += con_l.item(); nb += 1

        ae.eval(); text_proj.eval(); img_proj.eval()
        vl, vb = 0., 0
        with torch.no_grad():
            for x, t_emb, v_emb, _ in valid_loader:
                x, t_emb, v_emb = x.to(DEVICE), t_emb.to(DEVICE), v_emb.to(DEVICE)
                latents = ae.encoder(x)
                recon   = ae.decoder(latents)
                recon_l = F.mse_loss(recon, x)
                con_l   = cross_modal_infonce_loss(
                    latents, t_emb, v_emb, text_proj, img_proj, temperature=CON_TEMP)
                vl += (recon_l + LAMBDA_CON * con_l).item(); vb += 1

        print(f"  Epoch {epoch:03d}/{epochs} | loss {tl/nb:.4f} | "
              f"recon {trl/nb:.4f} | infonce {tcl/nb:.4f} | val {vl/vb:.4f}")

        if vl / vb < best_val_loss:
            best_val_loss = vl / vb
            ae.save(ae_ckpt)

    print(f"  CrossModalAE saved → {ae_ckpt}")

    # Cache latents
    _dos = "_decoded" if DECODE_ON_SERVER else ""
    latent_cache_dir = os.path.join(cache_dir, f"crossmodalae_latents_{vc.d_latent}{_seed}{_dos}")
    os.makedirs(latent_cache_dir, exist_ok=True)
    ae = AutoEncoder.load(ae_ckpt).to(DEVICE)
    ae.eval()
    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        with torch.no_grad():
            z   = ae.encoder(fused.to(DEVICE))
            out = ae.decoder(z).cpu() if DECODE_ON_SERVER else z.cpu()
        torch.save(out,    os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels, os.path.join(latent_cache_dir, f"{split}_label.pt"))
        mode = "decoded" if DECODE_ON_SERVER else "latent"
        print(f"  cached {split} crossmodalae {mode}: {out.shape}")

    return latent_cache_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nBuilding models (debug={DEBUG})...")
    server, tokenizer = build_server(cfg, debug=DEBUG)
    server            = server.to(DEVICE)

    instr_enc  = tokenizer(INSTRUCTION, return_tensors="pt",
                           padding=True, truncation=True, max_length=32)
    instr_ids  = instr_enc["input_ids"]
    instr_mask = instr_enc["attention_mask"]

    # ── Cache fused CLIP embeddings (run once, reuse across experiments) ───────
    # Raw image/text data is only loaded when the embed cache must be built;
    # otherwise skip it entirely (loading ~12GB of pickles is slow, and many
    # concurrent jobs reading the same files causes I/O-contention timeouts).
    embed_cache = os.path.join(CACHE_DIR, "train_fused.pt")
    if os.path.exists(embed_cache):
        print(f"\nFound embedding cache at {CACHE_DIR} — skipping raw data loading and encoding.")
    else:
        print("Loading data...")
        loaders = get_loaders(batch_size=BATCH_SIZE, max_text_len=MAX_TEXT_LEN, num_workers=0,
                              n_train=N_TRAIN, n_valid=N_VALID, n_test=N_TEST,
                              encoder=ENCODER)
        print(f"  train={len(loaders['train'].dataset)}  "
              f"valid={len(loaders['valid'].dataset)}  "
              f"test={len(loaders['test'].dataset)}")

        print("\nEncoding with frozen CLIP...")
        edge = build_edge(cfg).to(DEVICE)
        cache_embeddings(edge, loaders, CACHE_DIR)
        del edge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compression stage (optional) ──────────────────────────────────────────
    if COMPRESSION == "ae":
        server_cache = train_ae_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            ae_ckpt=os.path.join(CHECKPOINT_DIR, f"ae_{D_LATENT}{_enc}{_fus}{_seed}.pt"),
            epochs=STAGE_EPOCHS["vae"],
        )
    elif COMPRESSION == "vae":
        server_cache = train_vae_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            vae_ckpt=os.path.join(CHECKPOINT_DIR, f"vae_{D_LATENT}{_enc}{_fus}{_seed}.pt"),
            epochs=STAGE_EPOCHS["vae"],
        )
    elif COMPRESSION == "pca":
        server_cache = apply_pca_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            pca_ckpt=os.path.join(CHECKPOINT_DIR, f"pca_{D_LATENT}{_seed}"),
        )
    elif COMPRESSION == "contrastiveae":
        server_cache = train_contrastiveae_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            ae_ckpt=os.path.join(CHECKPOINT_DIR, f"contrastiveae_{D_LATENT}{_enc}{_fus}{_seed}.pt"),
            epochs=STAGE_EPOCHS["vae"],
        )
    elif COMPRESSION == "blockpca":
        server_cache = apply_block_pca_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            blockpca_ckpt=os.path.join(CHECKPOINT_DIR, f"blockpca_{D_LATENT}{_enc}{_fus}{_seed}"),
        )
    elif COMPRESSION == "crossmodalae":
        server_cache = train_crossmodalae_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            ae_ckpt=os.path.join(CHECKPOINT_DIR, f"crossmodalae_{D_LATENT}{_enc}{_fus}{_seed}.pt"),
            epochs=STAGE_EPOCHS["vae"],
        )
    elif COMPRESSION == "lda":
        server_cache = apply_lda_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            lda_ckpt=os.path.join(CHECKPOINT_DIR, f"lda_{D_LATENT}{_seed}"),
        )
    elif COMPRESSION == "distilae":
        server_cache = train_distilae_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            ae_ckpt=os.path.join(CHECKPOINT_DIR, f"distilae_{D_LATENT}{_variant}{_seed}.pt"),
            epochs=STAGE_EPOCHS["vae"],
            use_var=False,
        )
    elif COMPRESSION == "distvarae":
        server_cache = train_distilae_compression(
            cache_dir=CACHE_DIR,
            cfg=cfg,
            ae_ckpt=os.path.join(CHECKPOINT_DIR, f"distvarae_{D_LATENT}{_seed}.pt"),
            epochs=STAGE_EPOCHS["vae"],
            use_var=True,
        )
    else:
        server_cache = CACHE_DIR   # pass fused embeddings directly

    cached_loaders = get_cached_loaders(server_cache, batch_size=BATCH_SIZE)

    # ── Server stage ──────────────────────────────────────────────────────────
    optimizer = configure_server_stage(server)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=MAX_EPOCHS * len(cached_loaders["train"]))
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_m = run_server_epoch(server, cached_loaders["train"], optimizer, scheduler,
                                   instr_ids, instr_mask, training=True)
        val_m   = run_server_epoch(server, cached_loaders["valid"], optimizer, scheduler,
                                   instr_ids, instr_mask, training=False)

        print(f"  [server] Epoch {epoch:02d}/{MAX_EPOCHS} | "
              f"train loss {train_m['loss']:.4f} acc {train_m['accuracy']:.3f} | "
              f"val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.3f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            no_improve    = 0
            torch.save({
                "stage":       "server",
                "epoch":       epoch,
                "server":      server.state_dict(),
                "val_loss":    best_val_loss,
                "val_metrics": val_m,
            }, os.path.join(CHECKPOINT_DIR, f"best_server_{CKPT_TAG}.pt"))
            print(f"    ✓ saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f"  [server] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # ── Final test ────────────────────────────────────────────────────────────
    print("\nFinal test evaluation...")
    test_m = run_server_epoch(server, cached_loaders["test"], optimizer, scheduler,
                              instr_ids, instr_mask, training=False)
    print(f"Test | loss {test_m['loss']:.4f} | accuracy {test_m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
