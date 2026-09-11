"""
EMMI training script — full MLLM server scenario (LLaVA-1.5-7B backbone).

Standalone from train.py by design: the only thing this script changes is
the server LLM — LLaVA-1.5-7B's frozen language_model backbone (loaded via
emma.server.pipeline._load_llm, scenario="llava") instead of GPT-2. It does
not import or modify train.py; it duplicates the small amount of shared
plumbing (embed caching, cached-loader dataset, server epoch loop) so the
two scripts can evolve independently.

Compression is either off (--compression none) or one of the paper's 7
compressors (pca, ae, vae, lda, blockpca, contrastiveae, crossmodalae).
For ae/vae/blockpca/contrastiveae/crossmodalae, an already-trained
checkpoint is loaded from disk and only used to encode fused embeddings
into latents -- never retrained here, since compression is decoupled from
the server LLM by design (it only depends on encoder/fusion/latent-dim),
so reusing the GPT-2 runs' compressor checkpoint isolates the GPT-2 ->
LLaVA swap as the only variable. pca/lda are the exception: train.py's own
checkpoint naming for those two doesn't include encoder/fusion, so an
existing pca_64.npz/lda_64.npz can't be trusted to match this run's
encoder/fusion -- both are refit here from the cached fused embeddings
instead (cheap, closed-form, no gradient training either way). Edge
encoders are frozen; only the server projection MLP + match head train.

Reuses the same embed-cache directory naming as train.py
(checkpoints/embed_cache_<enc>_<fusion>_<n_train>_<n_valid>_<n_test>), so
if that cache already exists from a train.py run it's picked up directly
and no data re-encoding happens.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from emma.config import EMMAConfig, EncoderConfig, AlignmentConfig, VAEConfig, ServerConfig
from emma.data import get_loaders
from emma.pipeline import EdgePipeline
from emma.server.pipeline import ServerPipeline, _build_projection, _load_llm

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compression",
                   choices=["none", "pca", "ae", "vae", "lda", "blockpca",
                            "contrastiveae", "crossmodalae"],
                   default="none",
                   help="'none' = raw fused embeddings. 'pca'/'lda' are refit here (closed-form, "
                        "cheap). 'ae'/'vae'/'blockpca'/'contrastiveae'/'crossmodalae' load an "
                        "already-trained checkpoint and apply it (never retrained here).")
    p.add_argument("--latent-dim",  type=int,   default=64,
                   help="compressor latent dim (default: 64, matches the paper's headline config)")
    p.add_argument("--compression-checkpoint", default=None,
                   help="override path to the compressor checkpoint (default: "
                        "checkpoints/<compression>_<latent-dim>_<encoder>_<fusion>[_s<seed>].pt, "
                        "matching train.py's naming convention)")
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=30, help="server training epochs (soft cap)")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--n-train",     type=int,   default=72_000)
    p.add_argument("--n-valid",     type=int,   default=5_000)
    p.add_argument("--n-test",      type=int,   default=5_000)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--encoder", choices=["clip", "mobileclip"], default="mobileclip",
                   help="edge encoder family (default: mobileclip, matches the paper's headline config)")
    p.add_argument("--fusion",  choices=["mean", "concat", "match"], default="match",
                   help="cross-modal fusion (default: match [t;v;|t-v|;t*v], 2048-dim)")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="load LLaVA's language_model backbone with bitsandbytes 8-bit quantization")
    p.add_argument("--llm-dtype", choices=["float16", "float32"], default="float16",
                   help="precision for the LLaVA backbone when --load-in-8bit is not set (default: "
                        "float16, matches all prior runs). float32 removes fp16's ~65504 dynamic-range "
                        "ceiling -- diagnostic option for the non-finite-gradient instability seen on "
                        "some (compressor, encoder) pairs. Ignored when --load-in-8bit is set.")
    p.add_argument("--standardize-input", action="store_true",
                   help="standardize the projection's input (zero mean, unit variance per "
                        "feature, stats from train split) before it reaches LLaVA's fp16 "
                        "backbone. Opt-in and off by default -- existing runs are unaffected "
                        "unless this flag is passed. See standardize_cache() docstring.")
    p.add_argument("--patience",   type=int, default=10, help="early-stopping patience in epochs")
    p.add_argument("--max-epochs", type=int, default=None, help="hard cap on server epochs")
    p.add_argument("--debug-forward", action="store_true",
                   help="print min/max/isnan/isinf at each stage of ServerPipeline.forward() "
                        "for the first few batches, to localize where a non-finite value first "
                        "appears. Opt-in diagnostic only -- no effect on training behavior.")
    p.add_argument("--debug-every", type=int, default=0,
                   help="with --debug-forward, also print a lightweight projection-weight-norm/"
                        "soft-token-magnitude snapshot every N forward calls (0 = off), to see "
                        "whether the projection drifts over training rather than only checking "
                        "the first few batches.")
    return p.parse_args()

_args = _parse_args()

# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH_SIZE     = _args.batch_size
LR             = _args.lr
N_TRAIN        = _args.n_train
N_VALID        = _args.n_valid
N_TEST         = _args.n_test
SEED           = _args.seed
torch.manual_seed(SEED)
GRAD_CLIP      = 1.0
MAX_TEXT_LEN   = 77
N_SOFT_TOKENS  = 8
CHECKPOINT_DIR = _args.checkpoint_dir
INSTRUCTION    = "Does the image match the description?"
LOAD_IN_8BIT   = _args.load_in_8bit
LLM_DTYPE      = _args.llm_dtype
STANDARDIZE_INPUT = _args.standardize_input
DEBUG_FORWARD  = _args.debug_forward
DEBUG_EVERY    = _args.debug_every

ENCODER = _args.encoder
FUSION  = _args.fusion
D_FUSED = {"mean": 512, "concat": 1024, "match": 2048}[FUSION]
_enc    = "" if ENCODER == "clip" else f"_{ENCODER}"
_fus    = "" if FUSION  == "mean" else f"_{FUSION}"
_seed   = f"_s{SEED}" if SEED != 0 else ""

COMPRESSION = _args.compression
D_LATENT    = _args.latent_dim
# d_server_in: what the projection MLP actually consumes -- raw fused
# embeddings for "none", or the compressor's latent dim otherwise.
D_SERVER_IN = D_FUSED if COMPRESSION == "none" else D_LATENT

# Same naming convention as train.py so an existing cache is reused directly.
CACHE_DIR = os.path.join(
    CHECKPOINT_DIR, f"embed_cache{_enc}{_fus}_{N_TRAIN}_{N_VALID}_{N_TEST}")

# Matches train.py's compressor-checkpoint naming (scale-independent -- the
# same compressor works regardless of how many samples the server trains on).
# blockpca saves as a directory (no extension); ae/vae/contrastiveae/crossmodalae
# save as .pt; pca/lda are refit fresh below and don't use this at all.
_ckpt_ext = "" if COMPRESSION == "blockpca" else ".pt"
COMPRESSION_CHECKPOINT = _args.compression_checkpoint or os.path.join(
    CHECKPOINT_DIR, f"{COMPRESSION}_{D_LATENT}{_enc}{_fus}{_seed}{_ckpt_ext}")

_std       = "_std" if STANDARDIZE_INPUT else ""
_dtype_tag = "" if (LOAD_IN_8BIT or LLM_DTYPE == "float16") else f"_{LLM_DTYPE}"
CKPT_TAG   = (f"llava{_enc}{_fus}{_seed}{_std}{_dtype_tag}" if COMPRESSION == "none"
              else f"llava_{COMPRESSION}_{D_LATENT}{_enc}{_fus}{_seed}{_std}{_dtype_tag}")
PATIENCE   = _args.patience
MAX_EPOCHS = _args.max_epochs or (_args.epochs * 10)

cfg = EMMAConfig(
    encoder=EncoderConfig(family=ENCODER, text_freeze_base=True, image_freeze_base=True),
    alignment=AlignmentConfig(d_shared=512, fusion=FUSION),
    vae=VAEConfig(),                 # unused (compression is off)
    server=ServerConfig(
        scenario="llava",
        n_soft_tokens=N_SOFT_TOKENS,
        freeze_llm=True,
        load_in_8bit=LOAD_IN_8BIT,
    ),
    use_compression=False,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model builders ────────────────────────────────────────────────────────────

def build_edge(cfg: EMMAConfig) -> EdgePipeline:
    return EdgePipeline(config=cfg)


def build_server(cfg: EMMAConfig) -> tuple[ServerPipeline, AutoTokenizer]:
    llm, d_llm = _load_llm("llava", load_in_8bit=LOAD_IN_8BIT, dtype=LLM_DTYPE)
    if not LOAD_IN_8BIT:
        llm = llm.to(DEVICE)   # 8-bit checkpoints are already device-placed at load time

    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    projection = _build_projection(D_SERVER_IN, N_SOFT_TOKENS, d_llm).to(DEVICE)
    match_head = nn.Linear(d_llm, 1).to(DEVICE)

    server = ServerPipeline(
        llm=llm,
        d_llm=d_llm,
        projection=projection,
        match_head=match_head,
        n_soft_tokens=N_SOFT_TOKENS,
        freeze_llm=True,
        vae_decoder=None,
        debug=DEBUG_FORWARD,
        debug_every=DEBUG_EVERY,
    )
    return server, tokenizer


def configure_server_stage(server: ServerPipeline) -> AdamW:
    for p in server.parameters():
        p.requires_grad = False
    for module in [server.projection, server.match_head]:
        for p in module.parameters():
            p.requires_grad = True

    trainable = [p for p in server.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"\n── Stage 'server' (llava): training projection + match head ({n_params:,} params) ──")

    return AdamW(trainable, lr=LR, weight_decay=1e-2)


# ── Loss & metrics ────────────────────────────────────────────────────────────

def match_loss(preds: dict, batch: dict) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(preds["match"].squeeze(-1), batch["label"])


def match_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return ((logits.squeeze(-1) > 0) == (labels > 0.5)).float().mean().item()


# ── Embedding cache ───────────────────────────────────────────────────────────

def cache_embeddings(edge: EdgePipeline, loaders: dict, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    edge.eval()

    for split, loader in loaders.items():
        fused_list, label_list = [], []
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

        torch.save(torch.cat(fused_list), os.path.join(cache_dir, f"{split}_fused.pt"))
        torch.save(torch.cat(label_list), os.path.join(cache_dir, f"{split}_label.pt"))
        n = sum(len(f) for f in fused_list)
        print(f"  cached {split}: {n} samples")

    print(f"  embeddings written to {cache_dir}/")


def standardize_cache(cache_dir: str, tag: str) -> str:
    """
    Standardize each split's cached fused/latent tensor to zero mean, unit
    variance per feature, using statistics from the train split only, and
    write the result to a sibling cache dir (original is left untouched).

    Only invoked when --standardize-input is passed -- opt-in, not a
    default. Why it might be needed: unlike GPT-2 (fp32, wide numeric
    headroom), LLaVA's fp16 backbone overflows easily -- an unbounded-scale
    input (e.g. raw PCA output, which just preserves whatever scale the
    source embeddings have) can produce inf/NaN within the first batch.
    Only touches this script's cached data, not ServerPipeline itself --
    train.py's GPT-2 pipeline is unaffected either way.
    """
    norm_dir = f"{cache_dir}_normalized"
    os.makedirs(norm_dir, exist_ok=True)

    train_fused = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)
    mean = train_fused.mean(dim=0, keepdim=True)
    std  = train_fused.std(dim=0, keepdim=True).clamp_min(1e-6)

    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        z = (fused - mean) / std
        torch.save(z,      os.path.join(norm_dir, f"{split}_fused.pt"))
        torch.save(labels, os.path.join(norm_dir, f"{split}_label.pt"))

    print(f"  standardized {tag} inputs (mean/std from train split) -> {norm_dir}")
    return norm_dir


def _cache_latents(cache_dir: str, tag: str, encode_fn) -> str:
    """
    Shared plumbing for every compressor type below: run encode_fn over
    each split's cached fused embeddings ([N, d_in] CPU tensor -> [N,
    d_latent] CPU tensor) and write the result to a latent cache dir,
    mirroring train.py's cache layout so get_cached_loaders can read it.
    """
    latent_cache_dir = os.path.join(cache_dir, f"{tag}_latents_{D_LATENT}{_seed}")
    os.makedirs(latent_cache_dir, exist_ok=True)
    for split in ("train", "valid", "test"):
        fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)
        z = encode_fn(fused)
        torch.save(z,      os.path.join(latent_cache_dir, f"{split}_fused.pt"))
        torch.save(labels, os.path.join(latent_cache_dir, f"{split}_label.pt"))
        print(f"  cached {split} {tag} latents: {tuple(z.shape)}")
    return latent_cache_dir


def _require_checkpoint(path: str, exists_check):
    if not exists_check(path):
        raise FileNotFoundError(
            f"Compressor checkpoint not found: {path}\n"
            f"This script only applies an already-trained/fit compressor -- "
            f"it does not train one. Pass --compression-checkpoint to point "
            f"at the right path, or produce one first with train.py.")


def apply_pretrained_autoencoder(cache_dir: str, ae_ckpt: str, tag: str) -> str:
    """
    Load an already-trained AE/ContrastiveAE/CrossModalAE checkpoint (all
    three use the same AutoEncoder class -- only their training loss
    differs) and encode every split's fused embeddings into cached latents.
    No training happens here: the compressor is reused as-is from the
    GPT-2 runs, since compression is decoupled from which server LLM
    consumes the latents.
    """
    from emma.compression.autoencoder import AutoEncoder

    _require_checkpoint(ae_ckpt, os.path.exists)
    ae = AutoEncoder.load(ae_ckpt).to(DEVICE)
    ae.eval()

    def encode_fn(fused):
        with torch.no_grad():
            return ae.encoder(fused.to(DEVICE)).cpu()

    return _cache_latents(cache_dir, tag, encode_fn)


def apply_pretrained_vae(cache_dir: str, vae_ckpt: str, tag: str) -> str:
    """
    Load an already-trained VAE checkpoint and encode every split's fused
    embeddings using the encoder's mean (mu) only -- no sampling, matching
    the deterministic inference path used everywhere else in the paper.
    No training happens here.
    """
    from emma.compression.vae import VAE

    _require_checkpoint(vae_ckpt, os.path.exists)
    vae = VAE.load(vae_ckpt).to(DEVICE)
    vae.eval()

    def encode_fn(fused):
        with torch.no_grad():
            return vae.encoder.encode(fused.to(DEVICE)).cpu()

    return _cache_latents(cache_dir, tag, encode_fn)


def apply_pretrained_blockpca(cache_dir: str, blockpca_ckpt: str, tag: str) -> str:
    """
    Load an already-fit BlockPCA compressor (closed-form, no gradient
    training -- 4 independent per-block PCAs) and encode every split's
    fused embeddings. Only valid with match fusion, same restriction as
    train.py.
    """
    from emma.compression.block_pca import BlockPCACompressor

    if FUSION != "match":
        raise ValueError("BlockPCA is only supported with match fusion "
                         f"(fusion='match'). Got fusion='{FUSION}'.")
    _require_checkpoint(blockpca_ckpt, os.path.isdir)
    bpca = BlockPCACompressor.load(blockpca_ckpt)

    def encode_fn(fused):
        return bpca.encode(fused)

    return _cache_latents(cache_dir, tag, encode_fn)


def fit_and_apply_pca(cache_dir: str, tag: str) -> str:
    """
    Fit PCA fresh on this run's cached training embeddings and encode
    every split. Not loaded from disk: train.py's pca_<d>.npz checkpoint
    name doesn't encode which encoder/fusion it was fit on (a pre-existing
    gap in train.py, not something safe to reuse blind), so refitting here
    -- cheap, closed-form, no gradient training -- avoids silently applying
    a mismatched compressor.
    """
    from emma.compression.pca import PCACompressor

    train_fused = torch.load(os.path.join(cache_dir, "train_fused.pt"), weights_only=True)
    print(f"\n── Fitting PCA-{D_LATENT} fresh on {ENCODER}+{FUSION} embeddings ──")
    pca = PCACompressor(n_components=D_LATENT)
    pca.fit(train_fused)

    def encode_fn(fused):
        return pca.encode(fused)

    return _cache_latents(cache_dir, tag, encode_fn)


def fit_and_apply_lda(cache_dir: str, tag: str) -> str:
    """
    Fit the LDA-PCA hybrid fresh on this run's cached training embeddings +
    labels and encode every split. Same reasoning as PCA above: train.py's
    lda_<d>.npz checkpoint name doesn't encode encoder/fusion either.
    """
    from emma.compression.lda import LDACompressor

    train_fused  = torch.load(os.path.join(cache_dir, "train_fused.pt"),  weights_only=True)
    train_labels = torch.load(os.path.join(cache_dir, "train_label.pt"),  weights_only=True)
    print(f"\n── Fitting LDA-{D_LATENT} fresh on {ENCODER}+{FUSION} embeddings ──")
    lda = LDACompressor(n_components=D_LATENT)
    lda.fit(train_fused, train_labels)

    def encode_fn(fused):
        return lda.encode(fused)

    return _cache_latents(cache_dir, tag, encode_fn)


class CachedEmbeddingDataset(Dataset):
    def __init__(self, cache_dir: str, split: str):
        self.fused  = torch.load(os.path.join(cache_dir, f"{split}_fused.pt"), weights_only=True)
        self.labels = torch.load(os.path.join(cache_dir, f"{split}_label.pt"), weights_only=True)

    def __len__(self):
        return len(self.fused)

    def __getitem__(self, idx):
        return {"fused": self.fused[idx], "label": self.labels[idx]}


def get_cached_loaders(cache_dir: str, batch_size: int) -> dict:
    loaders = {}
    for split in ("train", "valid", "test"):
        ds = CachedEmbeddingDataset(cache_dir, split)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
    return loaders


# ── Server epoch ──────────────────────────────────────────────────────────────

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
                grad_norm = nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    # A non-finite gradient would otherwise permanently corrupt
                    # the model (NaN parameters never recover on later steps).
                    # Skip just this update instead of losing the whole run.
                    print(f"    [warn] non-finite grad norm ({float(grad_norm):.3g}) "
                          f"-- skipping optimizer step")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nBuilding models (scenario=llava, compression={COMPRESSION}, load_in_8bit={LOAD_IN_8BIT})...")
    server, tokenizer = build_server(cfg)

    instr_enc  = tokenizer(INSTRUCTION, return_tensors="pt",
                           padding=True, truncation=True, max_length=32)
    instr_ids  = instr_enc["input_ids"]
    instr_mask = instr_enc["attention_mask"]

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

    if COMPRESSION == "none":
        server_cache = CACHE_DIR
    elif COMPRESSION in ("ae", "contrastiveae", "crossmodalae"):
        print(f"\nApplying pretrained {COMPRESSION} compressor from {COMPRESSION_CHECKPOINT} ...")
        server_cache = apply_pretrained_autoencoder(
            cache_dir=CACHE_DIR, ae_ckpt=COMPRESSION_CHECKPOINT, tag=COMPRESSION)
    elif COMPRESSION == "vae":
        print(f"\nApplying pretrained VAE compressor from {COMPRESSION_CHECKPOINT} ...")
        server_cache = apply_pretrained_vae(
            cache_dir=CACHE_DIR, vae_ckpt=COMPRESSION_CHECKPOINT, tag=COMPRESSION)
    elif COMPRESSION == "blockpca":
        print(f"\nApplying pretrained BlockPCA compressor from {COMPRESSION_CHECKPOINT} ...")
        server_cache = apply_pretrained_blockpca(
            cache_dir=CACHE_DIR, blockpca_ckpt=COMPRESSION_CHECKPOINT, tag=COMPRESSION)
    elif COMPRESSION == "pca":
        server_cache = fit_and_apply_pca(cache_dir=CACHE_DIR, tag=COMPRESSION)
    else:  # "lda"
        server_cache = fit_and_apply_lda(cache_dir=CACHE_DIR, tag=COMPRESSION)

    if STANDARDIZE_INPUT:
        server_cache = standardize_cache(server_cache, COMPRESSION)

    cached_loaders = get_cached_loaders(server_cache, batch_size=BATCH_SIZE)

    optimizer = configure_server_stage(server)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * len(cached_loaders["train"]))
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

    print("\nFinal test evaluation...")
    test_m = run_server_epoch(server, cached_loaders["test"], optimizer, scheduler,
                              instr_ids, instr_mask, training=False)
    print(f"Test | loss {test_m['loss']:.4f} | accuracy {test_m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
