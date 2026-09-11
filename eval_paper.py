"""
Targeted test-set evaluation for paper results only.
Evaluates only the checkpoints needed for the paper tables.
Saves to checkpoints/paper_results.txt.
"""

import os, gc, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from emma.server.pipeline import ServerPipeline, _build_projection

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION   = "Does the image match the description?"
N_SOFT_TOKENS = 8
BATCH_SIZE    = 512
CKPT_DIR      = "checkpoints"

GPT2_PATH = (
    "/scratch/user/motahare/hf_cache/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

# ── Checkpoints needed for the paper ─────────────────────────────────────────
# Format: (checkpoint_tag, latent_cache_subdir or None for embed_cache root)
# embed_cache root = no compression (none)
PAPER_CKPTS = [
    # CLIP fusion ablation (no compression)
    ("none_72k",             "embed_cache_72000_5000_5000",          None),
    ("none_concat_72k",      "embed_cache_concat_72000_5000_5000",   None),
    ("none_match_72k",       "embed_cache_match_72000_5000_5000",    None),

    # CLIP compression (mean fusion, 64-dim, 5 seeds)
    ("ae_64_72k",            "embed_cache_72000_5000_5000",          "ae_latents_64"),
    ("ae_64_72k_s1",         "embed_cache_72000_5000_5000",          "ae_latents_64_s1"),
    ("ae_64_72k_s2",         "embed_cache_72000_5000_5000",          "ae_latents_64_s2"),
    ("ae_64_72k_s3",         "embed_cache_72000_5000_5000",          "ae_latents_64_s3"),
    ("ae_64_72k_s4",         "embed_cache_72000_5000_5000",          "ae_latents_64_s4"),
    ("vae_64_72k",           "embed_cache_72000_5000_5000",          "vae_latents_64"),
    ("vae_64_72k_s1",        "embed_cache_72000_5000_5000",          "vae_latents_64_s1"),
    ("vae_64_72k_s2",        "embed_cache_72000_5000_5000",          "vae_latents_64_s2"),
    ("vae_64_72k_s3",        "embed_cache_72000_5000_5000",          "vae_latents_64_s3"),
    ("vae_64_72k_s4",        "embed_cache_72000_5000_5000",          "vae_latents_64_s4"),
    ("pca_64_72k",           "embed_cache_72000_5000_5000",          "pca_latents_64"),
    ("pca_64_72k_s1",        "embed_cache_72000_5000_5000",          "pca_latents_64_s1"),
    ("pca_64_72k_s2",        "embed_cache_72000_5000_5000",          "pca_latents_64_s2"),
    ("pca_64_72k_s3",        "embed_cache_72000_5000_5000",          "pca_latents_64_s3"),
    ("pca_64_72k_s4",        "embed_cache_72000_5000_5000",          "pca_latents_64_s4"),
    ("distilae_64_72k",      "embed_cache_72000_5000_5000",          "distilae_latents_64"),
    ("distilae_64_72k_s1",   "embed_cache_72000_5000_5000",          "distilae_latents_64_s1"),
    ("distilae_64_72k_s2",   "embed_cache_72000_5000_5000",          "distilae_latents_64_s2"),
    ("distilae_64_72k_s3",   "embed_cache_72000_5000_5000",          "distilae_latents_64_s3"),
    ("distilae_64_72k_s4",   "embed_cache_72000_5000_5000",          "distilae_latents_64_s4"),

    # MC fusion ablation (no compression)
    ("none_mobileclip_72k",        "embed_cache_mobileclip_72000_5000_5000",        None),
    ("none_mobileclip_concat_72k", "embed_cache_mobileclip_concat_72000_5000_5000", None),
    ("none_mobileclip_match_72k",  "embed_cache_mobileclip_match_72000_5000_5000",  None),

    # MC compression — mean fusion
    ("ae_64_72k_mobileclip",    "embed_cache_mobileclip_72000_5000_5000",   "ae_latents_64"),
    ("ae_64_72k_mobileclip_s1", "embed_cache_mobileclip_72000_5000_5000",   "ae_latents_64_s1"),
    ("ae_64_72k_mobileclip_s2", "embed_cache_mobileclip_72000_5000_5000",   "ae_latents_64_s2"),
    ("vae_64_72k_mobileclip",   "embed_cache_mobileclip_72000_5000_5000",   "vae_latents_64"),
    ("vae_64_72k_mobileclip_s1","embed_cache_mobileclip_72000_5000_5000",   "vae_latents_64_s1"),
    ("vae_64_72k_mobileclip_s2","embed_cache_mobileclip_72000_5000_5000",   "vae_latents_64_s2"),
    ("pca_64_72k_mobileclip",   "embed_cache_mobileclip_72000_5000_5000",   "pca_latents_64"),
    ("pca_64_72k_mobileclip_s1","embed_cache_mobileclip_72000_5000_5000",   "pca_latents_64_s1"),
    ("pca_64_72k_mobileclip_s2","embed_cache_mobileclip_72000_5000_5000",   "pca_latents_64_s2"),
    ("lda_64_72k_mobileclip",   "embed_cache_mobileclip_72000_5000_5000",   "lda_latents_64"),
    ("lda_64_72k_mobileclip_s1","embed_cache_mobileclip_72000_5000_5000",   "lda_latents_64_s1"),
    ("lda_64_72k_mobileclip_s2","embed_cache_mobileclip_72000_5000_5000",   "lda_latents_64_s2"),
    ("contrastiveae_64_72k_mobileclip",   "embed_cache_mobileclip_72000_5000_5000", "contrastiveae_latents_64"),
    ("contrastiveae_64_72k_mobileclip_s1","embed_cache_mobileclip_72000_5000_5000", "contrastiveae_latents_64_s1"),
    ("contrastiveae_64_72k_mobileclip_s2","embed_cache_mobileclip_72000_5000_5000", "contrastiveae_latents_64_s2"),
    ("crossmodalae_64_72k_mobileclip",   "embed_cache_mobileclip_72000_5000_5000",  "crossmodalae_latents_64"),
    ("crossmodalae_64_72k_mobileclip_s1","embed_cache_mobileclip_72000_5000_5000",  "crossmodalae_latents_64_s1"),
    ("crossmodalae_64_72k_mobileclip_s2","embed_cache_mobileclip_72000_5000_5000",  "crossmodalae_latents_64_s2"),

    # MC compression — match fusion
    ("ae_64_72k_mobileclip_match",    "embed_cache_mobileclip_match_72000_5000_5000", "ae_latents_64"),
    ("ae_64_72k_mobileclip_match_s1", "embed_cache_mobileclip_match_72000_5000_5000", "ae_latents_64_s1"),
    ("ae_64_72k_mobileclip_match_s2", "embed_cache_mobileclip_match_72000_5000_5000", "ae_latents_64_s2"),
    ("vae_64_72k_mobileclip_match",   "embed_cache_mobileclip_match_72000_5000_5000", "vae_latents_64"),
    ("vae_64_72k_mobileclip_match_s1","embed_cache_mobileclip_match_72000_5000_5000", "vae_latents_64_s1"),
    ("vae_64_72k_mobileclip_match_s2","embed_cache_mobileclip_match_72000_5000_5000", "vae_latents_64_s2"),
    ("pca_64_72k_mobileclip_match",   "embed_cache_mobileclip_match_72000_5000_5000", "pca_latents_64"),
    ("pca_64_72k_mobileclip_match_s1","embed_cache_mobileclip_match_72000_5000_5000", "pca_latents_64_s1"),
    ("pca_64_72k_mobileclip_match_s2","embed_cache_mobileclip_match_72000_5000_5000", "pca_latents_64_s2"),
    ("lda_64_72k_mobileclip_match",   "embed_cache_mobileclip_match_72000_5000_5000", "lda_latents_64"),
    ("lda_64_72k_mobileclip_match_s1","embed_cache_mobileclip_match_72000_5000_5000", "lda_latents_64_s1"),
    ("lda_64_72k_mobileclip_match_s2","embed_cache_mobileclip_match_72000_5000_5000", "lda_latents_64_s2"),
    ("blockpca_64_72k_mobileclip_match",   "embed_cache_mobileclip_match_72000_5000_5000", "blockpca_latents_64"),
    ("blockpca_64_72k_mobileclip_match_s1","embed_cache_mobileclip_match_72000_5000_5000", "blockpca_latents_64_s1"),
    ("blockpca_64_72k_mobileclip_match_s2","embed_cache_mobileclip_match_72000_5000_5000", "blockpca_latents_64_s2"),
    ("contrastiveae_64_72k_mobileclip_match",   "embed_cache_mobileclip_match_72000_5000_5000", "contrastiveae_latents_64"),
    ("contrastiveae_64_72k_mobileclip_match_s1","embed_cache_mobileclip_match_72000_5000_5000", "contrastiveae_latents_64_s1"),
    ("contrastiveae_64_72k_mobileclip_match_s2","embed_cache_mobileclip_match_72000_5000_5000", "contrastiveae_latents_64_s2"),
    ("crossmodalae_64_72k_mobileclip_match",   "embed_cache_mobileclip_match_72000_5000_5000",  "crossmodalae_latents_64"),
    ("crossmodalae_64_72k_mobileclip_match_s1","embed_cache_mobileclip_match_72000_5000_5000",  "crossmodalae_latents_64_s1"),
    ("crossmodalae_64_72k_mobileclip_match_s2","embed_cache_mobileclip_match_72000_5000_5000",  "crossmodalae_latents_64_s2"),

    # MC compression — concat fusion
    ("ae_64_72k_mobileclip_concat",    "embed_cache_mobileclip_concat_72000_5000_5000", "ae_latents_64"),
    ("ae_64_72k_mobileclip_concat_s1", "embed_cache_mobileclip_concat_72000_5000_5000", "ae_latents_64_s1"),
    ("ae_64_72k_mobileclip_concat_s2", "embed_cache_mobileclip_concat_72000_5000_5000", "ae_latents_64_s2"),
    ("vae_64_72k_mobileclip_concat",   "embed_cache_mobileclip_concat_72000_5000_5000", "vae_latents_64"),
    ("vae_64_72k_mobileclip_concat_s1","embed_cache_mobileclip_concat_72000_5000_5000", "vae_latents_64_s1"),
    ("vae_64_72k_mobileclip_concat_s2","embed_cache_mobileclip_concat_72000_5000_5000", "vae_latents_64_s2"),
    ("lda_64_72k_mobileclip_concat",   "embed_cache_mobileclip_concat_72000_5000_5000", "lda_latents_64"),
    ("lda_64_72k_mobileclip_concat_s1","embed_cache_mobileclip_concat_72000_5000_5000", "lda_latents_64_s1"),
    ("lda_64_72k_mobileclip_concat_s2","embed_cache_mobileclip_concat_72000_5000_5000", "lda_latents_64_s2"),
    ("contrastiveae_64_72k_mobileclip_concat",   "embed_cache_mobileclip_concat_72000_5000_5000", "contrastiveae_latents_64"),
    ("contrastiveae_64_72k_mobileclip_concat_s1","embed_cache_mobileclip_concat_72000_5000_5000", "contrastiveae_latents_64_s1"),
    ("contrastiveae_64_72k_mobileclip_concat_s2","embed_cache_mobileclip_concat_72000_5000_5000", "contrastiveae_latents_64_s2"),
    ("crossmodalae_64_72k_mobileclip_concat",   "embed_cache_mobileclip_concat_72000_5000_5000",  "crossmodalae_latents_64"),
    ("crossmodalae_64_72k_mobileclip_concat_s1","embed_cache_mobileclip_concat_72000_5000_5000",  "crossmodalae_latents_64_s1"),
    ("crossmodalae_64_72k_mobileclip_concat_s2","embed_cache_mobileclip_concat_72000_5000_5000",  "crossmodalae_latents_64_s2"),
]


def load_gpt2():
    llm       = AutoModelForCausalLM.from_pretrained(GPT2_PATH, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(GPT2_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer, llm.config.hidden_size


def load_server(ckpt_path, llm, d_llm):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["server"]
    d_in  = state["projection.0.weight"].shape[1]
    server = ServerPipeline(
        llm=llm, d_llm=d_llm,
        projection=_build_projection(d_in, N_SOFT_TOKENS, d_llm),
        match_head=nn.Linear(d_llm, 1),
        n_soft_tokens=N_SOFT_TOKENS, freeze_llm=True, vae_decoder=None,
    )
    server.load_state_dict(state)
    server.eval()
    return server, ckpt["val_metrics"]["accuracy"]


def evaluate(server, tokenizer, cache_dir):
    fused  = torch.load(os.path.join(cache_dir, "test_fused.pt"),  weights_only=True)
    labels = torch.load(os.path.join(cache_dir, "test_label.pt"),  weights_only=True)
    enc    = tokenizer(INSTRUCTION, return_tensors="pt",
                       padding=True, truncation=True, max_length=32)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    loader = DataLoader(TensorDataset(fused, labels), batch_size=BATCH_SIZE)
    logits_all, labels_all = [], []
    server = server.to(DEVICE)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); B = x.size(0)
            out = server(x, ids.expand(B,-1).to(DEVICE), mask.expand(B,-1).to(DEVICE))
            logits_all.append(out["match"].cpu()); labels_all.append(y)
    logits = torch.cat(logits_all); labels = torch.cat(labels_all)
    return ((logits.squeeze(-1) > 0) == (labels > 0.5)).float().mean().item()


def main():
    print("Loading GPT-2 once...")
    llm, tokenizer, d_llm = load_gpt2()

    out_path = os.path.join(CKPT_DIR, "paper_results.txt")
    with open(out_path, "w") as f:
        f.write(f"{'Tag':65s}  {'Val':>7}  {'Test':>7}\n")
        f.write("-" * 85 + "\n")

        for tag, cache_subdir, latent_subdir in PAPER_CKPTS:
            ckpt_path  = os.path.join(CKPT_DIR, f"best_server_{tag}.pt")
            cache_dir  = os.path.join(CKPT_DIR, cache_subdir)
            if latent_subdir:
                cache_dir = os.path.join(cache_dir, latent_subdir)

            if not os.path.exists(ckpt_path):
                print(f"  MISSING ckpt: {tag}"); continue
            if not os.path.exists(os.path.join(cache_dir, "test_fused.pt")):
                print(f"  MISSING cache: {tag} → {cache_dir}"); continue

            try:
                server, val_acc = load_server(ckpt_path, llm, d_llm)
                test_acc = evaluate(server, tokenizer, cache_dir)
                line = f"  {tag:65s}  {val_acc:.4f}   {test_acc:.4f}"
                print(line); f.write(line + "\n"); f.flush()
            except Exception as e:
                print(f"  ERROR {tag}: {e}")
            finally:
                del server; gc.collect()

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
