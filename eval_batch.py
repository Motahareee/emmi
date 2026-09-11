"""
Evaluate a specific batch of checkpoints (3 at a time) for test-set accuracy.

Usage:
    python3 eval_batch.py --batch 0    # batch 0: MC none baselines
    python3 eval_batch.py --batch 1    # batch 1: CLIP none baselines
    python3 eval_batch.py --batch 2    # etc.
    python3 eval_batch.py --list       # show all batches

Results are appended to checkpoints/test_results_batched.txt
"""

import os
import gc
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from emma.server.pipeline import ServerPipeline, _build_projection

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION   = "Does the image match the description?"
N_SOFT_TOKENS = 8
BATCH_SIZE    = 256
CKPT_DIR      = "checkpoints"
OUT_FILE      = os.path.join(CKPT_DIR, "test_results_batched.txt")

_HF_CACHE = (
    "/scratch/user/motahare/hf_cache/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
GPT2_PATH = _HF_CACHE if os.path.isdir(_HF_CACHE) else "gpt2"

# ---------------------------------------------------------------------------
# (ckpt_filename, cache_subdir_relative_to_CKPT_DIR)
# cache_subdir is the directory that contains test_fused.pt / test_label.pt
# ---------------------------------------------------------------------------
BATCHES = [
    # ── batch 0: MC none baselines ──────────────────────────────────────────
    [
        ("best_server_none_mobileclip_72k.pt",
         "embed_cache_mobileclip_72000_5000_5000"),
        ("best_server_none_mobileclip_concat_72k.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000"),
        ("best_server_none_mobileclip_match_72k.pt",
         "embed_cache_mobileclip_match_72000_5000_5000"),
    ],
    # ── batch 1: CLIP none baselines ────────────────────────────────────────
    [
        ("best_server_none_72k.pt",
         "embed_cache_72000_5000_5000"),
        ("best_server_none_concat_72k.pt",
         "embed_cache_concat_72000_5000_5000"),
        ("best_server_none_match_72k.pt",
         "embed_cache_match_72000_5000_5000"),
    ],
    # ── batch 2: CLIP PCA seeds 0-2 ─────────────────────────────────────────
    [
        ("best_server_pca_64_72k.pt",
         "embed_cache_72000_5000_5000/pca_latents_64"),
        ("best_server_pca_64_72k_s1.pt",
         "embed_cache_72000_5000_5000/pca_latents_64_s1"),
        ("best_server_pca_64_72k_s2.pt",
         "embed_cache_72000_5000_5000/pca_latents_64_s2"),
    ],
    # ── batch 3: CLIP PCA seeds 3-4 + VAE seed 0 ────────────────────────────
    [
        ("best_server_pca_64_72k_s3.pt",
         "embed_cache_72000_5000_5000/pca_latents_64_s3"),
        ("best_server_pca_64_72k_s4.pt",
         "embed_cache_72000_5000_5000/pca_latents_64_s4"),
        ("best_server_vae_64_72k.pt",
         "embed_cache_72000_5000_5000/vae_latents_64"),
    ],
    # ── batch 4: CLIP VAE seeds 1-3 ─────────────────────────────────────────
    [
        ("best_server_vae_64_72k_s1.pt",
         "embed_cache_72000_5000_5000/vae_latents_64_s1"),
        ("best_server_vae_64_72k_s2.pt",
         "embed_cache_72000_5000_5000/vae_latents_64_s2"),
        ("best_server_vae_64_72k_s3.pt",
         "embed_cache_72000_5000_5000/vae_latents_64_s3"),
    ],
    # ── batch 5: CLIP VAE seed 4 + AE seeds 0-1 ─────────────────────────────
    [
        ("best_server_vae_64_72k_s4.pt",
         "embed_cache_72000_5000_5000/vae_latents_64_s4"),
        ("best_server_ae_64_72k.pt",
         "embed_cache_72000_5000_5000/ae_latents_64"),
        ("best_server_ae_64_72k_s1.pt",
         "embed_cache_72000_5000_5000/ae_latents_64_s1"),
    ],
    # ── batch 6: CLIP AE seeds 2-4 ──────────────────────────────────────────
    [
        ("best_server_ae_64_72k_s2.pt",
         "embed_cache_72000_5000_5000/ae_latents_64_s2"),
        ("best_server_ae_64_72k_s3.pt",
         "embed_cache_72000_5000_5000/ae_latents_64_s3"),
        ("best_server_ae_64_72k_s4.pt",
         "embed_cache_72000_5000_5000/ae_latents_64_s4"),
    ],
    # ── batch 7: CLIP DistilAE seeds 0-2 ────────────────────────────────────
    [
        ("best_server_distilae_64_72k.pt",
         "embed_cache_72000_5000_5000/distilae_latents_64"),
        ("best_server_distilae_64_72k_s1.pt",
         "embed_cache_72000_5000_5000/distilae_latents_64_s1"),
        ("best_server_distilae_64_72k_s2.pt",
         "embed_cache_72000_5000_5000/distilae_latents_64_s2"),
    ],
    # ── batch 8: CLIP DistilAE seeds 3-4 + MC PCA mean ──────────────────────
    [
        ("best_server_distilae_64_72k_s3.pt",
         "embed_cache_72000_5000_5000/distilae_latents_64_s3"),
        ("best_server_distilae_64_72k_s4.pt",
         "embed_cache_72000_5000_5000/distilae_latents_64_s4"),
        ("best_server_pca_64_72k_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000/pca_latents_64"),
    ],
    # ── batch 9: MC PCA concat+match + MC LDA mean ──────────────────────────
    [
        ("best_server_pca_64_72k_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/pca_latents_64"),
        ("best_server_pca_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/pca_latents_64"),
        ("best_server_lda_64_72k_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000/lda_latents_64"),
    ],
    # ── batch 10: MC LDA concat+match + MC VAE mean ─────────────────────────
    [
        ("best_server_lda_64_72k_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/lda_latents_64"),
        ("best_server_lda_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/lda_latents_64"),
        ("best_server_vae_64_72k_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000/vae_latents_64"),
    ],
    # ── batch 11: MC VAE concat+match + MC DistilAE mean ────────────────────
    [
        ("best_server_vae_64_72k_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/vae_latents_64"),
        ("best_server_vae_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/vae_latents_64"),
        ("best_server_distilae_64_72k_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000/distilae_latents_64"),
    ],
    # ── batch 12: MC DistilAE concat+match + MC ContrastiveAE mean ──────────
    [
        ("best_server_distilae_64_72k_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/distilae_latents_64"),
        ("best_server_distilae_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/distilae_latents_64"),
        ("best_server_contrastiveae_64_72k_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000/contrastiveae_latents_64"),
    ],
    # ── batch 13: MC ContrastiveAE concat+match ──────────────────────────────
    [
        ("best_server_contrastiveae_64_72k_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/contrastiveae_latents_64"),
        ("best_server_contrastiveae_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/contrastiveae_latents_64"),
        ("best_server_contrastiveae_64_72k_mobileclip_match_decoded.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/contrastiveae_latents_64_decoded"),
    ],
    # ── batch 14: MC CrossModalAE all fusions ───────────────────────────────
    [
        ("best_server_crossmodalae_64_72k_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000/crossmodalae_latents_64"),
        ("best_server_crossmodalae_64_72k_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/crossmodalae_latents_64"),
        ("best_server_crossmodalae_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/crossmodalae_latents_64"),
    ],
    # ── batch 15: MC BlockPCA match ──────────────────────────────────────────
    [
        ("best_server_blockpca_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/blockpca_latents_64"),
        ("best_server_blockpca_64_72k_mobileclip_match_s1.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/blockpca_latents_64_s1"),
        ("best_server_blockpca_64_72k_mobileclip_match_s2.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/blockpca_latents_64_s2"),
    ],
    # ── batch 16: MC none mean (3 seeds) ────────────────────────────────────
    [
        ("best_server_none_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000"),
        ("best_server_none_mobileclip_s1.pt",
         "embed_cache_mobileclip_72000_5000_5000"),
        ("best_server_none_mobileclip_s2.pt",
         "embed_cache_mobileclip_72000_5000_5000"),
    ],
    # ── batch 17: MC none concat (3 seeds) ──────────────────────────────────
    [
        ("best_server_none_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000"),
        ("best_server_none_mobileclip_concat_s1.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000"),
        ("best_server_none_mobileclip_concat_s2.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000"),
    ],
    # ── batch 18: MC none match (3 seeds) ───────────────────────────────────
    [
        ("best_server_none_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000"),
        ("best_server_none_mobileclip_match_s1.pt",
         "embed_cache_mobileclip_match_72000_5000_5000"),
        ("best_server_none_mobileclip_match_s2.pt",
         "embed_cache_mobileclip_match_72000_5000_5000"),
    ],
    # ── batch 19: CLIP none mean (seeds 0-2) ────────────────────────────────
    [
        ("best_server_none.pt",
         "embed_cache_72000_5000_5000"),
        ("best_server_none_s1.pt",
         "embed_cache_72000_5000_5000"),
        ("best_server_none_s2.pt",
         "embed_cache_72000_5000_5000"),
    ],
    # ── batch 20: CLIP none mean (seeds 3-4) + concat (seeds 0-1) ───────────
    [
        ("best_server_none_s3.pt",
         "embed_cache_72000_5000_5000"),
        ("best_server_none_s4.pt",
         "embed_cache_72000_5000_5000"),
        ("best_server_none_concat.pt",
         "embed_cache_concat_72000_5000_5000"),
    ],
    # ── batch 21: CLIP none concat (seeds 1-2) + match (seed 0) ────────────
    [
        ("best_server_none_concat_s1.pt",
         "embed_cache_concat_72000_5000_5000"),
        ("best_server_none_concat_s2.pt",
         "embed_cache_concat_72000_5000_5000"),
        ("best_server_none_match.pt",
         "embed_cache_match_72000_5000_5000"),
    ],
    # ── batch 22: CLIP none match (seeds 1-2) ───────────────────────────────
    [
        ("best_server_none_match_s1.pt",
         "embed_cache_match_72000_5000_5000"),
        ("best_server_none_match_s2.pt",
         "embed_cache_match_72000_5000_5000"),
    ],
    # ── batch 23: MC AE mean (3 seeds) ──────────────────────────────────────
    [
        ("best_server_ae_64_72k_mobileclip.pt",
         "embed_cache_mobileclip_72000_5000_5000/ae_latents_64"),
        ("best_server_ae_64_72k_mobileclip_s1.pt",
         "embed_cache_mobileclip_72000_5000_5000/ae_latents_64_s1"),
        ("best_server_ae_64_72k_mobileclip_s2.pt",
         "embed_cache_mobileclip_72000_5000_5000/ae_latents_64_s2"),
    ],
    # ── batch 24: MC AE concat (3 seeds) ────────────────────────────────────
    [
        ("best_server_ae_64_72k_mobileclip_concat.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/ae_latents_64"),
        ("best_server_ae_64_72k_mobileclip_concat_s1.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/ae_latents_64_s1"),
        ("best_server_ae_64_72k_mobileclip_concat_s2.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/ae_latents_64_s2"),
    ],
    # ── batch 25: MC AE match (3 seeds) ─────────────────────────────────────
    [
        ("best_server_ae_64_72k_mobileclip_match.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/ae_latents_64"),
        ("best_server_ae_64_72k_mobileclip_match_s1.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/ae_latents_64_s1"),
        ("best_server_ae_64_72k_mobileclip_match_s2.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/ae_latents_64_s2"),
    ],
    # ── batch 26: ContrastiveAE seeds 1-2 (mean + concat + match s1) ────────
    [
        ("best_server_contrastiveae_64_72k_mobileclip_s1.pt",
         "embed_cache_mobileclip_72000_5000_5000/contrastiveae_latents_64_s1"),
        ("best_server_contrastiveae_64_72k_mobileclip_concat_s1.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/contrastiveae_latents_64_s1"),
        ("best_server_contrastiveae_64_72k_mobileclip_match_s1.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/contrastiveae_latents_64_s1"),
    ],
    # ── batch 27: ContrastiveAE seed 2 (mean + concat + match) ─────────────
    [
        ("best_server_contrastiveae_64_72k_mobileclip_s2.pt",
         "embed_cache_mobileclip_72000_5000_5000/contrastiveae_latents_64_s2"),
        ("best_server_contrastiveae_64_72k_mobileclip_concat_s2.pt",
         "embed_cache_mobileclip_concat_72000_5000_5000/contrastiveae_latents_64_s2"),
        ("best_server_contrastiveae_64_72k_mobileclip_match_s2.pt",
         "embed_cache_mobileclip_match_72000_5000_5000/contrastiveae_latents_64_s2"),
    ],
]


# ---------------------------------------------------------------------------
# Inference helpers (same as eval_test.py)
# ---------------------------------------------------------------------------

def load_gpt2():
    llm = AutoModelForCausalLM.from_pretrained(GPT2_PATH, output_hidden_states=True)
    tok = AutoTokenizer.from_pretrained(GPT2_PATH)
    tok.pad_token = tok.eos_token
    return llm, tok, llm.config.hidden_size


def load_server(ckpt_path, llm, d_llm):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["server"]
    d_in  = state["projection.0.weight"].shape[1]
    srv = ServerPipeline(
        llm=llm,
        d_llm=d_llm,
        projection=_build_projection(d_in, N_SOFT_TOKENS, d_llm),
        match_head=nn.Linear(d_llm, 1),
        n_soft_tokens=N_SOFT_TOKENS,
        freeze_llm=True,
        vae_decoder=None,
    )
    srv.load_state_dict(state)
    srv.eval()
    return srv, ckpt.get("val_metrics", {}).get("accuracy", float("nan"))


def evaluate(srv, tok, cache_dir):
    fused  = torch.load(os.path.join(cache_dir, "test_fused.pt"),  weights_only=True)
    labels = torch.load(os.path.join(cache_dir, "test_label.pt"),  weights_only=True)
    enc    = tok(INSTRUCTION, return_tensors="pt", padding=True, truncation=True, max_length=32)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    srv = srv.to(DEVICE)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(fused, labels), batch_size=BATCH_SIZE):
            x = x.to(DEVICE); B = x.size(0)
            preds = srv(x, ids.expand(B, -1).to(DEVICE), mask.expand(B, -1).to(DEVICE))
            all_logits.append(preds["match"].cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return ((logits.squeeze(-1) > 0) == (labels > 0.5)).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_batch(batch_idx):
    batch = BATCHES[batch_idx]
    print(f"\n=== Batch {batch_idx} ({len(batch)} checkpoints) ===\n")

    # Check which entries exist before loading GPT-2
    runnable = []
    for ckpt_name, cache_subdir in batch:
        ckpt_path  = os.path.join(CKPT_DIR, ckpt_name)
        cache_path = os.path.join(CKPT_DIR, cache_subdir)
        test_fused = os.path.join(cache_path, "test_fused.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP (no ckpt): {ckpt_name}")
        elif not os.path.exists(test_fused):
            print(f"  SKIP (no cache): {cache_subdir}")
        else:
            runnable.append((ckpt_name, ckpt_path, cache_path))

    if not runnable:
        print("  Nothing to evaluate in this batch.")
        return

    print(f"  Loading GPT-2 from {GPT2_PATH} ...")
    llm, tok, d_llm = load_gpt2()

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(OUT_FILE, "a") as f:
        for ckpt_name, ckpt_path, cache_path in runnable:
            tag = ckpt_name.replace("best_server_", "").replace(".pt", "")
            try:
                srv, val_acc = load_server(ckpt_path, llm, d_llm)
                test_acc     = evaluate(srv, tok, cache_path)
                line = f"  {tag:65s}  val={val_acc:.4f}  test={test_acc:.4f}"
                print(line)
                f.write(line + "\n")
                f.flush()
            except Exception as e:
                msg = f"  ERROR {tag}: {e}"
                print(msg)
                f.write(msg + "\n")
                f.flush()
            finally:
                try:
                    del srv
                except Exception:
                    pass
                gc.collect()

    print(f"\nResults appended to {OUT_FILE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=None, help="batch index to run")
    p.add_argument("--list",  action="store_true", help="list all batches and exit")
    args = p.parse_args()

    if args.list:
        for i, batch in enumerate(BATCHES):
            print(f"\nBatch {i}:")
            for ckpt_name, cache_subdir in batch:
                tag = ckpt_name.replace("best_server_", "").replace(".pt", "")
                print(f"  {tag}")
        return

    if args.batch is None:
        p.print_help()
        sys.exit(1)

    if args.batch < 0 or args.batch >= len(BATCHES):
        print(f"Error: batch index must be 0–{len(BATCHES)-1}")
        sys.exit(1)

    run_batch(args.batch)


if __name__ == "__main__":
    main()
