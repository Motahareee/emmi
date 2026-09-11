"""
Test-set evaluation for trained server checkpoints.

Loads a saved server checkpoint + its corresponding latent cache,
runs inference on the test split, and reports accuracy.

Usage (single checkpoint):
    python3 eval_test.py --ckpt checkpoints/best_server_contrastiveae_64_72k_mobileclip_match.pt \
                         --cache checkpoints/embed_cache_mobileclip_match_72000_5000_5000/contrastiveae_latents_64

Usage (batch — evaluate all checkpoints matching a pattern):
    python3 eval_test.py --batch

The batch mode discovers checkpoints and infers cache dirs automatically
based on the naming conventions used in train.py.
"""

import os
import re
import gc
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from emma.server.pipeline import ServerPipeline, _build_projection

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION = "Does the image match the description?"
N_SOFT_TOKENS = 8
BATCH_SIZE    = 256

# Local HuggingFace cache path for GPT-2 (cluster has no internet)
_HF_CACHE = (
    "/scratch/user/motahare/hf_cache/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
GPT2_PATH = _HF_CACHE if os.path.isdir(_HF_CACHE) else "gpt2"


def load_gpt2():
    """Load GPT-2 once and return (llm, tokenizer, d_llm)."""
    llm       = AutoModelForCausalLM.from_pretrained(GPT2_PATH, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(GPT2_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    d_llm = llm.config.hidden_size
    return llm, tokenizer, d_llm


def load_server_from_ckpt(ckpt_path: str, llm, d_llm) -> ServerPipeline:
    """Rebuild server and load weights from checkpoint. Reuses a pre-loaded LLM."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["server"]

    # Infer d_server_in from the projection weight shape: Linear(d_in, d_llm)
    d_server_in = state["projection.0.weight"].shape[1]

    server = ServerPipeline(
        llm=llm,
        d_llm=d_llm,
        projection=_build_projection(d_server_in, N_SOFT_TOKENS, d_llm),
        match_head=nn.Linear(d_llm, 1),
        n_soft_tokens=N_SOFT_TOKENS,
        freeze_llm=True,
        vae_decoder=None,
    )
    server.load_state_dict(state)
    server.eval()
    return server


def evaluate(server, tokenizer, cache_dir: str) -> float:
    """Run test split from cache_dir through server, return accuracy."""
    fused  = torch.load(os.path.join(cache_dir, "test_fused.pt"),  weights_only=True)
    labels = torch.load(os.path.join(cache_dir, "test_label.pt"),  weights_only=True)

    instr_enc  = tokenizer(INSTRUCTION, return_tensors="pt",
                           padding=True, truncation=True, max_length=32)
    instr_ids  = instr_enc["input_ids"]
    instr_mask = instr_enc["attention_mask"]

    loader = DataLoader(TensorDataset(fused, labels), batch_size=BATCH_SIZE)
    all_logits, all_labels = [], []

    server = server.to(DEVICE)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            B = x.size(0)
            ids  = instr_ids.expand(B, -1).to(DEVICE)
            mask = instr_mask.expand(B, -1).to(DEVICE)
            preds = server(x, ids, mask)
            all_logits.append(preds["match"].cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    acc = ((logits.squeeze(-1) > 0) == (labels > 0.5)).float().mean().item()
    return acc


def infer_cache_dir(ckpt_path: str, checkpoint_dir: str) -> str:
    """
    Given a checkpoint path like:
      checkpoints/best_server_contrastiveae_64_72k_mobileclip_match_s1.pt
    infer the latent cache directory.

    Naming convention from train.py:
      best_server_{compression}_{d_latent}{_scale}{_enc}{_fus}{_variant}{_dos_tag}{_seed}.pt
    """
    name = os.path.basename(ckpt_path).replace("best_server_", "").replace(".pt", "")

    # Parse encoder
    enc = "mobileclip" if "_mobileclip" in name else "clip"
    _enc = "" if enc == "clip" else "_mobileclip"

    # Parse fusion
    fus = "mean"
    _fus = ""
    for f in ("match", "concat"):
        if f"_{f}" in name:
            fus = f
            _fus = f"_{f}"
            break

    # Parse scale (n_train)
    m = re.search(r"_(\d+)k", name)
    n_train = int(m.group(1)) * 1000 if m else 5000
    n_valid = 5000
    n_test  = 5000

    embed_cache = os.path.join(
        checkpoint_dir,
        f"embed_cache{_enc}{_fus}_{n_train}_{n_valid}_{n_test}"
    )

    # Parse seed
    m_seed = re.search(r"_s(\d+)$", name)
    seed = int(m_seed.group(1)) if m_seed else 0
    _seed = f"_s{seed}" if seed != 0 else ""

    # Parse decode-on-server
    decoded = "_decoded" in name
    _dos = "_decoded" if decoded else ""

    # Parse compression type and latent dim
    compression = name.split("_")[0]  # first token is compression type
    m_dim = re.search(r"_(\d+)_", name)
    d_latent = int(m_dim.group(1)) if m_dim else 64

    if compression == "none":
        return embed_cache

    latent_subdir = f"{compression}_latents_{d_latent}{_seed}{_dos}"
    return os.path.join(embed_cache, latent_subdir)


def batch_eval(checkpoint_dir: str):
    """Evaluate all best_server_*.pt checkpoints found in checkpoint_dir."""
    pattern = os.path.join(checkpoint_dir, "best_server_*.pt")
    ckpts   = sorted(glob.glob(pattern))

    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(ckpts)} checkpoints — loading GPT-2 once...\n")
    llm, tokenizer, d_llm = load_gpt2()
    results = []
    out_file = os.path.join(checkpoint_dir, "test_results.txt")

    with open(out_file, "w") as f:
        f.write(f"{'Tag':60s}  {'Val':>8}  {'Test':>8}\n")
        f.write("-" * 82 + "\n")

        for ckpt_path in ckpts:
            tag = os.path.basename(ckpt_path).replace("best_server_", "").replace(".pt", "")
            cache_dir = infer_cache_dir(ckpt_path, checkpoint_dir)

            if not os.path.exists(os.path.join(cache_dir, "test_fused.pt")):
                print(f"  SKIP {tag}")
                continue

            try:
                server  = load_server_from_ckpt(ckpt_path, llm, d_llm)
                acc     = evaluate(server, tokenizer, cache_dir)
                val_acc = torch.load(ckpt_path, map_location="cpu",
                                     weights_only=False)["val_metrics"]["accuracy"]
                line = f"  {tag:60s}  val={val_acc:.4f}  test={acc:.4f}"
                print(line)
                f.write(line + "\n")
                f.flush()
                results.append((tag, val_acc, acc))
            except Exception as e:
                print(f"  ERROR {tag}: {e}")
            finally:
                del server
                gc.collect()

    print(f"\nResults saved to {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",           default=None, help="single checkpoint path")
    p.add_argument("--cache",          default=None, help="latent cache dir (for single mode)")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--batch",          action="store_true", help="evaluate all checkpoints")
    args = p.parse_args()

    if args.batch:
        batch_eval(args.checkpoint_dir)
    elif args.ckpt:
        cache_dir = args.cache or infer_cache_dir(args.ckpt, args.checkpoint_dir)
        print(f"Checkpoint : {args.ckpt}")
        print(f"Cache dir  : {cache_dir}")
        llm, tokenizer, d_llm = load_gpt2()
        server = load_server_from_ckpt(args.ckpt, llm, d_llm)
        acc = evaluate(server, tokenizer, cache_dir)
        val_acc = torch.load(args.ckpt, map_location="cpu",
                             weights_only=False)["val_metrics"]["accuracy"]
        print(f"Val acc  : {val_acc:.4f}")
        print(f"Test acc : {acc:.4f}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
