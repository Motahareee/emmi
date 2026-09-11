"""
Quantization baseline for EMMI.

Tests scalar quantization of the full 2048-dim match-fused embedding
at different bit widths (INT8, INT4, INT2, INT1) as a compression baseline.

No retraining — uses the existing none (no compression) server checkpoint.
Quantization is applied at test time only: quantize → transmit → dequantize → server.

Usage:
    python3 eval_quantization.py
    python3 eval_quantization.py --encoder mobileclip --fusion match
    python3 eval_quantization.py --checkpoint-dir /path/to/checkpoints
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--encoder",        default="mobileclip", choices=["clip", "mobileclip"])
parser.add_argument("--fusion",         default="match",      choices=["mean", "concat", "match"])
parser.add_argument("--checkpoint-dir", default="checkpoints")
parser.add_argument("--batch-size",     type=int, default=64)
parser.add_argument("--n-train",        type=int, default=72000)
parser.add_argument("--n-valid",        type=int, default=5000)
parser.add_argument("--n-test",         type=int, default=5000)
args = parser.parse_args()

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_SOFT    = 8
INSTR     = "Does the image match the description?"

_HF_CACHE = (
    "/scratch/user/motahare/hf_cache/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
GPT2_PATH = _HF_CACHE if os.path.isdir(_HF_CACHE) else "gpt2"

# ── Locate embed cache ────────────────────────────────────────────────────────

_enc = "" if args.encoder == "clip" else f"_{args.encoder}"
_fus = "" if args.fusion  == "mean" else f"_{args.fusion}"
CACHE_DIR = os.path.join(
    args.checkpoint_dir,
    f"embed_cache{_enc}{_fus}_{args.n_train}_{args.n_valid}_{args.n_test}"
)

print(f"Encoder: {args.encoder}  Fusion: {args.fusion}")
print(f"Cache:   {CACHE_DIR}")
print(f"Device:  {DEVICE}")

test_fused  = torch.load(os.path.join(CACHE_DIR, "test_fused.pt"),  weights_only=True)
test_labels = torch.load(os.path.join(CACHE_DIR, "test_label.pt"),  weights_only=True)
D_FUSED = test_fused.shape[1]
print(f"Test set: {len(test_fused)} samples, d={D_FUSED}\n")


# ── Quantization functions ────────────────────────────────────────────────────

def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Symmetric min-max scalar quantization.
    Quantizes each sample independently to `bits` bits.
    Returns dequantized float tensor (simulates transmit → receive).
    """
    levels = 2 ** bits
    x_min  = x.min(dim=1, keepdim=True).values
    x_max  = x.max(dim=1, keepdim=True).values
    scale  = (x_max - x_min).clamp(min=1e-8) / (levels - 1)
    q      = ((x - x_min) / scale).round().clamp(0, levels - 1)
    return q * scale + x_min   # dequantize


def payload_bytes(d: int, bits: int) -> int:
    """Bytes to transmit d values at `bits` bits each (+ 2×4 bytes for min/max per sample)."""
    return (d * bits + 7) // 8 + 8   # +8 for min/max scale factors


# ── Load server ───────────────────────────────────────────────────────────────

_enc_tag = "" if args.encoder == "clip" else f"_{args.encoder}"
_fus_tag = "" if args.fusion  == "mean" else f"_{args.fusion}"
ckpt_tag = f"none{_enc_tag}{_fus_tag}"
ckpt_path = os.path.join(args.checkpoint_dir, f"best_server_{ckpt_tag}.pt")

print(f"Loading server checkpoint: {ckpt_path}")
print(f"Loading GPT-2 from {GPT2_PATH} ...")

llm = AutoModelForCausalLM.from_pretrained(GPT2_PATH, output_hidden_states=True).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(GPT2_PATH)
tok.pad_token = tok.eos_token
d_llm = llm.config.hidden_size   # 768

from emma.server.pipeline import _build_projection, ServerPipeline

proj  = _build_projection(D_FUSED, N_SOFT, d_llm)
head  = nn.Linear(d_llm, 1)

server = ServerPipeline(
    llm=llm, d_llm=d_llm,
    projection=proj, match_head=head,
    n_soft_tokens=N_SOFT, freeze_llm=True,
    vae_decoder=None,
).to(DEVICE).eval()

ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
server.load_state_dict(ckpt["server"])
print("Server loaded.\n")

enc      = tok(INSTR, return_tensors="pt", padding=True, truncation=True, max_length=32)
instr_ids  = enc["input_ids"].to(DEVICE)
instr_mask = enc["attention_mask"].to(DEVICE)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(fused: torch.Tensor, labels: torch.Tensor) -> float:
    loader = DataLoader(TensorDataset(fused, labels), batch_size=args.batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            B = x.size(0)
            ids  = instr_ids.expand(B, -1)
            mask = instr_mask.expand(B, -1)
            out  = server(x, ids, mask)
            preds = (out["match"].squeeze(-1) > 0).cpu()
            correct += (preds == (y > 0.5)).sum().item()
            total   += B
    return correct / total


# ── Baseline: no quantization ─────────────────────────────────────────────────

print("=" * 60)
print(f"{'Method':<30} {'Accuracy':>10} {'Payload':>12} {'Reduction':>12}")
print("=" * 60)

base_bytes = D_FUSED * 4   # float32
base_acc   = evaluate(test_fused, test_labels)
print(f"{'No compression (float32)':<30} {base_acc*100:>9.2f}%  {base_bytes:>9,} B  {'1.0×':>10}")

# ── Quantization sweep ────────────────────────────────────────────────────────

for bits in [16, 8, 4, 2, 1]:
    q_fused  = quantize(test_fused, bits)
    acc      = evaluate(q_fused, test_labels)
    p_bytes  = payload_bytes(D_FUSED, bits)
    reduction = base_bytes / p_bytes
    label    = f"Quantize INT{bits}"
    print(f"  {label:<28} {acc*100:>9.2f}%  {p_bytes:>9,} B  {reduction:>9.1f}×")

print("=" * 60)
print(f"\nNote: payload includes 8 bytes for per-sample min/max scale factors.")
print(f"No compression baseline: {base_bytes:,} B ({D_FUSED}-dim float32)")
