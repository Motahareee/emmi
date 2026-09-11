"""
Complete EMMA pipeline latency benchmark.

Measures wall-clock time for every stage of the EMMA pipeline. Edge stages
(1-3) run on a single CPU thread — a proxy for a constrained edge device.
Server-side inference (4a/4b) runs on GPU when available, since a 7B-param
model's CPU latency isn't a meaningful stand-in for real server hardware.

Stages timed:
  1. Encoders   — CLIP ViT-B/32 vs MobileCLIP2-S0 (image / text / both)
  2. Fusion     — mean / concat / match
  3. Compression encoder (edge-side only):
                  PCA-64, AE-64, ContrastiveAE-64, BlockPCA-64  (match fusion)
  4a. Server    — GPT-2 inference (soft-prompt projection + LLM + match head)
  4b. Server    — LLaVA-1.5-7B backbone inference (same, real 7B backbone)
  5. Transmission — payload bytes / bandwidth  (mathematical, not empirical)

Usage:
    python3 benchmark_latency.py
    python3 benchmark_latency.py --n-runs 200 --warmup 30
    python3 benchmark_latency.py --skip-mobileclip   # if open_clip not installed
    python3 benchmark_latency.py --skip-llava         # GPT-2 only
    python3 benchmark_latency.py --load-in-8bit        # 8-bit LLaVA backbone
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--n-runs",  type=int, default=200)
parser.add_argument("--warmup",  type=int, default=30)
parser.add_argument("--threads", type=int, default=1,
                    help="CPU threads (1 = single-core edge simulation)")
parser.add_argument("--skip-mobileclip", action="store_true")
parser.add_argument("--skip-server",     action="store_true")
parser.add_argument("--skip-llava",      action="store_true")
parser.add_argument("--load-in-8bit",    action="store_true",
                    help="load LLaVA's language_model backbone with bitsandbytes 8-bit quantization")
args = parser.parse_args()

torch.set_num_threads(args.threads)
torch.set_grad_enabled(False)
DEVICE = "cpu"
# Edge stages (1-3) stay on CPU as a proxy for a constrained device. Section 4
# (server-side inference) runs on GPU when available -- a 7B-parameter model
# on CPU says nothing about realistic server latency, and this was previously
# an inconsistency even for GPT-2 (server-side has no reason to be CPU-only).
SERVER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_HF_CACHE = (
    "/scratch/user/motahare/hf_cache/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
GPT2_PATH = _HF_CACHE if os.path.isdir(_HF_CACHE) else "gpt2"

N_SOFT_TOKENS = 8
INSTRUCTION   = "Does the image match the description?"

print(f"Threads: {args.threads}  |  warm-up: {args.warmup}  |  runs: {args.n_runs}")
print(f"Device:  {DEVICE}")
try:
    import cpuinfo
    print(f"CPU:     {cpuinfo.get_cpu_info()['brand_raw']}")
except Exception:
    import platform
    print(f"CPU:     {platform.processor()}")
print()


# ── Timing helper ─────────────────────────────────────────────────────────────

def timeit(fn, warmup, n_runs):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return a.mean(), a.std()


def row(label, mean_ms, std_ms):
    print(f"  {label:<50s}  {mean_ms:8.3f} ± {std_ms:.3f} ms")


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ── 1. ENCODERS ───────────────────────────────────────────────────────────────

section("1. ENCODER FORWARD PASS  (batch=1, single-thread CPU)")

dummy_img_clip = torch.randn(1, 3, 224, 224)
dummy_img_mc   = torch.randn(1, 3, 256, 256)
dummy_ids      = torch.zeros(1, 77, dtype=torch.long)
dummy_mask     = torch.ones(1, 77, dtype=torch.long)

# CLIP
try:
    from transformers import CLIPModel
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()

    m, s = timeit(lambda: clip.get_image_features(pixel_values=dummy_img_clip),
                  args.warmup, args.n_runs)
    row("CLIP ViT-B/32 — image encoder", m, s)

    m, s = timeit(lambda: clip.get_text_features(input_ids=dummy_ids,
                                                   attention_mask=dummy_mask),
                  args.warmup, args.n_runs)
    row("CLIP ViT-B/32 — text encoder", m, s)

    m, s = timeit(lambda: (clip.get_image_features(pixel_values=dummy_img_clip),
                            clip.get_text_features(input_ids=dummy_ids,
                                                    attention_mask=dummy_mask)),
                  args.warmup, args.n_runs)
    row("CLIP ViT-B/32 — both encoders", m, s)
    del clip
except Exception as e:
    print(f"  [CLIP] skipped: {e}")

# MobileCLIP
if not args.skip_mobileclip:
    try:
        import open_clip
        mc, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP2-S0", pretrained="dfndr2b")
        mc = mc.to(DEVICE).eval()

        m, s = timeit(lambda: mc.encode_image(dummy_img_mc, normalize=False),
                      args.warmup, args.n_runs)
        row("MobileCLIP2-S0 — image encoder", m, s)

        m, s = timeit(lambda: mc.encode_text(dummy_ids, normalize=False),
                      args.warmup, args.n_runs)
        row("MobileCLIP2-S0 — text encoder", m, s)

        m, s = timeit(lambda: (mc.encode_image(dummy_img_mc, normalize=False),
                                mc.encode_text(dummy_ids, normalize=False)),
                      args.warmup, args.n_runs)
        row("MobileCLIP2-S0 — both encoders", m, s)
        del mc
    except Exception as e:
        print(f"  [MobileCLIP] skipped: {e}")


# ── 2. FUSION ─────────────────────────────────────────────────────────────────

section("2. FUSION  (batch=1, 512-dim embeddings)")

t = torch.randn(1, 512)
v = torch.randn(1, 512)

for label, fn in [
    ("mean   (512-dim  → 512-dim)",  lambda: (t + v) / 2),
    ("concat (512-dim  → 1024-dim)", lambda: torch.cat([t, v], dim=-1)),
    ("match  (512-dim  → 2048-dim)", lambda: torch.cat([t, v, (t-v).abs(), t*v], dim=-1)),
]:
    m, s = timeit(fn, args.warmup, args.n_runs)
    row(f"Fusion: {label}", m, s)


# ── 3. COMPRESSION ENCODER (edge-side, match fusion 2048→64) ─────────────────

section("3. COMPRESSION ENCODER HALF  (match fusion, 2048→64, batch=1)")

x_match = torch.randn(1, 2048)
x_np    = x_match.numpy()

# PCA: matrix multiply
pca_components = np.random.randn(64, 2048).astype(np.float32)
pca_mean       = np.random.randn(2048).astype(np.float32)

def pca_encode():
    return (x_np - pca_mean) @ pca_components.T

m, s = timeit(pca_encode, args.warmup, args.n_runs)
row("PCA-64          (matmul, numpy)", m, s)

# LDA: same closed-form projection as PCA (mean-center + linear projection),
# just with a different fitted direction -- timing is architecture-driven,
# not value-driven, so real (unfitted) mean/components of the right shape
# give an accurate latency measurement without needing an actual .fit() call.
from emma.compression.lda import LDACompressor
lda = LDACompressor(n_components=64)
lda.mean_       = np.random.randn(2048).astype(np.float32)
lda.components_ = np.random.randn(64, 2048).astype(np.float32)

m, s = timeit(lambda: lda.encode(x_match), args.warmup, args.n_runs)
row("LDA-64          (matmul, numpy)", m, s)

# AE encoder: MLP 2048→1024→512→256→128→64
def _ae_encoder(d_in, d_out=64):
    layers, d = [], d_in
    while d > 128:
        layers += [nn.Linear(d, d // 2), nn.ReLU()]
        d //= 2
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers).eval()

ae_enc = _ae_encoder(2048)
m, s = timeit(lambda: ae_enc(x_match), args.warmup, args.n_runs)
row("AE-64 encoder   (MLP 2048→…→64)", m, s)

# ContrastiveAE: same MLP architecture as AE (contrastive loss only at train time)
contrastive_enc = _ae_encoder(2048)
m, s = timeit(lambda: contrastive_enc(x_match), args.warmup, args.n_runs)
row("ContrastiveAE-64 encoder (same MLP as AE)", m, s)

# VAE encoder: real VAEEncoder class (not an approximation like the plain-MLP
# rows above) -- has two output heads (mu, log_var) vs AE's one, so it's
# worth timing precisely rather than assuming it matches AE's latency.
# .encode() is the actual inference path (mu only, no sampling).
from emma.compression.vae import VAEEncoder
vae_enc = VAEEncoder(d_in=2048, d_latent=64, hidden_dims=(256, 128)).eval()
m, s = timeit(lambda: vae_enc.encode(x_match), args.warmup, args.n_runs)
row("VAE-64 encoder  (2-head MLP, mu only)", m, s)

# BlockPCA: 4 × PCA(16) on each 512-dim block
block_comps = [np.random.randn(16, 512).astype(np.float32) for _ in range(4)]
block_means = [np.random.randn(512).astype(np.float32) for _ in range(4)]

def blockpca_encode():
    x = x_np[0]  # shape (2048,)
    parts = [x[i*512:(i+1)*512] for i in range(4)]
    return np.concatenate([(p - block_means[i]) @ block_comps[i].T
                           for i, p in enumerate(parts)])

m, s = timeit(blockpca_encode, args.warmup, args.n_runs)
row("BlockPCA-64     (4 × PCA-16, numpy)", m, s)


# ── 4. SERVER-SIDE INFERENCE ──────────────────────────────────────────────────

if not args.skip_server:
    section(f"4a. SERVER-SIDE INFERENCE  (GPT-2, batch=1, {SERVER_DEVICE.upper()})")

    try:
        from emma.server.pipeline import ServerPipeline, _build_projection

        print(f"  Loading GPT-2 from {GPT2_PATH} ...")
        llm = AutoModelForCausalLM.from_pretrained(
            GPT2_PATH, output_hidden_states=True).to(SERVER_DEVICE).eval()
        tok = AutoTokenizer.from_pretrained(GPT2_PATH)
        tok.pad_token = tok.eos_token
        d_llm = llm.config.hidden_size  # 768

        enc   = tok(INSTRUCTION, return_tensors="pt",
                    padding=True, truncation=True, max_length=32)
        ids   = enc["input_ids"].to(SERVER_DEVICE)
        mask  = enc["attention_mask"].to(SERVER_DEVICE)

        for label, d_in in [
            ("Server (no compression,  d_in=512)",  512),
            ("Server (PCA/AE/CAE-64,   d_in=64)",   64),
        ]:
            proj   = _build_projection(d_in, N_SOFT_TOKENS, d_llm).to(SERVER_DEVICE)
            head   = nn.Linear(d_llm, 1).to(SERVER_DEVICE)
            server = ServerPipeline(
                llm=llm, d_llm=d_llm,
                projection=proj, match_head=head,
                n_soft_tokens=N_SOFT_TOKENS, freeze_llm=True,
                vae_decoder=None,
            ).eval()

            x_srv = torch.randn(1, d_in, device=SERVER_DEVICE)
            if SERVER_DEVICE == "cuda":
                torch.cuda.synchronize()
            m, s = timeit(lambda srv=server, x=x_srv: srv(x, ids, mask),
                          args.warmup, args.n_runs)
            row(label, m, s)
            del server, proj, head

        del llm
        if SERVER_DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [Server] skipped: {e}")


# ── 4b. SERVER-SIDE INFERENCE — LLaVA ─────────────────────────────────────────
# Same methodology as 4a: real pretrained backbone, freshly-initialized
# projection/match_head (weight values don't affect forward-pass timing,
# only shape/dtype/quantization do). d_in values match what was actually
# trained for this project (mobileclip + match fusion): 2048 = raw fused
# embedding (no compression), 64 = ContrastiveAE/CrossModalAE latent.

if not args.skip_server and not args.skip_llava:
    section(f"4b. SERVER-SIDE INFERENCE  (LLaVA-1.5-7B backbone, batch=1, "
            f"{SERVER_DEVICE.upper()}, 8bit={args.load_in_8bit})")

    if SERVER_DEVICE != "cuda":
        print("  [LLaVA] skipped: no GPU available -- a 7B model on CPU isn't a "
              "meaningful server-latency number.")
    else:
        try:
            from emma.server.pipeline import ServerPipeline, _build_projection, _load_llm

            print(f"  Loading LLaVA-1.5-7B language_model backbone "
                  f"(load_in_8bit={args.load_in_8bit}) ...")
            llm, d_llm = _load_llm("llava", load_in_8bit=args.load_in_8bit)
            if not args.load_in_8bit:
                llm = llm.to(SERVER_DEVICE)
            llm.eval()

            tok = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
            tok.pad_token = tok.eos_token

            enc  = tok(INSTRUCTION, return_tensors="pt",
                      padding=True, truncation=True, max_length=32)
            ids  = enc["input_ids"].to(SERVER_DEVICE)
            mask = enc["attention_mask"].to(SERVER_DEVICE)

            for label, d_in in [
                ("Server (no compression,      d_in=2048)", 2048),
                ("Server (ContrastiveAE/CrossModalAE-64, d_in=64)", 64),
            ]:
                proj   = _build_projection(d_in, N_SOFT_TOKENS, d_llm).to(SERVER_DEVICE)
                head   = nn.Linear(d_llm, 1).to(SERVER_DEVICE)
                server = ServerPipeline(
                    llm=llm, d_llm=d_llm,
                    projection=proj, match_head=head,
                    n_soft_tokens=N_SOFT_TOKENS, freeze_llm=True,
                    vae_decoder=None,
                ).eval()

                x_srv = torch.randn(1, d_in, device=SERVER_DEVICE)
                torch.cuda.synchronize()
                m, s = timeit(lambda srv=server, x=x_srv: srv(x, ids, mask),
                              args.warmup, args.n_runs)
                row(label, m, s)
                del server, proj, head

            del llm
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [LLaVA] skipped: {e}")


# ── 5. TRANSMISSION LATENCY (mathematical) ───────────────────────────────────

section("5. TRANSMISSION LATENCY  (payload / bandwidth, mathematical)")

PAYLOADS = {
    "match, no compress (2048-dim float32)": 2048 * 4,
    "concat, no compress (1024-dim float32)": 1024 * 4,
    "mean,   no compress ( 512-dim float32)":  512 * 4,
    "compressed     (64-dim float32, all methods)":   64 * 4,
}

BANDWIDTHS = {
    "IoT / LTE-M   (  0.1 Mbps)":   0.1e6,
    "LTE           ( 10.0 Mbps)":  10.0e6,
    "WiFi          ( 50.0 Mbps)":  50.0e6,
    "5G            (100.0 Mbps)": 100.0e6,
}

print()
header = f"  {'Payload':<45s}" + "".join(f"  {k[:10]:>12s}" for k in BANDWIDTHS)
print(header)
print("  " + "-" * (45 + 14 * len(BANDWIDTHS)))
for payload_label, n_bytes in PAYLOADS.items():
    cells = []
    for bw in BANDWIDTHS.values():
        ms = (n_bytes * 8 / bw) * 1000
        cells.append(f"  {ms:>12.2f}")
    print(f"  {payload_label:<45s}" + "".join(cells))
print()
print("  Units: ms")
print("  Compression ratio (match): 8192 B → 256 B = 32× reduction")


# ── 6. END-TO-END SUMMARY ─────────────────────────────────────────────────────

section("6. END-TO-END SUMMARY  (MC + match fusion, 1 Mbps uplink, estimated)")

print("""
  ┌──────────────────────────────────────┬────────────┬────────────┐
  │ Stage                                │  No Compr. │  PCA-64    │
  ├──────────────────────────────────────┼────────────┼────────────┤
  │ Edge: both encoders (MC)             │  ~187.7 ms │  ~187.7 ms │
  │ Edge: match fusion                   │   ~0.01 ms │   ~0.01 ms │
  │ Edge: compression (PCA)              │   ~0.00 ms │   ~0.01 ms │
  │ Transmission @ 0.1 Mbps             │  ~655.4 ms │   ~20.5 ms │
  │ Transmission @ 10  Mbps             │    ~6.6 ms │    ~0.2 ms │
  │ Server: GPT-2 inference              │  see §4    │  see §4    │
  ├──────────────────────────────────────┼────────────┼────────────┤
  │ Edge + TX savings (0.1 Mbps)         │  baseline  │  ~634.9 ms │
  └──────────────────────────────────────┴────────────┴────────────┘

  Note: TX savings dominate on bandwidth-constrained links (IoT, LTE-M).
  On WiFi/5G the encoder forward pass dominates regardless of compression.
""")

print("Done.")
