"""
benchmark.py — EMMA latency and accuracy benchmark on COCO test split.

Measures per-stage latency (median + p95 over N_RUNS single-sample forward passes):
  Edge  : CLIP text encoder | CLIP image encoder | mean fusion | AE/VAE encoder
  TX    : estimated transmission time at LTE / Wi-Fi / 5G uplink speeds
  Server: AE/VAE decoder | projection MLP | GPT-2 | match head

Runs all three compression modes and prints a unified comparison table.

Usage
-----
    python3 benchmark.py                        # all modes, N=200 latency runs
    python3 benchmark.py --compression ae       # single mode
    python3 benchmark.py --runs 500             # more latency samples
    python3 benchmark.py --device cpu           # force CPU
"""

import argparse
import os
import statistics
import time

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer

from emma.compression.autoencoder import AutoEncoder
from emma.compression.vae import VAE
from emma.compression.pca import PCACompressor
from emma.data.coco import COCOMatchingDataset, _stream_samples
from emma.server.pipeline import ServerPipeline, _build_projection

# ── Constants ─────────────────────────────────────────────────────────────────

CLIP_MODEL     = "openai/clip-vit-base-patch32"
D_SHARED       = 512
D_LATENT       = 64
N_SOFT_TOKENS  = 8
MAX_TEXT_LEN   = 77
N_WARMUP       = 20
INSTRUCTION    = "Does the image match the description?"

# Network uplink speeds (Mbps)
NETWORKS = {
    "LTE":  10.0,
    "WiFi": 50.0,
    "5G":  100.0,
}

# Payload bytes per sample
PAYLOAD = {
    "none": D_SHARED * 4,   # 512 floats × 4B = 2048 B
    "ae":   D_LATENT * 4,   # 64  floats × 4B =  256 B
    "vae":  D_LATENT * 4,   # 64  floats × 4B =  256 B (only μ transmitted)
    "pca":  D_LATENT * 4,   # 64  floats × 4B =  256 B
}

CHECKPOINT_DIR = "checkpoints"


# ── Timing ────────────────────────────────────────────────────────────────────

def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def timed_ms(fn, device) -> float:
    _sync(device)
    t0 = time.perf_counter()
    result = fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1_000, result

def stats(times: list) -> dict:
    s = sorted(times)
    n = len(s)
    return {
        "median": statistics.median(s),
        "p95":    s[int(0.95 * n)],
        "mean":   sum(s) / n,
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_clip(device):
    clip = CLIPModel.from_pretrained(CLIP_MODEL)
    text_model      = clip.text_model.to(device).eval()
    text_proj       = clip.text_projection.to(device).eval()
    vision_model    = clip.vision_model.to(device).eval()
    visual_proj     = clip.visual_projection.to(device).eval()
    return text_model, text_proj, vision_model, visual_proj


def load_compression(mode, device):
    """Returns (encoder_fn, None) or (None, None) for 'none' mode.
    For PCA, encoder_fn is a callable that wraps PCACompressor.encode.
    """
    if mode == "ae":
        ckpt = os.path.join(CHECKPOINT_DIR, f"ae_{D_LATENT}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"AE checkpoint not found: {ckpt}")
        ae = AutoEncoder.load(ckpt).to(device).eval()
        return ae.encoder, ae.decoder
    elif mode == "vae":
        ckpt = os.path.join(CHECKPOINT_DIR, f"vae_{D_LATENT}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"VAE checkpoint not found: {ckpt}")
        vae = VAE.load(ckpt).to(device).eval()
        return vae.encoder, vae.decoder
    elif mode == "pca":
        ckpt = os.path.join(CHECKPOINT_DIR, f"pca_{D_LATENT}")
        if not os.path.exists(ckpt + ".npz"):
            raise FileNotFoundError(f"PCA checkpoint not found: {ckpt}.npz")
        pca = PCACompressor.load(ckpt)
        # Wrap encode so it matches the same calling convention as AE/VAE
        def pca_encode(x):
            return pca.encode(x.cpu()).to(device)
        return pca_encode, None
    return None, None


def load_server(mode, device):
    llm       = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    d_llm     = llm.config.hidden_size

    d_in = D_LATENT if mode != "none" else D_SHARED
    proj = _build_projection(d_in, N_SOFT_TOKENS, d_llm).to(device)
    head = nn.Linear(d_llm, 1).to(device)

    ckpt_tag  = mode if mode == "none" else f"{mode}_{D_LATENT}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_server_{ckpt_tag}.pt")
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["server"]
        # Build a temporary ServerPipeline to load the full state dict cleanly
        server_tmp = ServerPipeline(
            llm=llm, d_llm=d_llm,
            projection=proj, match_head=head,
            n_soft_tokens=N_SOFT_TOKENS, freeze_llm=True,
        )
        server_tmp.load_state_dict(state)
        # Extract updated components
        llm  = server_tmp.llm
        proj = server_tmp.projection
        head = server_tmp.match_head
        print(f"  Loaded {ckpt_path}")
    else:
        print(f"  [warn] {ckpt_path} not found — using random weights")

    llm.eval(); proj.eval(); head.eval()
    return llm, proj, head, tokenizer, d_llm


def get_test_sample(device):
    """Load one test image-text pair (reuses the cached test split)."""
    from emma.data.coco import N_TRAIN, N_VALID, N_TEST
    proc     = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    tok      = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    raw      = _stream_samples(N_TEST, offset=N_TRAIN + N_VALID)  # uses samples_5500_500.pkl
    ds       = COCOMatchingDataset(raw[:1], max_text_len=MAX_TEXT_LEN,
                                   seed=0, processor=proc, tokenizer=tok)
    sample   = ds[0]
    return {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}


def get_test_loader(device, batch_size=64):
    from emma.data.coco import N_TRAIN, N_VALID, N_TEST
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    tok  = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    raw  = _stream_samples(N_TEST, offset=N_TRAIN + N_VALID)
    ds   = COCOMatchingDataset(raw, max_text_len=MAX_TEXT_LEN, seed=2,
                                processor=proc, tokenizer=tok)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                       shuffle=False, num_workers=0)


# ── Latency benchmark ─────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_latency(mode, sample, n_runs, device):
    print(f"\n  Loading models for '{mode}' mode...")
    text_model, text_proj, vision_model, visual_proj = load_clip(device)
    comp_enc, comp_dec = load_compression(mode, device)
    llm, proj, head, tokenizer, d_llm = load_server(mode, device)

    instr = tokenizer(INSTRUCTION, return_tensors="pt",
                      padding=True, truncation=True, max_length=32)
    instr_ids  = instr["input_ids"].to(device)
    instr_mask = instr["attention_mask"].to(device)

    t_text_enc, t_img_enc   = [], []
    t_enc_parallel, t_fusion = [], []
    t_comp_enc               = []
    t_proj, t_llm, t_head   = [], [], []

    total_runs = n_runs + N_WARMUP

    for i in range(total_runs):
        # ── Edge: sequential text encoding ───────────────────────────────────
        ms, text_out = timed_ms(lambda: text_model(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"]), device)
        text_emb = text_proj(text_out.pooler_output)
        if i >= N_WARMUP: t_text_enc.append(ms)

        # ── Edge: sequential image encoding ──────────────────────────────────
        ms, img_out = timed_ms(lambda: vision_model(
            pixel_values=sample["pixel_values"]), device)
        img_emb = visual_proj(img_out.pooler_output)
        if i >= N_WARMUP: t_img_enc.append(ms)

        # ── Edge: parallel text + image encoding ─────────────────────────────
        if device.startswith("cuda"):
            stream_text = torch.cuda.Stream()
            stream_img  = torch.cuda.Stream()
            text_out_r, img_out_r = [None], [None]
            _sync(device)
            t0 = time.perf_counter()
            with torch.cuda.stream(stream_text):
                text_out_r[0] = text_model(
                    input_ids=sample["input_ids"],
                    attention_mask=sample["attention_mask"])
            with torch.cuda.stream(stream_img):
                img_out_r[0] = vision_model(pixel_values=sample["pixel_values"])
            torch.cuda.synchronize()
            ms_par = (time.perf_counter() - t0) * 1_000
        else:
            from concurrent.futures import ThreadPoolExecutor
            _sync(device)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as ex:
                t_fut = ex.submit(lambda: text_model(
                    input_ids=sample["input_ids"],
                    attention_mask=sample["attention_mask"]))
                v_fut = ex.submit(lambda: vision_model(
                    pixel_values=sample["pixel_values"]))
                text_out_r[0], img_out_r[0] = t_fut.result(), v_fut.result()
            _sync(device)
            ms_par = (time.perf_counter() - t0) * 1_000
        if i >= N_WARMUP: t_enc_parallel.append(ms_par)

        # ── Edge: mean fusion ────────────────────────────────────────────────
        ms, fused = timed_ms(lambda: (text_emb + img_emb) / 2, device)
        if i >= N_WARMUP: t_fusion.append(ms)

        # ── Edge: compression encoder ────────────────────────────────────────
        if comp_enc is not None:
            if mode == "vae":
                ms, (mu, _) = timed_ms(lambda: comp_enc(fused), device)
                latent = mu
            else:
                ms, latent = timed_ms(lambda: comp_enc(fused), device)
            if i >= N_WARMUP: t_comp_enc.append(ms)
        else:
            latent = fused

        # ── [TX happens here] ────────────────────────────────────────────────

        # ── Server: projection MLP ───────────────────────────────────────────
        ms, soft = timed_ms(lambda: proj(latent).view(1, N_SOFT_TOKENS, d_llm), device)
        if i >= N_WARMUP: t_proj.append(ms)

        # ── Server: GPT-2 ────────────────────────────────────────────────────
        instr_emb  = llm.get_input_embeddings()(instr_ids)
        soft_mask  = torch.ones(1, N_SOFT_TOKENS, device=device, dtype=torch.long)
        inputs_emb = torch.cat([soft, instr_emb], dim=1)
        ms, gpt_out = timed_ms(lambda: llm(inputs_embeds=inputs_emb,
                                            output_hidden_states=True), device)
        cls_tok = gpt_out.hidden_states[-1][:, -1, :]   # last token
        if i >= N_WARMUP: t_llm.append(ms)

        # ── Server: match head ───────────────────────────────────────────────
        ms, _ = timed_ms(lambda: head(cls_tok), device)
        if i >= N_WARMUP: t_head.append(ms)

    results = {
        "text_enc":     stats(t_text_enc),
        "img_enc":      stats(t_img_enc),
        "enc_parallel": stats(t_enc_parallel),
        "fusion":       stats(t_fusion),
    }
    if t_comp_enc: results["comp_enc"] = stats(t_comp_enc)
    results["proj"]     = stats(t_proj)
    results["gpt2"]     = stats(t_llm)
    results["head"]     = stats(t_head)

    return results


# ── Accuracy evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(mode, loader, device):
    print(f"  Evaluating accuracy for '{mode}' mode...")
    text_model, text_proj, vision_model, visual_proj = load_clip(device)
    comp_enc, comp_dec = load_compression(mode, device)
    llm, proj, head, tokenizer, d_llm = load_server(mode, device)

    instr = tokenizer(INSTRUCTION, return_tensors="pt",
                      padding=True, truncation=True, max_length=32)
    instr_ids  = instr["input_ids"].to(device)
    instr_mask = instr["attention_mask"].to(device)

    correct, total = 0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
        B = batch["label"].size(0)

        text_out = text_model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"])
        text_emb = text_proj(text_out.pooler_output)
        img_out  = vision_model(pixel_values=batch["pixel_values"])
        img_emb  = visual_proj(img_out.pooler_output)
        fused    = (text_emb + img_emb) / 2

        if comp_enc is not None:
            if mode == "vae":
                mu, _ = comp_enc(fused)
                latent = mu
            else:
                latent = comp_enc(fused)
        else:
            latent = fused

        soft       = proj(latent).view(B, N_SOFT_TOKENS, d_llm)
        instr_emb  = llm.get_input_embeddings()(instr_ids.expand(B, -1))
        inputs_emb = torch.cat([soft, instr_emb], dim=1)
        gpt_out    = llm(inputs_embeds=inputs_emb, output_hidden_states=True)
        logits     = head(gpt_out.hidden_states[-1][:, -1, :]).squeeze(-1)

        preds   = (logits > 0).float()
        correct += (preds == batch["label"]).sum().item()
        total   += B

    return correct / total


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_latency_table(mode, lat):
    payload = PAYLOAD[mode]
    print(f"\n{'─'*65}")
    print(f"  Mode: {mode.upper()}  |  Payload: {payload}B  "
          f"({'no compression' if mode == 'none' else '8× compression'})")
    print(f"{'─'*65}")

    # Stage timing
    labels = {
        "text_enc":     "CLIP text encoder    [edge, sequential]",
        "img_enc":      "CLIP image encoder   [edge, sequential]",
        "enc_parallel": "CLIP encoders        [edge, parallel]  ",
        "fusion":       "Mean fusion          [edge]            ",
        "comp_enc":     f"{mode.upper()} encoder          [edge]            ",
        "proj":         "Projection MLP       [server]          ",
        "gpt2":         "GPT-2                [server]          ",
        "head":         "Match head           [server]          ",
    }
    print(f"  {'Stage':<35} {'Median':>8}  {'p95':>8}  {'Mean':>8}")
    print(f"  {'─'*35} {'─'*8}  {'─'*8}  {'─'*8}")

    edge_total = server_total = 0.0
    edge_stages   = {"text_enc", "img_enc", "fusion", "comp_enc"}
    server_stages = {"proj", "gpt2", "head"}

    for key, label in labels.items():
        if key not in lat: continue
        s = lat[key]
        print(f"  {label:<35} {s['median']:>7.2f}ms  {s['p95']:>7.2f}ms  {s['mean']:>7.2f}ms")
        if key in edge_stages:   edge_total   += s["median"]
        if key in server_stages: server_total += s["median"]

    # Transmission
    print(f"  {'─'*35} {'─'*8}  {'─'*8}  {'─'*8}")
    for net, mbps in NETWORKS.items():
        tx_ms = (payload * 8) / (mbps * 1e6) * 1_000
        print(f"  {'TX ' + net + f' ({mbps:.0f} Mbps)':<35} {tx_ms:>7.3f}ms")

    # Totals
    print(f"  {'─'*35} {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'Edge total (excl. TX)':<35} {edge_total:>7.2f}ms")
    print(f"  {'Server total':<35} {server_total:>7.2f}ms")
    tx_wifi = (payload * 8) / (NETWORKS["WiFi"] * 1e6) * 1_000
    print(f"  {'End-to-end (WiFi TX)':<35} {edge_total + tx_wifi + server_total:>7.2f}ms")


def print_summary_table(results):
    print(f"\n{'═'*70}")
    print(f"  EMMA — Summary Comparison Table")
    print(f"{'═'*70}")
    print(f"  {'Mode':<8} {'Payload':>8}  {'BW Reduc':>9}  {'Accuracy':>9}  "
          f"{'E2E WiFi':>9}  {'E2E LTE':>9}  {'E2E 5G':>9}")
    print(f"  {'─'*8} {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")

    for mode, (lat, acc) in results.items():
        payload  = PAYLOAD[mode]
        bw_red   = PAYLOAD["none"] / payload
        edge_t   = sum(lat[k]["median"] for k in ["text_enc", "img_enc", "fusion"]
                       + (["comp_enc"] if "comp_enc" in lat else []))
        server_t = sum(lat[k]["median"] for k in ["proj","gpt2","head"]
                       + (["comp_dec"] if "comp_dec" in lat else []))
        e2e = {net: edge_t + (payload*8)/(mbps*1e6)*1000 + server_t
               for net, mbps in NETWORKS.items()}

        print(f"  {mode.upper():<8} {payload:>7}B  {bw_red:>8.0f}×  "
              f"{acc:>8.1%}  "
              f"{e2e['WiFi']:>8.1f}ms  {e2e['LTE']:>8.1f}ms  {e2e['5G']:>8.1f}ms")

    print(f"  {'Random':<8} {'—':>8}  {'—':>9}  {'50.0%':>9}")
    print(f"{'═'*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", choices=["none","ae","vae","pca"],
                        default=None, help="single mode (default: run all)")
    parser.add_argument("--latent-dim", type=int, default=64,
                        help="compression bottleneck size (default: 64)")
    parser.add_argument("--runs",   type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    modes  = [args.compression] if args.compression else ["none", "ae", "vae", "pca"]

    # Override global D_LATENT from CLI
    global D_LATENT
    D_LATENT = args.latent_dim

    print(f"EMMA Benchmark  |  device={device}  |  latency_runs={args.runs}")

    print("\nLoading test data...")
    loader = get_test_loader(device, batch_size=args.batch_size)
    sample = get_test_sample(device)
    print(f"  {len(loader.dataset)} test pairs loaded")

    all_results = {}
    for mode in modes:
        print(f"\n{'━'*65}")
        print(f"  Benchmarking: {mode.upper()}")
        print(f"{'━'*65}")

        lat = benchmark_latency(mode, sample, args.runs, device)
        acc = evaluate_accuracy(mode, loader, device)
        all_results[mode] = (lat, acc)

        print_latency_table(mode, lat)
        print(f"\n  Test accuracy: {acc:.4f} ({acc:.1%})")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
