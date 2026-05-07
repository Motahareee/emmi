"""
benchmark.py — Evaluate EMMA (no compression) on the CMU-MOSEI test set.

Reports
-------
  Task metrics  — sentiment MAE, binary accuracy, emotion accuracy on test split
  Latency       — edge forward, server forward, end-to-end
                  (median + p95 over N_LATENCY_RUNS single-sample runs)

Checkpoint layout expected (produced by train.py):
    checkpoints/best_stage2.pt   — edge + server state dicts (after alignment)
    checkpoints/best_stage3.pt   — server state dict (after server fine-tune)
    checkpoints/embed_cache/     — cached fused embeddings (if available)

Usage
-----
    python benchmark.py                    # DEBUG=True, use GPT-2
    python benchmark.py --no-debug         # real LLM from config
    python benchmark.py --edge  checkpoints/best_stage2.pt
    python benchmark.py --server checkpoints/best_stage3.pt
    python benchmark.py --latency-runs 100
"""

import argparse
import os
import statistics
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from emma.config import (
    AlignmentConfig, EMMAConfig, EncoderConfig, ServerConfig, VAEConfig,
)
from emma.data.mosei import MOSEIDataset, mosei_collate_fn
from emma.pipeline import EdgePipeline
from emma.server.pipeline import ServerPipeline, _build_projection

# ── Defaults (must match train.py) ────────────────────────────────────────────

DEBUG             = True
BATCH_SIZE        = 16          # for metric evaluation
MAX_TEXT_LEN      = 128
MAX_AUDIO_SAMPLES = 132300
N_SOFT_TOKENS     = 8
N_LATENCY_RUNS    = 200         # single-sample runs for latency histogram
INSTRUCTION       = "Analyze sentiment and emotions:"

cfg = EMMAConfig(
    encoder=EncoderConfig(text_freeze_base=True, audio_freeze_base=True),
    alignment=AlignmentConfig(d_shared=256),
    vae=VAEConfig(d_latent=64),
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
    return EdgePipeline(
        config=cfg,
        text_encoder=None,
        audio_encoder=None,
    )


def build_server(cfg: EMMAConfig, debug: bool):
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
            {"llava":      "llava-hf/llava-1.5-7b-hf",
             "qwen_audio": "Qwen/Qwen2-Audio-7B-Instruct",
             "plain_llm":  "mistralai/Mistral-7B-v0.1"}[sc.scenario]
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


# ── Timing helpers ────────────────────────────────────────────────────────────

def _sync():
    """Synchronise CUDA stream so timing is accurate on GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_ms(fn) -> float:
    """Run fn() and return wall time in milliseconds (GPU-safe)."""
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    return (time.perf_counter() - t0) * 1_000


def latency_stats(times_ms: list[float]) -> dict:
    s = sorted(times_ms)
    n = len(s)
    return {
        "median_ms": statistics.median(s),
        "p95_ms":    s[int(0.95 * n)],
        "min_ms":    s[0],
        "max_ms":    s[-1],
    }


# ── Task metric evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_metrics(edge, server, loader, instr_ids, instr_mask) -> dict:
    """Full-dataset evaluation using the live edge pipeline."""
    edge.eval()
    server.eval()

    sent_preds, sent_tgts, emo_preds, emo_tgts = [], [], [], []
    total_loss = 0.0
    n_batches  = 0

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
        preds = server(fused, instr_ids_b, instr_mask_b)

        s_loss = torch.nn.functional.mse_loss(
            preds["sentiment"].squeeze(-1), batch["sentiment"])
        e_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            preds["emotions"], batch["emotions"])
        total_loss += (s_loss + e_loss).item()

        sent_preds.append(preds["sentiment"].cpu())
        sent_tgts.append(batch["sentiment"].cpu())
        emo_preds.append(preds["emotions"].cpu())
        emo_tgts.append(batch["emotions"].cpu())
        n_batches += 1

    sp = torch.cat(sent_preds).squeeze(-1)
    st = torch.cat(sent_tgts)
    ep = torch.cat(emo_preds)
    et = torch.cat(emo_tgts)

    return {
        "test_loss":     total_loss / n_batches,
        "sentiment_mae": (sp - st).abs().mean().item(),
        "sentiment_acc": ((sp > 0) == (st > 0)).float().mean().item(),
        "emotion_acc":   ((torch.sigmoid(ep) > 0.5) == et.bool()).float().mean().item(),
        "n_samples":     len(sp),
    }


# ── Latency benchmark ─────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_latency(edge, server, sample_batch, instr_ids, instr_mask,
                      n_runs: int = N_LATENCY_RUNS) -> dict:
    """
    Time edge and server forward passes independently on a single sample.
    Runs n_runs times (first 10 are warmup, excluded from stats).
    """
    edge.eval()
    server.eval()
    WARMUP = min(10, n_runs // 5)

    # Single sample from the batch
    single = {k: v[:1].to(DEVICE) for k, v in sample_batch.items()}
    iids   = instr_ids[:, :].to(DEVICE)   # [1, L]
    imask  = instr_mask[:, :].to(DEVICE)

    edge_times, server_times, total_times = [], [], []

    for i in range(n_runs + WARMUP):
        # Edge forward
        t_edge = timed_ms(lambda: edge(
            text_inputs  = {"input_ids":      single["input_ids"],
                            "attention_mask": single["attention_mask"]},
            audio_inputs = {"waveform":       single["waveform"]},
            training=False,
        ))

        # Run edge once more cleanly to get fused (no timing overhead)
        fused = edge(
            text_inputs  = {"input_ids":      single["input_ids"],
                            "attention_mask": single["attention_mask"]},
            audio_inputs = {"waveform":       single["waveform"]},
            training=False,
        )

        # Server forward
        t_server = timed_ms(lambda: server(fused, iids, imask))

        if i >= WARMUP:
            edge_times.append(t_edge)
            server_times.append(t_server)
            total_times.append(t_edge + t_server)

    return {
        "edge":   latency_stats(edge_times),
        "server": latency_stats(server_times),
        "total":  latency_stats(total_times),
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_banner(title: str):
    w = 60
    print("\n" + "─" * w)
    print(f"  {title}")
    print("─" * w)


def print_latency_table(lat: dict):
    print(f"  {'Component':<12} {'Median':>9} {'p95':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    for name, stats in lat.items():
        print(f"  {name:<12} "
              f"{stats['median_ms']:>8.1f}ms "
              f"{stats['p95_ms']:>8.1f}ms "
              f"{stats['min_ms']:>8.1f}ms "
              f"{stats['max_ms']:>8.1f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMMA no-compression benchmark")
    parser.add_argument("--debug",        action="store_true",  default=DEBUG)
    parser.add_argument("--no-debug",     action="store_false", dest="debug")
    parser.add_argument("--edge",         default="checkpoints/best_edge.pt",
                        help="Edge checkpoint (contains 'edge' key)")
    parser.add_argument("--server",       default="checkpoints/best_server.pt",
                        help="Server checkpoint (contains 'server' key)")
    parser.add_argument("--latency-runs", type=int, default=N_LATENCY_RUNS)
    parser.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"Device: {DEVICE}  |  debug={args.debug}")

    # ── Build models ──────────────────────────────────────────────────────────
    print("Building models...")
    edge              = build_edge(cfg).to(DEVICE)
    server, tokenizer = build_server(cfg, debug=args.debug)
    server            = server.to(DEVICE)

    # ── Load checkpoints ──────────────────────────────────────────────────────
    if os.path.exists(args.edge):
        ckpt = torch.load(args.edge, map_location=DEVICE, weights_only=False)
        edge.load_state_dict(ckpt["edge"])
        stage_info = f"stage={ckpt.get('stage','?')} epoch={ckpt.get('epoch','?')}"
        print(f"  Loaded edge from {args.edge}  ({stage_info})")
    else:
        print(f"  [warn] Edge checkpoint not found: {args.edge} — using random weights")

    if os.path.exists(args.server):
        ckpt = torch.load(args.server, map_location=DEVICE, weights_only=False)
        server.load_state_dict(ckpt["server"])
        stage_info = f"stage={ckpt.get('stage','?')} epoch={ckpt.get('epoch','?')}"
        print(f"  Loaded server from {args.server}  ({stage_info})")
    else:
        print(f"  [warn] Server checkpoint not found: {args.server} — using random weights")

    # ── Instruction encoding ──────────────────────────────────────────────────
    instr_enc  = tokenizer(INSTRUCTION, return_tensors="pt",
                           padding=True, truncation=True, max_length=32)
    instr_ids  = instr_enc["input_ids"]
    instr_mask = instr_enc["attention_mask"]

    # ── Test data ─────────────────────────────────────────────────────────────
    print("Loading test split...")
    test_ds = MOSEIDataset(split="test", max_text_len=MAX_TEXT_LEN,
                           max_audio_samples=MAX_AUDIO_SAMPLES)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=mosei_collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  {len(test_ds)} test samples")

    # ── Task metrics ──────────────────────────────────────────────────────────
    print_banner("Task Metrics  (test split)")
    m = evaluate_metrics(edge, server, test_loader, instr_ids, instr_mask)
    print(f"  Samples          : {m['n_samples']}")
    print(f"  Loss (sent+emo)  : {m['test_loss']:.4f}")
    print(f"  Sentiment MAE    : {m['sentiment_mae']:.4f}")
    print(f"  Sentiment Acc    : {m['sentiment_acc']:.4f}  (binary sign)")
    print(f"  Emotion Acc      : {m['emotion_acc']:.4f}  (multi-label @ 0.5)")

    # ── Latency benchmark ─────────────────────────────────────────────────────
    print_banner(f"Latency Benchmark  (single sample, N={args.latency_runs})")
    # Grab one batch from the test loader as our timing input
    sample_batch = next(iter(test_loader))
    lat = benchmark_latency(edge, server, sample_batch,
                            instr_ids, instr_mask,
                            n_runs=args.latency_runs)
    print_latency_table(lat)
    print(f"\n  Compression: disabled  (fused embedding → server directly)")
    print(f"  Edge output dim: {cfg.alignment.d_shared}  "
          f"Server scenario: {cfg.server.scenario}  "
          f"Soft tokens: {N_SOFT_TOKENS}")


if __name__ == "__main__":
    main()
