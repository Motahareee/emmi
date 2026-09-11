"""
Capability check for the real EMMI "llava" scenario.

Loads LLaVA-1.5-7B's language_model backbone only (the vision tower is
discarded -- EMMI supplies its own soft tokens via the projection MLP,
same as emma/server/pipeline.py's "llava" scenario), then runs a forward
pass on a short sequence (8 soft tokens + a short instruction) matching
actual server-pipeline usage. No image processing, no generate() loop.

Usage:
    python3 check_llava_capability.py --load-in-8bit
"""

import argparse
import time

import torch

from emma.server.pipeline import _load_llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-in-8bit", action="store_true",
                    help="load with bitsandbytes 8-bit quantization")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
        torch.cuda.reset_peak_memory_stats()

    print(f"\nLoading llava-hf/llava-1.5-7b-hf (load_in_8bit={args.load_in_8bit})...")
    t0 = time.perf_counter()
    llm, d_llm = _load_llm("llava", load_in_8bit=args.load_in_8bit)
    if not args.load_in_8bit:
        llm = llm.to(device)
    load_time = time.perf_counter() - t0

    n_params = sum(param.numel() for param in llm.parameters())
    print(f"Load time: {load_time:.1f}s")
    print(f"Backbone: {llm.__class__.__name__}  ({n_params / 1e9:.2f}B params)")
    print(f"d_llm (hidden size): {d_llm}")
    if device == "cuda":
        print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Smoke test: forward pass shaped like the real server pipeline --
    # N_SOFT_TOKENS soft tokens (from EMMI's projection MLP) + a short
    # instruction, no vision tower, no generation.
    llm.eval()
    B, N_SOFT, SEQ = 4, 8, 12
    param_dtype = next(llm.parameters()).dtype
    fake_embeds = torch.randn(B, N_SOFT + SEQ, d_llm, device=device, dtype=param_dtype)
    attn_mask = torch.ones(B, N_SOFT + SEQ, device=device, dtype=torch.long)

    with torch.no_grad():
        out = llm(inputs_embeds=fake_embeds, attention_mask=attn_mask,
                    output_hidden_states=True)

    print(f"\nForward pass OK -- last hidden state shape: {tuple(out.hidden_states[-1].shape)}")
    if device == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n=== Result: LLaVA-1.5-7B language_model backbone loads and runs. ===")


if __name__ == "__main__":
    main()
