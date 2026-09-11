"""
Benchmark LLaVA-1.5-7B inference latency and GPU memory on a single image-text query.
Reports numbers comparable to EMMA's per-stage latency table.

Measures:
  - Model load time
  - Peak GPU memory (GB)
  - Inference latency per sample (ms), median over N_RUNS
  - Time to first token (ms)
"""

import time
import torch
import numpy as np
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

MODEL_ID  = "llava-hf/llava-v1.6-mistral-7b-hf"   # LLaVA-1.5 NeXT 7B
N_RUNS    = 20
N_WARMUP  = 3
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

def reset_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def main():
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_ID}")
    print(f"Runs:   {N_RUNS} (+ {N_WARMUP} warmup)\n")

    # ── Load model ────────────────────────────────────────────────────────────
    reset_memory()
    t0 = time.perf_counter()
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    load_time = time.perf_counter() - t0
    mem_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"Model load time : {load_time:.1f} s")
    print(f"GPU memory (load): {mem_after_load:.2f} GB\n")

    # ── Prepare input ─────────────────────────────────────────────────────────
    # Synthetic 336x336 RGB image (LLaVA-1.5 default resolution)
    image = Image.fromarray(
        np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
    )
    prompt = "USER: <image>\nDoes this image match the description: a dog sitting on a bench? ASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("Warming up...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()

    # ── Benchmark: time to first token ────────────────────────────────────────
    ttft_times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            reset_memory()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            ttft_times.append((time.perf_counter() - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # ── Benchmark: full short response (max 20 tokens) ────────────────────────
    full_times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            torch.cuda.synchronize()
            full_times.append((time.perf_counter() - t0) * 1000)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== LLaVA-1.5-7B Benchmark Results ===")
    print(f"GPU memory (model weights) : {mem_after_load:.2f} GB")
    print(f"GPU memory (peak inference): {peak_mem:.2f} GB")
    print(f"Time to first token        : {np.median(ttft_times):.1f} ms  "
          f"(min {np.min(ttft_times):.1f}, max {np.max(ttft_times):.1f})")
    print(f"Full inference (20 tokens) : {np.median(full_times):.1f} ms  "
          f"(min {np.min(full_times):.1f}, max {np.max(full_times):.1f})")
    print(f"\nFor comparison — EMMA pipeline on same A40:")
    print(f"  Edge (CLIP encode + compress) : ~10.9 ms")
    print(f"  Server (GPT-2 + head)         :  ~7.1 ms")
    print(f"  EMMA total (excl. TX)         : ~18.0 ms")

if __name__ == "__main__":
    main()
