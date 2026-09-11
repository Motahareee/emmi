"""
COCO image-text matching dataset for EMMA.

Task: given an image and a caption, predict whether they match (binary yes/no).

Positive pairs : image + one of its own captions  (label=1)
Negative pairs : image + a caption from a different image (label=0)
Balance        : 50/50, pairs shuffled.

Source: clip-benchmark/wds_mscoco_captions (HuggingFace Hub, streaming).
The dataset has only a single 'train' split; we carve out valid/test by
advancing an offset into the stream.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader


# ── Dataset sizes ─────────────────────────────────────────────────────────────
N_TRAIN = 5_000   # unique images → 10,000 pairs (50% pos, 50% neg)
N_VALID =   500   # →  1,000 pairs
N_TEST  =   500   # →  1,000 pairs

HF_DATASET  = "clip-benchmark/wds_mscoco_captions"
CACHE_DIR   = ".coco_cache"


# ── Streaming helper ──────────────────────────────────────────────────────────

def _stream_samples(n: int, offset: int = 0) -> list[dict]:
    """
    Pull n samples from the COCO wds streaming dataset, starting at `offset`.
    Results are cached to disk so subsequent calls load instantly.

    Returns:
        list of {"image": PIL.Image, "captions": [str, ...]}
    """
    import os, pickle
    from datasets import load_dataset

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"samples_{offset}_{n}.pkl")

    if os.path.exists(cache_file):
        print(f"  loading from cache ({cache_file})...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  streaming {n} images from HuggingFace (offset={offset})...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    results = []
    for idx, item in enumerate(ds):
        if idx < offset:
            continue
        if len(results) >= n:
            break

        txt = item["txt"]
        if isinstance(txt, str):
            captions = [c.strip() for c in txt.splitlines() if c.strip()]
        else:
            captions = [str(txt).strip()]

        if captions and item["jpg"] is not None:
            img = item["jpg"]
            if img.size != (224, 224):
                img = img.resize((224, 224))
            results.append({"image": img, "captions": captions})

    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    print(f"  cached to {cache_file}")

    return results


# ── Dataset ───────────────────────────────────────────────────────────────────

class COCOMatchingDataset(Dataset):
    """
    Image-text matching dataset built from COCO captions.

    Each __getitem__ returns:
        pixel_values  : float [3, 224, 224]  CLIP-preprocessed
        input_ids     : long  [max_text_len]
        attention_mask: long  [max_text_len]
        label         : float  1.0 (match) or 0.0 (no match)
    """

    def __init__(self, raw_samples: list[dict],
                 max_text_len: int = 64,
                 seed: int = 0,
                 processor=None,
                 tokenizer=None):
        """
        Args:
            raw_samples  — list of {"image": PIL.Image, "captions": [str]}
            max_text_len — tokenizer padding/truncation length
            seed         — for reproducible negative sampling
            processor    — CLIPImageProcessor; built from default checkpoint if None
            tokenizer    — AutoTokenizer; built from distilbert-base-uncased if None
        """
        from transformers import CLIPImageProcessor, CLIPTokenizer

        self.proc = processor or CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tok  = tokenizer or CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.max_text_len = max_text_len

        rng = random.Random(seed)

        # Store PIL images; pick one caption per image
        self.images   = [s["image"]                     for s in raw_samples]
        self.captions = [rng.choice(s["captions"])      for s in raw_samples]

        N = len(self.images)

        # Positives: image_i paired with caption_i
        positives = [(i, i, 1) for i in range(N)]

        # Negatives: image_i paired with caption_j (j ≠ i)
        neg_order = list(range(N))
        rng.shuffle(neg_order)
        negatives = []
        for i in range(N):
            j = neg_order[i]
            if j == i:                         # guarantee different image
                j = (j + 1) % N
            negatives.append((i, j, 0))

        self.pairs = positives + negatives
        rng.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_i, cap_i, label = self.pairs[idx]

        pv = self.proc(
            images=self.images[img_i], return_tensors="pt"
        )["pixel_values"][0]                                   # [3, 224, 224]

        enc = self.tok(
            self.captions[cap_i],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values":   pv,
            "input_ids":      enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label":          torch.tensor(float(label)),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_loaders(batch_size: int = 16,
                max_text_len: int = 64,
                num_workers: int = 0,
                n_train: int = N_TRAIN,
                n_valid: int = N_VALID,
                n_test:  int = N_TEST,
                encoder: str = "clip") -> dict:
    """
    Build train / valid / test DataLoaders for COCO image-text matching.

    Streaming order is deterministic (no shuffle in the HF loader), so
    the three splits are non-overlapping windows into the dataset.

    encoder: "clip" (HF CLIP ViT-B/32 preprocessing, 224x224) or
             "mobileclip" (open_clip MobileCLIP2-S0 preprocessing, 256x256).
    """
    # Build shared processor / tokenizer once to avoid repeated downloads
    if encoder == "mobileclip":
        from emma.encoders.mobileclip_encoder import build_mobileclip_processors
        proc, tok = build_mobileclip_processors()
    else:
        from transformers import CLIPImageProcessor, CLIPTokenizer
        proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        tok  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    offsets = {
        "train": 0,
        "valid": n_train,
        "test":  n_train + n_valid,
    }
    sizes = {"train": n_train, "valid": n_valid, "test": n_test}

    loaders = {}
    for split in ("train", "valid", "test"):
        print(f"  streaming COCO {split} ({sizes[split]} images)...")
        raw = _stream_samples(sizes[split], offset=offsets[split])

        ds = COCOMatchingDataset(
            raw,
            max_text_len=max_text_len,
            seed={"train": 0, "valid": 1, "test": 2}[split],
            processor=proc,
            tokenizer=tok,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
        print(f"    → {len(ds)} pairs ({len(raw)} images)")

    return loaders
