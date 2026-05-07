"""
CMU-MOSEI data loader.

Modalities
----------
Text    — tokenised transcriptions (cairocode/cmu_mosei_wav, HuggingFace Hub)
Audio   — raw waveforms decoded with soundfile (same HF dataset)
Vision  — Facet 4.2 features (42-dim) downloaded via mmsdk from the CMU
          MOSEI server.  Falls back to zeros if mmsdk is unavailable or the
          server cannot be reached; a warning is printed in that case.

Audio is decoded on-the-fly with soundfile (pip install soundfile).
Visual features are downloaded once per machine and cached in MMSDK_CACHE_DIR.
Dataset is split by video ID to prevent leakage between splits.
"""

import io
import os
import warnings

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Audio
from typing import Dict, List

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from mmsdk import mmdatasdk
    MMSDK_AVAILABLE = True
except ImportError:
    MMSDK_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

HF_DATASET      = "cairocode/cmu_mosei_wav"
EMOTION_COLS    = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
SAMPLE_RATE     = 22050
MMSDK_CACHE_DIR = "./mmsdk_cache"
VISUAL_KEY      = "FACET 4.2"
VISUAL_DIM      = 42          # CMU-MOSEI Facet 4.2 feature dimensionality

FEATURE_DIMS  = {"vision": VISUAL_DIM}
_SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}

# Module-level visual feature cache — loaded once per process
_visual_cache: dict | None = None
_visual_load_attempted: bool = False


# ── Visual feature loading ────────────────────────────────────────────────────

def _load_visual_features() -> dict | None:
    """
    Download (first call) and return CMU-MOSEI Facet 4.2 visual features.

    Returns
    -------
    dict  {video_id: np.ndarray [n_segments, VISUAL_DIM]}  on success
    None  if mmsdk is not installed or the CMU server is unreachable.

    The downloaded .csd file is cached in MMSDK_CACHE_DIR so subsequent
    calls (and subsequent runs) skip the download.
    """
    global _visual_cache, _visual_load_attempted
    if _visual_load_attempted:
        return _visual_cache

    _visual_load_attempted = True

    if not MMSDK_AVAILABLE:
        warnings.warn(
            "mmsdk is not installed — vision features will be zeros.\n"
            "Install with:  pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git"
        )
        return None

    os.makedirs(MMSDK_CACHE_DIR, exist_ok=True)
    try:
        recipe  = {VISUAL_KEY: mmdatasdk.cmu_mosei.highlevel[VISUAL_KEY]}
        dataset = mmdatasdk.mmdataset(recipe, MMSDK_CACHE_DIR)
        seq     = dataset[VISUAL_KEY]

        features: dict[str, np.ndarray] = {}
        for vid in seq.keys():
            arr = seq[vid]["features"]   # [n_segments, dim]  or  [n_segments, T, dim]
            if arr.ndim == 3:            # time-series → mean-pool over frames
                arr = arr.mean(axis=1)
            # Facet can emit NaN for undetected frames — replace with 0
            arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
            features[vid] = arr          # [n_segments, VISUAL_DIM]

        _visual_cache = features
        print(f"[mosei] Loaded Facet 4.2 visual features: "
              f"{len(features)} videos, {VISUAL_DIM}-dim")
        return _visual_cache

    except Exception as exc:
        warnings.warn(
            f"[mosei] Could not load mmsdk visual features ({exc}).\n"
            "Vision modality will be zeros for this run.\n"
            "Make sure the CMU MOSEI server (immortal.multicomp.cs.cmu.edu) "
            "is reachable and try again."
        )
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class MOSEIDataset(Dataset):
    """
    Args
    ----
    split              : "train" | "valid" | "test"
    tokenizer_name     : HuggingFace model name for text tokenisation
    max_text_len       : max token length
    max_audio_samples  : truncate / pad waveform to this many samples

    Each item
    ---------
    input_ids        [max_text_len]       tokenised text
    attention_mask   [max_text_len]       padding mask
    waveform         [max_audio_samples]  raw audio, float32 in [-1, 1]
    sentiment        scalar float         continuous score
    emotions         [6]                  continuous emotion scores
    """

    def __init__(self, split: str = "train",
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_text_len: int = 128,
                 max_audio_samples: int = 132300):
        if not SOUNDFILE_AVAILABLE:
            raise ImportError("Install soundfile:  pip install soundfile")

        self.max_text_len      = max_text_len
        self.max_audio_samples = max_audio_samples
        self.tokenizer         = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load raw audio/text from HuggingFace
        raw = load_dataset(HF_DATASET, split="train")
        raw = raw.cast_column("audio", Audio(decode=False))

        # Load visual features (may be None if server unreachable)
        self.visual = _load_visual_features()

        # Split by video ID; also record each segment's index within its video
        # so we can index into the mmsdk feature arrays correctly.
        self.samples = _split_by_video(raw, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        row, seg_idx = self.samples[idx]

        # ── Text ──────────────────────────────────────────────────────────────
        enc = self.tokenizer(
            row["text"] or row.get("ASR") or "",
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Audio ─────────────────────────────────────────────────────────────
        wav, _ = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        T = len(wav)
        if T >= self.max_audio_samples:
            wav = wav[:self.max_audio_samples]
        else:
            wav = np.pad(wav, (0, self.max_audio_samples - T))

        # ── Labels ────────────────────────────────────────────────────────────
        emotions = torch.tensor(
            [float(row[c]) for c in EMOTION_COLS], dtype=torch.float32
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "waveform":       torch.tensor(wav, dtype=torch.float32),
            "sentiment":      torch.tensor(float(row["sentiment"]), dtype=torch.float32),
            "emotions":       emotions,
        }


# ── Split helper ──────────────────────────────────────────────────────────────

def _split_by_video(raw_dataset, split: str) -> list:
    """
    Group segments by video ID and split groups 80/10/10.
    Keeps all segments from one video in the same split (no leakage).

    Also records each segment's index within its video (in dataset order)
    so the caller can look up mmsdk visual features by position.

    Returns list of (row_dict, seg_idx_within_video).
    """
    video_ids = sorted(set(raw_dataset["video"]))
    n     = len(video_ids)
    t_end = int(n * _SPLIT_RATIOS["train"])
    v_end = t_end + int(n * _SPLIT_RATIOS["valid"])

    if split == "train":
        keep = set(video_ids[:t_end])
    elif split == "valid":
        keep = set(video_ids[t_end:v_end])
    else:
        keep = set(video_ids[v_end:])

    # Count segments per video across the ENTIRE dataset (not just the split)
    # so the index into the mmsdk array is correct.
    seg_counter: dict[str, int] = {}
    result = []
    for row in raw_dataset:
        vid     = row["video"]
        seg_idx = seg_counter.get(vid, 0)
        seg_counter[vid] = seg_idx + 1
        if vid in keep:
            result.append((row, seg_idx))

    return result


# ── Collate + loaders ─────────────────────────────────────────────────────────

def mosei_collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def get_loaders(batch_size: int = 16,
                max_text_len: int = 128,
                max_audio_samples: int = 132300,
                tokenizer_name: str = "distilbert-base-uncased",
                num_workers: int = 2) -> Dict[str, DataLoader]:
    """
    Returns {"train": DataLoader, "valid": DataLoader, "test": DataLoader}.

    Example
    -------
    loaders = get_loaders(batch_size=16)
    for batch in loaders["train"]:
        text_inputs   = {"input_ids": batch["input_ids"],
                         "attention_mask": batch["attention_mask"]}
        audio_inputs  = {"waveform": batch["waveform"]}
        vision_inputs = {"features": batch["vision_feat"]}
        labels        = {"sentiment": batch["sentiment"],
                         "emotions":  batch["emotions"]}
    """
    loaders = {}
    for split in ("train", "valid", "test"):
        ds = MOSEIDataset(split=split, tokenizer_name=tokenizer_name,
                          max_text_len=max_text_len,
                          max_audio_samples=max_audio_samples)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=mosei_collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
