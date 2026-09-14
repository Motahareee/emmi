"""
VQAv2 visual question answering dataset for EMMI.

Task: given an image and a question, generate the answer (open-ended
text generation, not classification) -- unlike COCOMatchingDataset,
this keeps the pipeline genuinely multimodal (image + question text in,
generated answer text out) while requiring real language generation
that can't be solved by embedding similarity alone.

Source: lmms-lab/VQAv2 (HuggingFace Hub, streaming). This mirror only
ships validation/test/testdev splits (no train), so -- exactly like
coco.py does with COCO's single streaming split -- we carve our own
non-overlapping train/valid/test windows out of the "validation" split
(214K questions over 40K images) via offset, rather than depend on the
much larger official train split we don't need at our scale.

Official VQA v2 fields used here:
  question                — question text
  image                   — PIL image
  multiple_choice_answer  — single most-common human answer (training target)
  answers                 — list of 10 {"answer": str, ...} human annotations
                            (used for the official VQA accuracy metric at eval time)
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader


# ── Dataset sizes ─────────────────────────────────────────────────────────────
N_TRAIN = 5_000
N_VALID =   500
N_TEST  =   500

HF_DATASET   = "lmms-lab/VQAv2"
HF_SPLIT     = "validation"   # the only split large enough to carve train/valid/test from
CACHE_DIR    = ".vqa_cache"
MAX_ANSWERS  = 10             # official VQA annotation count per question


# ── Streaming helper ──────────────────────────────────────────────────────────

def _stream_samples(n: int, offset: int = 0) -> list[dict]:
    """
    Pull n samples from the VQAv2 streaming dataset, starting at `offset`.
    Results are cached to disk so subsequent calls load instantly.

    Returns:
        list of {"image": PIL.Image, "question": str,
                 "answer": str,             # multiple_choice_answer
                 "answers": [str, ...]}     # up to 10 raw human answers
    """
    import os, pickle
    from datasets import load_dataset

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"samples_{offset}_{n}.pkl")

    if os.path.exists(cache_file):
        print(f"  loading from cache ({cache_file})...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  streaming {n} VQA samples from HuggingFace (offset={offset})...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)

    results = []
    for idx, item in enumerate(ds):
        if idx < offset:
            continue
        if len(results) >= n:
            break

        question = (item.get("question") or "").strip()
        answer   = (item.get("multiple_choice_answer") or "").strip()
        img      = item.get("image")
        raw_answers = item.get("answers") or []
        answers = [a["answer"].strip() for a in raw_answers if a.get("answer")]

        if question and answer and img is not None:
            if img.mode != "RGB":
                img = img.convert("RGB")
            results.append({
                "image": img,
                "question": question,
                "answer": answer,
                "answers": answers[:MAX_ANSWERS],
            })

    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    print(f"  cached to {cache_file}")

    return results


# ── Dataset ───────────────────────────────────────────────────────────────────

class VQADataset(Dataset):
    """
    Visual question answering dataset built from VQAv2.

    Each __getitem__ returns:
        pixel_values         : float [3, H, W]  encoder-preprocessed image
        question_input_ids   : long  [max_question_len]
        question_attn_mask   : long  [max_question_len]
        answer_input_ids     : long  [max_answer_len]   teacher-forcing target
        answer_attn_mask     : long  [max_answer_len]
        answers              : list[str]   up to 10 raw human answers, for
                                the official VQA accuracy metric at eval time
                                (kept as a python list -- see the custom
                                collate_fn in get_loaders, since this can't
                                be stacked into a regular tensor batch)
    """

    def __init__(self, raw_samples: list[dict], tokenizer,
                 max_question_len: int = 32,
                 max_answer_len: int = 8,
                 processor=None):
        """
        tokenizer must be the *server LLM's* tokenizer (e.g. built alongside
        the model in build_server()) -- question/answer ids are looked up
        directly in the LLM's embedding table (see ServerPipelineVQA.forward),
        so they have to share its vocabulary. A CLIPTokenizer/other mismatched
        vocab would silently produce nonsense embeddings, not an error.
        """
        from transformers import CLIPImageProcessor

        self.proc = processor or CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tok  = tokenizer
        self.max_question_len = max_question_len
        self.max_answer_len   = max_answer_len

        self.samples = raw_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        pv = self.proc(
            images=s["image"], return_tensors="pt"
        )["pixel_values"][0]

        q_enc = self.tok(
            s["question"],
            max_length=self.max_question_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        a_enc = self.tok(
            s["answer"],
            max_length=self.max_answer_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        a_ids  = a_enc["input_ids"][0]
        a_mask = a_enc["attention_mask"][0]

        # See emma/data/avqa.py's AVQADataset for why: pad_token == eos_token
        # for these tokenizers, so masking out *all* padding also masks out
        # the one position that should teach the model to emit eos right
        # after the real answer. Un-mask exactly that first pad position.
        real_len = int(a_mask.sum())
        if real_len < len(a_mask):
            a_mask[real_len] = 1

        return {
            "pixel_values":       pv,
            "question_input_ids": q_enc["input_ids"][0],
            "question_attn_mask": q_enc["attention_mask"][0],
            "answer_input_ids":   a_ids,
            "answer_attn_mask":   a_mask,
            "answers":            s["answers"],   # left as a python list
        }


def _vqa_collate(batch: list[dict]) -> dict:
    """
    Default collate stacks tensors normally but keeps `answers` (a list of
    variable-length string lists, one per sample) as a plain python list --
    it can't be stacked into a tensor and is only needed at eval time for
    the official VQA accuracy metric, not during training.
    """
    out = {}
    for key in batch[0]:
        if key == "answers":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_loaders(tokenizer,
                batch_size: int = 16,
                max_question_len: int = 32,
                max_answer_len: int = 8,
                num_workers: int = 0,
                n_train: int = N_TRAIN,
                n_valid: int = N_VALID,
                n_test:  int = N_TEST,
                encoder: str = "clip") -> dict:
    """
    Build train / valid / test DataLoaders for VQAv2.

    Streaming order is deterministic (no shuffle in the HF loader), so
    the three splits are non-overlapping windows into the stream -- same
    convention as emma/data/coco.py.

    tokenizer: the server LLM's tokenizer -- see VQADataset's docstring
               for why this can't default to an internally-built one.
    encoder: "clip" (HF CLIP ViT-B/32 preprocessing, 224x224) or
             "mobileclip" (open_clip MobileCLIP2-S0 preprocessing, 256x256).
    """
    if encoder == "mobileclip":
        from emma.encoders.mobileclip_encoder import build_mobileclip_processors
        proc, _ = build_mobileclip_processors()
    else:
        from transformers import CLIPImageProcessor
        proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tok = tokenizer

    offsets = {
        "train": 0,
        "valid": n_train,
        "test":  n_train + n_valid,
    }
    sizes = {"train": n_train, "valid": n_valid, "test": n_test}

    loaders = {}
    for split in ("train", "valid", "test"):
        print(f"  streaming VQAv2 {split} ({sizes[split]} questions)...")
        raw = _stream_samples(sizes[split], offset=offsets[split])

        ds = VQADataset(
            raw, tok,
            max_question_len=max_question_len,
            max_answer_len=max_answer_len,
            processor=proc,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=_vqa_collate,
        )
        print(f"    → {len(ds)} questions")

    return loaders
