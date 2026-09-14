"""
MUSIC-AVQA audio-visual question answering dataset for EMMI.

Task: given a short video clip (audio + frames) and a question, generate
the answer -- unlike VQADataset (image + question), the question here is
frequently unanswerable from either modality alone (e.g. "how many
instruments are sounding" needs audio to count active sources and video
to disambiguate which instruments are visible) -- so edge-side compression
must genuinely fuse audio and video, not just carry one modality through.
See emma/pipeline.py's AVEdgePipeline and the module note there.

Source: DraculaDragon/MUSIC-AVQA-v2.0 (HuggingFace Hub) -- QA annotations
only (no video bytes). Of the ~10K QA video_ids in this release, only the
~1,035 "additional videos" collected for v2.0 (in
MUSIC-AVQA-v2.0_additional_videos.csv) ship real YouTube IDs + clip
start_time/duration; the rest use MUSIC-AVQA v1's opaque numeric video_ids,
which need a separate ID-mapping file we don't have. So this loader is
restricted to that CSV's ~1,035 videos / ~6,000 QA pairs -- real audio-visual
QA pairs, just a small slice of the full academic benchmark.

Each video is downloaded once via yt-dlp (trimmed to a MAX_CLIP_SEC window
starting at the CSV's start_time -- long enough to cover most of these
~60s clips' content without downloading/caching full-length video for a
prototype-scale run) and cached to disk as raw frames + waveform, exactly
like coco.py/vqa.py cache PIL images rather than preprocessed tensors, so
switching --encoder doesn't require re-downloading. Some videos will have
been deleted/made private since 2024 (YouTube data rot is a known problem
with this benchmark) -- those are skipped, not retried.

Question reconstruction: question_content has "<Object>" placeholders,
filled in order from templ_values. A few templates also reference
positional markers (e.g. "<FL>", first/left) that this HF mirror's CSV
doesn't resolve -- those questions are dropped rather than guessed at.

Answer vocabulary is small and closed (yes/no, counts, left/right/middle,
indoor/outdoor) -- evaluation is plain normalized exact-match accuracy,
not VQAv2's leave-one-out metric (there's a single "anser" field here, not
10 human annotations per question).
"""

import json
import os
import random
import re
import subprocess

import torch
from torch.utils.data import Dataset, DataLoader

HF_DATASET   = "DraculaDragon/MUSIC-AVQA-v2.0"
CACHE_DIR    = ".avqa_cache"
CLIP_CACHE   = os.path.join(CACHE_DIR, "clips")

N_FRAMES       = 8
MAX_CLIP_SEC   = 10          # window from clip start used for both frames + audio
AUDIO_SR       = 16_000

# ── Dataset sizes (video-level, prototype scale) ───────────────────────────────
# Real per-video downloads (yt-dlp + ffmpeg) are ~100x more expensive than the
# single-HTTP-request-per-sample streaming coco.py/vqa.py do, so these defaults
# are intentionally small -- enough to prove the pipeline works end-to-end.
# Scaling up to the full ~1,035-video pool is a straightforward re-run with
# larger N_*_VIDEOS, just slower.
N_TRAIN_VIDEOS = 60
N_VALID_VIDEOS = 15
N_TEST_VIDEOS  = 15


# ── QA annotation loading ──────────────────────────────────────────────────────

def _parse_timestamp(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return float(seconds)


def _fill_template(content: str, templ_values: str) -> str:
    vals = json.loads(templ_values) if templ_values else []
    for v in vals:
        content = content.replace("<Object>", v, 1)
    return content.strip()


_UNRESOLVED_TAG = re.compile(r"<[A-Za-z]+>")


def _load_annotations() -> dict:
    """
    Returns {video_id: {"start_time": float, "qas": [{"question": str, "answer": str}]}}
    restricted to video_ids that have real YouTube start_time/duration metadata.
    """
    from huggingface_hub import hf_hub_download

    files = ["train_balance.json", "test_balance.json",
             "MUSIC-AVQA-v2.0_additional_videos.csv"]
    paths = {f: hf_hub_download(HF_DATASET, f, repo_type="dataset") for f in files}

    csv_meta = {}
    import csv
    with open(paths["MUSIC-AVQA-v2.0_additional_videos.csv"], encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            csv_meta[row["video_id"]] = _parse_timestamp(row["start_time"])

    qa_records = (json.load(open(paths["train_balance.json"])) +
                  json.load(open(paths["test_balance.json"])))

    by_video = {}
    for x in qa_records:
        vid = x["video_id"]
        if vid not in csv_meta or x.get("question_deleted", 0):
            continue
        question = _fill_template(x["question_content"], x["templ_values"])
        if _UNRESOLVED_TAG.search(question):
            continue
        answer = x["anser"].strip().lower()

        entry = by_video.setdefault(vid, {"start_time": csv_meta[vid], "qas": []})
        entry["qas"].append({"question": question, "answer": answer})

    return by_video


# ── Video download + extraction (cached per video_id) ──────────────────────────

def _download_and_extract(video_id: str, start_time: float) -> dict | None:
    """
    Downloads a MAX_CLIP_SEC window starting at start_time, extracts
    N_FRAMES uniformly-sampled PIL frames + a mono 16kHz waveform.
    Returns None (and leaves nothing cached) if the video is unavailable --
    YouTube videos referenced by 2024-era research datasets routinely get
    deleted/made private.
    """
    import numpy as np
    import soundfile as sf
    from PIL import Image

    os.makedirs(CLIP_CACHE, exist_ok=True)
    cache_file = os.path.join(CLIP_CACHE, f"{video_id}.pt")
    if os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)

    tmp_video = os.path.join(CLIP_CACHE, f"_tmp_{video_id}.mp4")
    tmp_audio = os.path.join(CLIP_CACHE, f"_tmp_{video_id}.wav")
    section = f"*{start_time}-{start_time + MAX_CLIP_SEC}"

    try:
        subprocess.run(
            ["yt-dlp", "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
             "--merge-output-format", "mp4",
             "--download-sections", section, "--force-keyframes-at-cuts",
             "-o", tmp_video, f"https://www.youtube.com/watch?v={video_id}"],
            check=True, capture_output=True, timeout=90,
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp_video,
             "-ac", "1", "-ar", str(AUDIO_SR), tmp_audio],
            check=True, capture_output=True, timeout=60,
        )

        wav, _ = sf.read(tmp_audio, dtype="float32")
        target_len = AUDIO_SR * MAX_CLIP_SEC
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", tmp_video],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip() or MAX_CLIP_SEC)

        frames = []
        for i in range(N_FRAMES):
            t = duration * (i + 0.5) / N_FRAMES
            frame_path = os.path.join(CLIP_CACHE, f"_tmp_{video_id}_f{i}.jpg")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", tmp_video,
                 "-frames:v", "1", frame_path],
                check=True, capture_output=True, timeout=30,
            )
            frames.append(Image.open(frame_path).convert("RGB").copy())
            os.remove(frame_path)

        result = {"frames": frames, "waveform": wav}
        torch.save(result, cache_file)
        return result

    except Exception as e:
        print(f"    [skip] {video_id}: {e}")
        return None

    finally:
        for p in (tmp_video, tmp_audio):
            if os.path.exists(p):
                os.remove(p)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AVQADataset(Dataset):
    """
    Each __getitem__ returns:
        frames         : float [N_FRAMES, 3, H, W]  encoder-preprocessed
        waveform       : float [MAX_CLIP_SEC * AUDIO_SR]
        question_input_ids / question_attn_mask : long [max_question_len]
        answer_input_ids   / answer_attn_mask   : long [max_answer_len]
        answer         : str  (normalized ground truth, for eval)
    """

    def __init__(self, samples: list[dict], tokenizer,
                 image_processor, max_question_len: int = 32,
                 max_answer_len: int = 4):
        self.samples          = samples   # [{"frames": [...], "waveform": ..., "question": ..., "answer": ...}]
        self.tok               = tokenizer
        self.proc               = image_processor
        self.max_question_len  = max_question_len
        self.max_answer_len    = max_answer_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        pv = self.proc(images=s["frames"], return_tensors="pt")["pixel_values"]  # [N_FRAMES, 3, H, W]

        q_enc = self.tok(s["question"], max_length=self.max_question_len,
                         padding="max_length", truncation=True, return_tensors="pt")
        a_enc = self.tok(s["answer"], max_length=self.max_answer_len,
                         padding="max_length", truncation=True, return_tensors="pt")
        a_ids  = a_enc["input_ids"][0]
        a_mask = a_enc["attention_mask"][0]

        # Padding token == eos token for these tokenizers, so the filler
        # positions already *contain* eos_token_id -- but attention_mask=0
        # marks all of them (correctly) as ignored during loss, which means
        # the model is never actually supervised on *emitting* eos after the
        # real answer, only on the answer content itself. Un-mask exactly the
        # first pad position so there's one genuine "stop here" label;
        # anything past that stays masked out as before.
        real_len = int(a_mask.sum())
        if real_len < len(a_mask):
            a_mask[real_len] = 1

        return {
            "frames":              pv,
            "waveform":            torch.from_numpy(s["waveform"]).float(),
            "question_input_ids":  q_enc["input_ids"][0],
            "question_attn_mask":  q_enc["attention_mask"][0],
            "answer_input_ids":    a_ids,
            "answer_attn_mask":    a_mask,
            "answer":              s["answer"],
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_loaders(tokenizer,
               batch_size: int = 8,
               max_question_len: int = 32,
               max_answer_len: int = 4,
               num_workers: int = 0,
               n_train_videos: int = N_TRAIN_VIDEOS,
               n_valid_videos: int = N_VALID_VIDEOS,
               n_test_videos:  int = N_TEST_VIDEOS,
               encoder: str = "clip",
               seed: int = 0) -> dict:
    """
    Splits are carved at the video level (not QA-pair level) so the same
    clip never appears in two splits -- otherwise a model could "cheat" by
    memorizing a clip's audio/visual content from train and just reading
    off the answer to an unseen question about it at test time.

    tokenizer must be the *server LLM's* tokenizer (e.g. from build_server()
    in train_avqa.py) -- question/answer ids are looked up directly in the
    LLM's embedding table (see ServerPipelineVQA.forward), so they have to
    share its vocabulary. Passing e.g. a CLIPTokenizer here would silently
    produce nonsense token embeddings rather than an error.
    """
    if encoder == "mobileclip":
        from emma.encoders.mobileclip_encoder import build_mobileclip_processors
        proc, _ = build_mobileclip_processors()
    else:
        from transformers import CLIPImageProcessor
        proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    tok = tokenizer

    print("  loading MUSIC-AVQA annotations...")
    by_video = _load_annotations()
    video_ids = sorted(by_video.keys())
    random.Random(seed).shuffle(video_ids)

    needed = n_train_videos + n_valid_videos + n_test_videos
    splits_videos = {
        "train": video_ids[:n_train_videos],
        "valid": video_ids[n_train_videos:n_train_videos + n_valid_videos],
        "test":  video_ids[n_train_videos + n_valid_videos: needed],
    }

    loaders = {}
    for split, vids in splits_videos.items():
        print(f"  downloading {split}: {len(vids)} videos...")
        samples = []
        for vid in vids:
            clip = _download_and_extract(vid, by_video[vid]["start_time"])
            if clip is None:
                continue
            for qa in by_video[vid]["qas"]:
                samples.append({**clip, **qa})

        ds = AVQADataset(samples, tok, proc,
                         max_question_len=max_question_len,
                         max_answer_len=max_answer_len)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=False,
        )
        print(f"    -> {len(ds)} QA pairs ({len(samples)} from {len(vids)} requested videos)")

    return loaders
