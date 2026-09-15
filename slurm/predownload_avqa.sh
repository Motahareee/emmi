#!/bin/bash
# Pre-downloads MUSIC-AVQA video clips + HF model weights on the login node.
#
# Compute nodes on this cluster run fully offline (see train_llava.sh's
# TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1) -- yt-dlp needs to hit YouTube
# and _load_annotations() needs to hit the HF Hub, neither of which will
# work once the GPU job lands on a compute node. Run this here first so
# train_avqa_llava.sh's sbatch job finds everything already cached
# (.avqa_cache/ for clips, $HF_HOME for the llava weights) and never
# touches the network.
#
# Usage: bash slurm/predownload_avqa.sh
#
# Not an sbatch script -- run directly on the login node (no GPU needed for
# downloading). Takes a while (~1000 videos x a few seconds each); safe to
# re-run/resume since already-cached clips are skipped (see
# emma/data/avqa.py's _download_and_extract).

set -e

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache

cd "$(dirname "$0")/.."

echo "=== Checking Python deps (yt-dlp, soundfile -- see setup.sh) ==="
python3 -c "import yt_dlp"   2>/dev/null || pip install --user yt-dlp
python3 -c "import soundfile" 2>/dev/null || pip install --user soundfile
echo "=== Checking ffmpeg / ffprobe ==="
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    module load FFmpeg 2>/dev/null || {
        echo "ffmpeg/ffprobe not found and no FFmpeg module available on this cluster."
        echo "Check 'module spider ffmpeg' / 'module avail' for the right name,"
        echo "or install a static build into ~/.local/bin and put it on PATH."
        exit 1
    }
fi
echo "yt-dlp/soundfile/ffmpeg/ffprobe OK"

echo "=== Pre-fetching llava-1.5-7b + CLIP weights into HF_HOME ==="
# llava-1.5-7b: the server LLM (train_avqa_llava.sh --scenario llava).
# CLIP ViT-B/32: AVEdgePipeline's frame encoder + get_loaders' image
# processor (train_avqa_llava.sh --encoder clip, the default) -- a
# separate checkpoint from llava's own (discarded) vision tower.
python3 -c "
from transformers import LlavaForConditionalGeneration, AutoTokenizer, CLIPModel, CLIPImageProcessor
LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')
AutoTokenizer.from_pretrained('llava-hf/llava-1.5-7b-hf')
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('cached.')
"

echo "=== Downloading MUSIC-AVQA clips (all available videos) ==="
python3 -c "
from emma.data.avqa import _load_annotations, _download_and_extract

by_video = _load_annotations()
video_ids = sorted(by_video.keys())
print(f'{len(video_ids)} videos with real YouTube ids/timestamps available')

n_ok, n_fail = 0, 0
for i, vid in enumerate(video_ids):
    clip = _download_and_extract(vid, by_video[vid]['start_time'])
    if clip is None:
        n_fail += 1
    else:
        n_ok += 1
    if (i + 1) % 25 == 0:
        print(f'  {i+1}/{len(video_ids)}  (ok={n_ok} failed={n_fail})')

print(f'Done: {n_ok} cached, {n_fail} unavailable (deleted/private -- expected, not an error)')
"

echo "Done: $(date)"
