"""
Pre-cache COCO splits for the 72K/5K/5K experiment.

The dataset (clip-benchmark/wds_mscoco_captions) has 82,783 images total.
Splits: train 0-72K, valid 72K-77K, test 77K-82K.

Train and valid are sliced from the existing samples_0_80000.pkl cache;
only the last ~2,000 test images are streamed from HuggingFace.

Run on a login node (needs internet access):

    cd /scratch/user/motahare/emma
    module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
    source /scratch/user/motahare/emma_env/bin/activate
    export HF_HOME=/scratch/user/motahare/hf_cache
    python3 precache_coco.py
"""

import os
import pickle

from emma.data.coco import _stream_samples, CACHE_DIR

SRC = os.path.join(CACHE_DIR, "samples_0_80000.pkl")


def save(samples, offset, n):
    path = os.path.join(CACHE_DIR, f"samples_{offset}_{n}.pkl")
    if os.path.exists(path):
        print(f"  {path} already exists, skipping")
        return
    assert len(samples) == n, f"expected {n} samples, got {len(samples)}"
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    print(f"  wrote {path} ({n} samples)")


print(f"Loading {SRC}...")
with open(SRC, "rb") as f:
    base = pickle.load(f)
print(f"  {len(base)} samples")

print("Slicing train (0 - 72000)...")
save(base[:72_000], 0, 72_000)

print("Slicing valid (72000 - 77000)...")
save(base[72_000:77_000], 72_000, 5_000)

print("Building test (77000 - 82000)...")
test_path = os.path.join(CACHE_DIR, "samples_77000_5000.pkl")
if os.path.exists(test_path):
    print(f"  {test_path} already exists, skipping")
else:
    tail = _stream_samples(2_000, offset=80_000)   # streams + caches 80K-82K
    test = base[77_000:80_000] + tail
    save(test, 77_000, 5_000)

print("Done")
