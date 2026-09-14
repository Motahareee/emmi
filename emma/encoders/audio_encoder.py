"""
Lightweight audio encoder — Stage 1 (edge), per CLAUDE.md's "spectrogram +
CNN or small wav2vec" suggestion. Unlike TextEncoder/ImageEncoder, there is
no pretrained-and-frozen-by-default checkpoint here (no CLIP-style joint
audio-text-image space is being reused) — this is a small CNN trained
jointly with alignment/compression, matching CLAUDE.md's Stage 1/2 coupling.

Input:  waveform [B, T]  (mono, 16 kHz)
Output: [B, 512]          — same dim as the image encoders, so it drops
                             into CrossModalAlignment unchanged.
"""

import torch
import torch.nn as nn
import torchaudio


SAMPLE_RATE = 16_000
N_MELS      = 64


class AudioEncoder(nn.Module):
    def __init__(self, d_out: int = 512, freeze_base: bool = False, **kwargs):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, d_out)

        if freeze_base:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: [B, T] mono @ 16kHz -> [B, d_out]"""
        spec = self.db(self.melspec(waveform))          # [B, n_mels, frames]
        spec = spec.unsqueeze(1)                        # [B, 1, n_mels, frames]
        h = self.conv(spec).flatten(1)                  # [B, 128]
        return self.proj(h)                              # [B, d_out]
