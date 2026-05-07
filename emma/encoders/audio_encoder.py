import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor


class AudioEncoder(nn.Module):
    """
    Whisper-small encoder backbone with a linear projection head.

    Mirrors the TextEncoder design:
        frozen pre-trained backbone  →  mean-pool  →  trainable projection

    Input:  waveform [B, T]  raw audio at audio_input_sr Hz
    Output: [B, d_out]

    Whisper-small specifics:
        - Expects 16 kHz mono audio → we resample internally if needed
        - Produces hidden states of shape [B, T', 384] (d_model=384)
        - We mean-pool over the time axis before projecting
    """

    WHISPER_SR = 16000   # Whisper always expects 16 kHz

    def __init__(self, model_name: str = "openai/whisper-small",
                 d_out: int = 256, freeze_base: bool = True,
                 audio_input_sr: int = 22050):
        super().__init__()
        self.audio_input_sr = audio_input_sr

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.backbone          = WhisperModel.from_pretrained(model_name).encoder
        d_model                = self.backbone.config.d_model   # 384 for whisper-small

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, T]
        device = waveform.device

        # Resample to 16 kHz if the input is at a different rate
        if self.audio_input_sr != self.WHISPER_SR:
            waveform = torchaudio.functional.resample(
                waveform, self.audio_input_sr, self.WHISPER_SR
            )

        # Whisper encoder requires exactly 3000 mel frames (30 s).
        # Pad each waveform to 30 s before the feature extractor so all
        # inputs produce the expected spectrogram size regardless of clip length.
        target_samples = 30 * self.WHISPER_SR   # 480 000 samples
        waveforms_np = []
        for i in range(waveform.size(0)):
            w = waveform[i].cpu().numpy()
            if len(w) < target_samples:
                w = np.pad(w, (0, target_samples - len(w)))
            else:
                w = w[:target_samples]
            waveforms_np.append(w)

        inputs = self.feature_extractor(
            waveforms_np,
            sampling_rate=self.WHISPER_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)   # [B, 80, 3000]

        # Encoder forward
        hidden = self.backbone(input_features).last_hidden_state  # [B, T', 384]

        # Mean-pool over time then project
        pooled = hidden.mean(dim=1)                               # [B, 384]
        return self.proj(pooled)                                  # [B, d_out]
