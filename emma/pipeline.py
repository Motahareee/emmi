import torch
import torch.nn as nn

from emma.config import EMMAConfig
from emma.encoders import TextEncoder, AudioEncoder, VisionEncoder
from emma.alignment import CrossModalAlignment
from emma.compression.vae import VAE


class EdgePipeline(nn.Module):
    """
    EMMA edge-side pipeline (Stages 1–3).

    Stages:
        1. Modality encoders  — text, audio, vision → per-modality embeddings
        2. Cross-modal alignment — project & fuse into shared embedding
        3. VAE encoder — compress to latent z for transmission

    Two encoder modes:
      Raw inputs  — default; builds DistilBERT / Whisper-small / MobileNetV3 from config
      Pre-extracted — pass text_encoder, audio_encoder, vision_encoder explicitly
                      (e.g. ModalityProjector instances for pre-extracted features)

    The VAE encoder is loaded from a pre-trained checkpoint produced by
    training emma.compression.vae.VAE standalone. Pass the checkpoint path
    via vae_checkpoint; if None the encoder is randomly initialised (useful
    only for architecture tests).

    use_compression=False skips the VAE entirely — the fused embedding is
    returned as-is for direct transmission to the server.

    Training mode  : returns (mu, log_var, fused)  or  fused (no compression)
    Inference mode : returns mu                     or  fused (no compression)
    """

    def __init__(self, config: EMMAConfig = None, vae_checkpoint: str = None,
                 text_encoder: nn.Module = None,
                 audio_encoder: nn.Module = None,
                 vision_encoder: nn.Module = None):
        super().__init__()
        cfg = config or EMMAConfig()
        ec  = cfg.encoder
        ac  = cfg.alignment
        vc  = cfg.vae
        self.use_compression = cfg.use_compression

        # Stage 1 — use provided encoders or build from config
        self.text_encoder = text_encoder or TextEncoder(
            model_name=ec.text_model_name,
            d_out=ec.d_text,
            freeze_base=ec.text_freeze_base,
        )
        self.audio_encoder = audio_encoder or AudioEncoder(
            model_name=ec.audio_model_name,
            d_out=ec.d_audio,
            freeze_base=ec.audio_freeze_base,
            audio_input_sr=ec.audio_input_sr,
        )
        # Vision encoder reserved for future extension (Facet 4.2 unavailable)
        self.vision_encoder = None

        # Stage 2
        self.alignment = CrossModalAlignment(
            d_text=ec.d_text,
            d_audio=ec.d_audio,
            d_shared=ac.d_shared,
            use_attention_fusion=ac.use_attention_fusion,
        )

        # Stage 3 — encoder half of the VAE only (skipped if use_compression=False)
        if self.use_compression:
            if vae_checkpoint is not None:
                self.vae_encoder = VAE.load(vae_checkpoint).encoder
            else:
                self.vae_encoder = VAE(
                    d_in=ac.d_shared,
                    d_latent=vc.d_latent,
                    encoder_hidden_dims=vc.encoder_hidden_dims,
                    decoder_hidden_dims=tuple(reversed(vc.encoder_hidden_dims)),
                ).encoder
        else:
            self.vae_encoder = None

    def forward(self, text_inputs: dict, audio_inputs: dict,
                training: bool = True):
        """
        Args:
            text_inputs  : {"input_ids": ..., "attention_mask": ...}
            audio_inputs : {"waveform": ...}
            training     : if True returns (mu, log_var, fused)
                           if False returns mu only (or fused if no compression)

        Returns:
            use_compression=True,  training=True  → (mu, log_var, fused)
            use_compression=True,  training=False → mu
            use_compression=False                 → fused
        """
        # Stage 1
        t = self.text_encoder(**text_inputs)    # [B, d_text]
        a = self.audio_encoder(**audio_inputs)  # [B, d_audio]

        # Stage 2
        fused = self.alignment(t, a)            # [B, d_shared]

        # Stage 3
        if not self.use_compression:
            return fused                                   # [B, d_shared] — sent as-is

        if training:
            mu, log_var = self.vae_encoder(fused)
            return mu, log_var, fused                      # fused = reconstruction target
        else:
            return self.vae_encoder.encode(fused)          # [B, d_latent]
