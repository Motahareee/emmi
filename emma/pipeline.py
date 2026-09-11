import torch
import torch.nn as nn

from emma.config import EMMAConfig
from emma.encoders import TextEncoder, ImageEncoder
from emma.alignment import CrossModalAlignment
from emma.compression.vae import VAE


class EdgePipeline(nn.Module):
    """
    EMMA edge-side pipeline (Stages 1–3).

    Stages:
        1. CLIP encoders  — text [B, 512], image [B, 512]
           Both encoders share the same CLIP joint embedding space.
        2. Cross-modal fusion — mean pool → [B, 512]
        3. VAE encoder — compress to latent z for transmission (optional)

    use_compression=False skips Stage 3 — the fused embedding is transmitted.
    """

    def __init__(self, config: EMMAConfig = None, vae_checkpoint: str = None,
                 text_encoder: nn.Module = None,
                 image_encoder: nn.Module = None):
        super().__init__()
        cfg = config or EMMAConfig()
        ec  = cfg.encoder
        ac  = cfg.alignment
        vc  = cfg.vae
        self.use_compression = cfg.use_compression

        if text_encoder is not None or image_encoder is not None:
            self.text_encoder  = text_encoder
            self.image_encoder = image_encoder
        elif ec.family == "mobileclip":
            from emma.encoders.mobileclip_encoder import build_mobileclip_encoders
            self.text_encoder, self.image_encoder = build_mobileclip_encoders(
                freeze_base=ec.text_freeze_base,
            )
        else:
            self.text_encoder = TextEncoder(
                model_name=ec.model_name,
                freeze_base=ec.text_freeze_base,
            )
            self.image_encoder = ImageEncoder(
                model_name=ec.model_name,
                freeze_base=ec.image_freeze_base,
            )
        self.alignment = CrossModalAlignment(d_shared=ac.d_shared,
                                             fusion=ac.fusion)

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

    def _encode_parallel(self, text_inputs: dict, pixel_values: torch.Tensor):
        """
        Run text and image encoders in parallel using CUDA streams (GPU)
        or threads (CPU). Returns (t_emb, v_emb).
        """
        device = next(self.text_encoder.parameters()).device

        if device.type == "cuda":
            stream_text = torch.cuda.Stream(device=device)
            stream_img  = torch.cuda.Stream(device=device)
            t_result, v_result = [None], [None]

            with torch.cuda.stream(stream_text):
                t_result[0] = self.text_encoder(**text_inputs)

            with torch.cuda.stream(stream_img):
                v_result[0] = self.image_encoder(pixel_values)

            torch.cuda.synchronize(device)
            return t_result[0], v_result[0]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                t_fut = ex.submit(lambda: self.text_encoder(**text_inputs))
                v_fut = ex.submit(lambda: self.image_encoder(pixel_values))
                return t_fut.result(), v_fut.result()

    def encode_contrastive(self, text_inputs: dict,
                           image_inputs: dict) -> tuple:
        """
        Returns (t_emb, v_emb) — CLIP joint embeddings for both modalities.
        Used for InfoNCE contrastive fine-tuning on the edge.

        CLIP already aligns text and image, so embeddings are returned
        directly without any additional projection.

        Returns:
            t_emb : [B, 512]
            v_emb : [B, 512]
        """
        return self._encode_parallel(text_inputs, image_inputs["pixel_values"])

    def forward(self, text_inputs: dict, image_inputs: dict,
                training: bool = True):
        """
        Args:
            text_inputs  : {"input_ids": ..., "attention_mask": ...}
            image_inputs : {"pixel_values": Tensor [B, 3, 224, 224]}
            training     : controls VAE output (ignored when use_compression=False)

        Returns:
            use_compression=True,  training=True  → (mu, log_var, fused)
            use_compression=True,  training=False → mu
            use_compression=False                 → fused  [B, 512]
        """
        t, v  = self._encode_parallel(text_inputs, image_inputs["pixel_values"])
        fused = self.alignment(t, v)                          # [B, 512]

        if not self.use_compression:
            return fused

        if training:
            mu, log_var = self.vae_encoder(fused)
            return mu, log_var, fused
        else:
            return self.vae_encoder.encode(fused)             # [B, d_latent]
