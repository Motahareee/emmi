"""
Server-side pipeline (Stage 4).

Three scenarios — all share the same ServerPipeline class; only the LLM
backbone differs. In every case the interface to the edge is identical:
a latent vector z ∈ ℝ^d_latent arrives, the VAEDecoder reconstructs the
shared embedding, and a projection MLP maps it to soft tokens that are
prepended to any text instruction tokens before the LLM runs.

SCENARIO_REGISTRY maps scenario name → (model_name, human description,
loader strategy). The loader strategy determines how the HF model is
loaded and which component is used as the LLM backbone:

  "causal_lm"   — AutoModelForCausalLM, used directly
  "llava"       — LlavaForConditionalGeneration; we extract .language_model
                  so the LLM has weights pre-trained on visual soft tokens
  "qwen_audio"  — Qwen2AudioForConditionalGeneration; we extract .language_model
                  so the LLM has weights pre-trained on audio soft tokens

In all three cases the vision/audio encoder inside the original model is
discarded — we replace it with EMMA's VAEDecoder + custom projection.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from emma.compression.vae import VAE, VAEDecoder
from emma.config import EMMAConfig


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY = {
    "llava": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "loader": "llava",
        "description": (
            "LLM whose weights were fine-tuned on visual soft tokens (LLaVA 1.5). "
            "We discard the CLIP encoder and inject our VAE-decoded embeddings "
            "through a retrained projection."
        ),
    },
    "qwen_audio": {
        "model_name": "Qwen/Qwen2-Audio-7B-Instruct",
        "loader": "qwen_audio",
        "description": (
            "LLM whose weights were fine-tuned on audio soft tokens (Qwen2-Audio). "
            "We discard the Whisper encoder and inject our VAE-decoded embeddings "
            "through a retrained projection."
        ),
    },
    "plain_llm": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "loader": "causal_lm",
        "description": (
            "Text-only LLM with no multimodal pre-training (Mistral-7B). "
            "Serves as the baseline: any gain from the other two scenarios "
            "must come from their multimodal pre-training, not model capacity."
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_llm(scenario: str, load_in_8bit: bool = False):
    """
    Load the LLM backbone for the given scenario.

    Returns (llm_module, d_llm) where llm_module is an nn.Module that
    accepts inputs_embeds + attention_mask and returns an object with
    a .hidden_states attribute when output_hidden_states=True.
    """
    entry  = SCENARIO_REGISTRY[scenario]
    name   = entry["model_name"]
    loader = entry["loader"]

    kwargs = dict(output_hidden_states=True)
    if load_in_8bit:
        kwargs["load_in_8bit"] = True       # requires bitsandbytes

    if loader == "llava":
        from transformers import LlavaForConditionalGeneration
        model  = LlavaForConditionalGeneration.from_pretrained(name, **kwargs)
        llm    = model.language_model        # LlamaForCausalLM
        d_llm  = model.config.text_config.hidden_size

    elif loader == "qwen_audio":
        from transformers import Qwen2AudioForConditionalGeneration
        model  = Qwen2AudioForConditionalGeneration.from_pretrained(
            name, trust_remote_code=True, **kwargs
        )
        llm    = model.language_model        # Qwen2ForCausalLM
        d_llm  = model.config.text_config.hidden_size

    else:  # "causal_lm"
        llm   = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        d_llm = llm.config.hidden_size

    return llm, d_llm


def _build_projection(d_shared: int, n_soft_tokens: int, d_llm: int) -> nn.Module:
    """
    Two-layer MLP that maps the decoded shared embedding to n_soft_tokens
    vectors in the LLM's token embedding space.

    Mirrors the design of LLaVA's vision projection layer.

    Input:  [B, d_shared]
    Output: [B, n_soft_tokens, d_llm]
    """
    return nn.Sequential(
        nn.Linear(d_shared, d_llm),
        nn.GELU(),
        nn.Linear(d_llm, n_soft_tokens * d_llm),
    )


# ---------------------------------------------------------------------------
# ServerPipeline
# ---------------------------------------------------------------------------

class ServerPipeline(nn.Module):
    """
    Server-side pipeline for all three scenarios.

    Multi-task: a single LLM produces one shared hidden state that is
    passed to two independent task heads simultaneously —
        sentiment_head  → regression  (continuous score, MSE loss)
        emotion_head    → multi-label classification (6 emotions, BCE loss)

    This demonstrates that one server-side LLM can serve multiple tasks
    from the same compressed multimodal representation.

    Forward returns:
        {
            "sentiment": [B, 1]   raw regression score
            "emotions":  [B, 6]   raw logits (apply sigmoid for probabilities)
        }
    """

    def __init__(self, llm: nn.Module, d_llm: int,
                 projection: nn.Module,
                 sentiment_head: nn.Module,
                 emotion_head: nn.Module,
                 n_soft_tokens: int, freeze_llm: bool = True,
                 vae_decoder: VAEDecoder = None):
        super().__init__()
        self.vae_decoder    = vae_decoder
        self.projection     = projection
        self.llm            = llm
        self.sentiment_head = sentiment_head
        self.emotion_head   = emotion_head
        self.n_soft_tokens  = n_soft_tokens

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def forward(self, z: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        B = z.size(0)

        # --- VAE decode (skipped when use_compression=False) ----------
        emb = self.vae_decoder(z) if self.vae_decoder is not None else z  # [B, d_shared]

        # --- Project → soft tokens ------------------------------------
        soft = self.projection(emb)                            # [B, n * d_llm]
        soft = soft.view(B, self.n_soft_tokens, -1)            # [B, n, d_llm]

        # --- Build combined input embeddings --------------------------
        text_emb  = self.llm.get_input_embeddings()(input_ids) # [B, L, d_llm]
        combined  = torch.cat([soft, text_emb], dim=1)         # [B, n+L, d_llm]

        soft_mask = torch.ones(B, self.n_soft_tokens,
                               device=z.device,
                               dtype=attention_mask.dtype)
        full_mask = torch.cat([soft_mask, attention_mask], dim=1)  # [B, n+L]

        # --- LLM forward — one pass, shared representation -----------
        out = self.llm(
            inputs_embeds=combined,
            attention_mask=full_mask,
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1][:, -1, :]          # [B, d_llm]

        # --- Two task heads on the same hidden state -----------------
        return {
            "sentiment": self.sentiment_head(last_hidden),     # [B, 1]
            "emotions":  self.emotion_head(last_hidden),       # [B, 6]
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_server_pipeline(config: EMMAConfig = None,
                           vae_checkpoint: str = None) -> ServerPipeline:
    """
    Build a ServerPipeline for the scenario specified in config.server.scenario.

    The VAE decoder is loaded from the same checkpoint used by EdgePipeline.
    Pass the path via vae_checkpoint; if None the decoder is randomly
    initialised (useful only for architecture tests).

    Trainable components:   vae_decoder, projection, sentiment_head, emotion_head
    Frozen by default:      llm backbone
    """
    cfg = config or EMMAConfig()
    sc  = cfg.server
    vc  = cfg.vae
    ac  = cfg.alignment

    llm, d_llm = _load_llm(sc.scenario, sc.load_in_8bit)

    if cfg.use_compression:
        if vae_checkpoint is not None:
            vae_decoder = VAE.load(vae_checkpoint).decoder
        else:
            vae_decoder = VAEDecoder(
                d_latent=vc.d_latent,
                d_out=ac.d_shared,
                hidden_dims=tuple(reversed(vc.encoder_hidden_dims)),
            )
    else:
        vae_decoder = None

    projection     = _build_projection(ac.d_shared, sc.n_soft_tokens, d_llm)
    sentiment_head = nn.Linear(d_llm, 1)
    emotion_head   = nn.Linear(d_llm, 6)

    return ServerPipeline(
        llm=llm,
        d_llm=d_llm,
        vae_decoder=vae_decoder,
        projection=projection,
        sentiment_head=sentiment_head,
        emotion_head=emotion_head,
        n_soft_tokens=sc.n_soft_tokens,
        freeze_llm=sc.freeze_llm,
    )
