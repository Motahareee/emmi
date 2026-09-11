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

def _load_llm(scenario: str, load_in_8bit: bool = False, dtype: str = "float16"):
    """
    Load the LLM backbone for the given scenario.

    dtype: "float16" (default, matches all prior runs) or "float32" -- diagnostic
    option to rule out fp16's limited dynamic range (~65504 max) as the cause of
    the non-finite-gradient instability seen on some (compressor, encoder) pairs.
    Ignored when load_in_8bit=True (quantized weights aren't touched by this).

    Returns (llm_module, d_llm) where llm_module is an nn.Module that
    accepts inputs_embeds + attention_mask and returns an object with
    a .hidden_states attribute when output_hidden_states=True.
    """
    entry  = SCENARIO_REGISTRY[scenario]
    name   = entry["model_name"]
    loader = entry["loader"]
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]

    kwargs = dict(output_hidden_states=True)
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)  # requires bitsandbytes

    if loader == "llava":
        from transformers import LlavaForConditionalGeneration
        model  = LlavaForConditionalGeneration.from_pretrained(name, **kwargs)
        llm    = model.model.language_model  # LlamaForCausalLM (nested under .model in transformers>=5.x)
        d_llm  = model.config.text_config.hidden_size
        if not load_in_8bit:
            # Some params (e.g. RMSNorm weights) can load in fp32 even when the
            # checkpoint is otherwise fp16, causing an internal dtype mismatch
            # partway through the forward pass. Normalize the whole backbone
            # to one dtype so every layer sees consistent input. Using
            # torch_dtype here (rather than always float16) lets a caller
            # request genuine fp32 -- unlike the load_in_8bit=False default,
            # that actually removes fp16's dynamic-range ceiling instead of
            # just avoiding the dtype-mismatch crash while staying in fp16.
            llm = llm.to(dtype=torch_dtype)

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

    Task: binary image-text matching.
        match_head → 1 logit (apply sigmoid for probability; BCE loss)

    Forward returns:
        {"match": [B, 1]}  raw logit (positive = image matches caption)
    """

    def __init__(self, llm: nn.Module, d_llm: int,
                 projection: nn.Module,
                 match_head: nn.Module,
                 n_soft_tokens: int, freeze_llm: bool = True,
                 vae_decoder: VAEDecoder = None,
                 debug: bool = False, debug_max_calls: int = 3,
                 debug_every: int = 0, debug_max_track: int = 200):
        super().__init__()
        self.vae_decoder   = vae_decoder
        self.projection    = projection
        self.llm           = llm
        self.match_head    = match_head
        self.n_soft_tokens = n_soft_tokens

        # Opt-in diagnostic instrumentation -- off by default and a no-op for
        # every existing caller unless debug=True is passed explicitly.
        #  - debug_max_calls: full stage-by-stage dump for the first N calls
        #    (min/max/isnan/isinf at each point in forward()).
        #  - debug_every: if >0, a lightweight one-line snapshot (projection
        #    weight norm + soft-token magnitude) every debug_every calls,
        #    capped at debug_max_track lines -- lets us see whether the
        #    projection's weights/outputs grow over the course of training,
        #    not just in the first few calls.
        #  - the first time any non-finite value appears anywhere, a full
        #    dump fires once regardless of the above counters, to capture
        #    exactly where/when it happens.
        self.debug              = debug
        self.debug_max_calls    = debug_max_calls
        self.debug_every        = debug_every
        self.debug_max_track    = debug_max_track
        self._debug_calls       = 0
        self._debug_call_idx    = 0
        self._debug_track_lines = 0
        self._debug_first_bad_reported = False

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def _debug_tensor(self, name: str, t: torch.Tensor):
        with torch.no_grad():
            print(f"    [debug]   {name}: dtype={t.dtype} "
                  f"min={t.min().item():.4g} max={t.max().item():.4g} "
                  f"isnan={torch.isnan(t).any().item()} isinf={torch.isinf(t).any().item()}")

    def forward(self, z: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        B = z.size(0)

        # --- VAE decode (skipped when use_compression=False) ----------
        emb = self.vae_decoder(z) if self.vae_decoder is not None else z  # [B, d_shared]

        # --- Project → soft tokens ------------------------------------
        soft = self.projection(emb)                            # [B, n * d_llm]
        soft = soft.view(B, self.n_soft_tokens, -1)            # [B, n, d_llm]

        self._debug_call_idx += 1

        debug_this_call = self.debug and self._debug_calls < self.debug_max_calls
        if debug_this_call:
            self._debug_calls += 1
            print(f"  [debug] ServerPipeline.forward call {self._debug_calls}/{self.debug_max_calls}")
            self._debug_tensor("z (input)",            z)
            self._debug_tensor("emb (post vae_decode)", emb)
            self._debug_tensor("soft (post projection)", soft)

        debug_track = (self.debug and self.debug_every > 0
                       and self._debug_track_lines < self.debug_max_track
                       and self._debug_call_idx % self.debug_every == 0)
        if debug_track:
            self._debug_track_lines += 1
            with torch.no_grad():
                proj_w_norm = sum(p.norm().item() for p in self.projection.parameters())
                print(f"  [debug-track] call {self._debug_call_idx}: "
                      f"projection_weight_norm={proj_w_norm:.4g} "
                      f"soft_absmax={soft.abs().max().item():.4g} "
                      f"soft_isfinite={torch.isfinite(soft).all().item()}")

        # --- Build combined input embeddings --------------------------
        text_emb  = self.llm.get_input_embeddings()(input_ids) # [B, L, d_llm]
        llm_dtype = next(self.llm.parameters()).dtype
        combined  = torch.cat([soft, text_emb], dim=1).to(dtype=llm_dtype)  # [B, n+L, d_llm]

        if debug_this_call:
            self._debug_tensor("text_emb",                text_emb)
            self._debug_tensor("combined (post cast)",     combined)

        soft_mask = torch.ones(B, self.n_soft_tokens,
                               device=z.device,
                               dtype=attention_mask.dtype)
        full_mask = torch.cat([soft_mask, attention_mask], dim=1)  # [B, n+L]

        # --- LLM forward ----------------------------------------------
        out = self.llm(
            inputs_embeds=combined,
            attention_mask=full_mask,
            output_hidden_states=True,
        )

        if debug_this_call:
            first_bad = None
            for i, h in enumerate(out.hidden_states):
                if torch.isnan(h).any() or torch.isinf(h).any():
                    first_bad = i
                    break
            if first_bad is None:
                print(f"    [debug]   all {len(out.hidden_states)} hidden_states "
                      f"(embeddings + each layer) are finite")
            else:
                print(f"    [debug]   FIRST non-finite hidden_states at index {first_bad} "
                      f"(0=embeddings, 1..N=after layer i) out of {len(out.hidden_states)}")
                self._debug_tensor(f"hidden_states[{first_bad}]", out.hidden_states[first_bad])
                self._debug_first_bad_reported = True
        elif self.debug and not self._debug_first_bad_reported:
            # Beyond the first debug_max_calls: still watch for the FIRST
            # non-finite forward anywhere, and dump full context exactly
            # when/where it happens -- this is what actually tells us
            # whether the projection has drifted into instability partway
            # through training, rather than only seeing healthy early calls.
            bad_idx = None
            for i, h in enumerate(out.hidden_states):
                if torch.isnan(h).any() or torch.isinf(h).any():
                    bad_idx = i
                    break
            if bad_idx is not None:
                self._debug_first_bad_reported = True
                print(f"  [debug] FIRST non-finite forward at call {self._debug_call_idx} "
                      f"(hidden_states index {bad_idx} of {len(out.hidden_states)})")
                self._debug_tensor("z (input)",              z)
                self._debug_tensor("soft (post projection)", soft)
                with torch.no_grad():
                    proj_w_norm = sum(p.norm().item() for p in self.projection.parameters())
                    print(f"    [debug]   projection weight norm at failure: {proj_w_norm:.4g}")
                self._debug_tensor(f"hidden_states[{bad_idx}]", out.hidden_states[bad_idx])

        # match_head may be a different (e.g. fp32) dtype than the LLM backbone
        last_hidden = out.hidden_states[-1][:, -1, :].to(dtype=self.match_head.weight.dtype)  # [B, d_llm]

        if debug_this_call:
            self._debug_tensor("last_hidden (final)", last_hidden)

        # --- Binary classification head -------------------------------
        return {"match": self.match_head(last_hidden)}         # [B, 1]


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

    Trainable components:   vae_decoder, projection, match_head
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

    projection = _build_projection(ac.d_shared, sc.n_soft_tokens, d_llm)
    match_head = nn.Linear(d_llm, 1)

    return ServerPipeline(
        llm=llm,
        d_llm=d_llm,
        vae_decoder=vae_decoder,
        projection=projection,
        match_head=match_head,
        n_soft_tokens=sc.n_soft_tokens,
        freeze_llm=sc.freeze_llm,
    )
