"""
Post-training dynamic quantization (PTQ) for the edge encoders (CLIP,
MobileCLIP). First baseline in the encoder-compression rollout -- see the
branch's own history for the fuller PTQ -> QAT -> structured pruning ->
low-rank plan.

Quantizes nn.Linear weights to int8 with no calibration data and no
fine-tuning; activation scales are computed dynamically per-batch at
inference time. This is the cheapest PTQ variant to apply, which is
exactly why it's first: a fast baseline before the recovery-requiring
techniques (QAT, structured pruning) are worth the extra engineering.

Only nn.Linear is targeted -- PyTorch's native dynamic-quantization
kernels only cover Linear/LSTM/GRU, not Conv2d. CLIP's ViT is almost all
Linear layers, so this should quantize most of it; MobileCLIP's image
encoder (MCi, a FastViT-based hybrid CNN-transformer) mixes Conv2d
stem/blocks with Linear attention/MLP layers, so a smaller fraction of
its weights are eligible here. That's an expected, reportable difference
in achievable compression between the two models, not a bug in this code.

torch.ao.quantization is deprecated in favor of torchao's eager-mode
quantize_ API (as of torch 2.11), but still functional and torchao isn't
an existing project dependency -- using the legacy API for this first
pass; migrating is a reasonable future cleanup, not a blocker.
"""

import copy
import io
import platform

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear


def select_quantized_engine() -> str:
    """
    torch.backends.quantized.engine defaults to 'x86' regardless of actual
    CPU architecture -- on aarch64 (e.g. Apple Silicon under Docker, ARM
    edge devices) that default silently fails at quantize_dynamic() time
    ("unknown architecture") rather than at import time. Pick the engine
    that matches the actual machine instead of trusting the default.
    """
    machine = platform.machine().lower()
    preferred = "qnnpack" if machine in ("aarch64", "arm64") else "x86"
    supported = torch.backends.quantized.supported_engines
    engine = preferred if preferred in supported else supported[0]
    torch.backends.quantized.engine = engine
    return engine


def quantize_encoder_ptq(encoder: nn.Module) -> nn.Module:
    """
    Returns a new module with all nn.Linear submodules dynamically
    quantized to int8. Non-Linear layers (Conv2d, LayerNorm, attention
    softmax, etc.) are left at their original precision -- see module
    docstring for why.

    Runs on CPU only -- PyTorch's int8 dynamic-quant kernels (fbgemm/
    qnnpack/onednn) are CPU backends, no CUDA support. This happens to
    match the project's actual target (on-device/edge inference, not
    datacenter GPU), so CPU-only isn't a limitation for evaluating this
    specific technique -- it's the realistic deployment target anyway.

    The input encoder is left untouched (quantize_dynamic does not
    mutate in place); use the returned module for quantized inference.
    """
    select_quantized_engine()
    orig_dtype = next(encoder.parameters()).dtype
    encoder = copy.deepcopy(encoder).to("cpu").eval()
    quantized = quantize_dynamic(encoder, {nn.Linear}, dtype=torch.qint8)

    # open_clip's transformer blocks introspect a Linear's original dtype
    # via `self.mlp.c_fc.weight.dtype` (see ResidualAttentionBlock.
    # get_weight_dtype) -- that breaks on a quantized Linear, where
    # `.weight` is a method, not a tensor attribute. open_clip *does*
    # already special-case its own int8 modules via a `hasattr(...,
    # 'int8_original_dtype')` check first; PyTorch's native dynamic-quant
    # Linear just doesn't set that attribute. Set it ourselves so
    # open_clip-based models (MobileCLIP) don't need any other change.
    for module in quantized.modules():
        if isinstance(module, DynamicQuantizedLinear):
            module.int8_original_dtype = orig_dtype

    return quantized


def model_size_mb(model: nn.Module) -> float:
    """Serialized state_dict size in MB -- the actual compression signal."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)
