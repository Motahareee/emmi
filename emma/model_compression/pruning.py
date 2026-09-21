"""
Structured pruning -- third compression family (distinct from PTQ/QAT,
which only change numerical precision; this actually shrinks weight
matrices, removing real FLOPs regardless of what kernel/backend runs
the model, which is why it doesn't share quantization's "need a real
int8 kernel to see the latency win" problem).

Scope: CLIP ViT's MLP blocks only, for this first pass. Each transformer
block's MLP is `fc1: Linear(768, 3072) -> GELU -> fc2: Linear(3072, 768)`
(see CLIPMLP) -- pruning channels out of the 3072-dim intermediate width
is the standard, safest place to start structured pruning in a
transformer, because that dimension is purely internal to one MLP block:
nothing else in the network (attention heads, the residual stream,
other blocks) depends on it, unlike pruning attention head dimensions or
embedding width, which cascade through skip connections and shared
dimensions across the whole model. MobileCLIP's MLPs are Conv-based
(1x1 convs, not nn.Linear) -- same idea would apply to Conv2d output/
input channels, not implemented yet.

Unstructured pruning (zeroing individual weights without removing them)
is deliberately not what this does -- it needs sparse-matrix hardware
support to turn into an actual speedup, which most CPUs/edge devices
don't have. Removing whole rows/columns (structured) shrinks the actual
matrix shapes, so the FLOPs reduction is real on any hardware.
"""

import copy

import torch
import torch.nn as nn


def compute_channel_importance(fc1: nn.Linear, criterion: str = "l2") -> torch.Tensor:
    """
    Per-output-channel importance score for fc1's 3072 intermediate
    channels -- each row of fc1.weight is one channel's full set of
    incoming weights. L2 norm is the standard cheap proxy: a channel
    whose weights are all near zero contributes almost nothing to the
    layer's output regardless of the input, so it's a safe pruning
    candidate.
    """
    if criterion == "l2":
        return fc1.weight.data.norm(dim=1)
    elif criterion == "l1":
        return fc1.weight.data.abs().sum(dim=1)
    raise ValueError(f"unknown criterion: {criterion!r}")


def prune_mlp_pair(fc1: nn.Linear, fc2: nn.Linear, prune_ratio: float,
                   criterion: str = "l2") -> tuple:
    """
    Removes the lowest-importance prune_ratio fraction of fc1's output
    channels (and the matching fc2 input channels -- they have to move
    together, since fc2 reads exactly the channels fc1 produces).

    Returns (new_fc1, new_fc2) -- genuinely smaller Linear layers with
    the surviving rows/columns copied over, not the original layers with
    anything zeroed out.
    """
    importance = compute_channel_importance(fc1, criterion)
    n_keep = max(1, int(round(fc1.out_features * (1 - prune_ratio))))
    keep_idx = importance.topk(n_keep).indices.sort().values

    new_fc1 = nn.Linear(fc1.in_features, n_keep, bias=fc1.bias is not None)
    new_fc1.weight.data = fc1.weight.data[keep_idx].clone()
    if fc1.bias is not None:
        new_fc1.bias.data = fc1.bias.data[keep_idx].clone()

    new_fc2 = nn.Linear(n_keep, fc2.out_features, bias=fc2.bias is not None)
    new_fc2.weight.data = fc2.weight.data[:, keep_idx].clone()
    if fc2.bias is not None:
        new_fc2.bias.data = fc2.bias.data.clone()  # fc2's output width is untouched

    return new_fc1, new_fc2


def prune_clip_vit_mlps(image_encoder: nn.Module, prune_ratio: float,
                        criterion: str = "l2") -> nn.Module:
    """
    Applies prune_mlp_pair to every transformer block's MLP in a CLIP
    ImageEncoder (emma/encoders/image_encoder.py). Returns a new,
    smaller encoder -- the input is left untouched.
    """
    encoder = copy.deepcopy(image_encoder)
    for layer in encoder.vision_model.encoder.layers:
        new_fc1, new_fc2 = prune_mlp_pair(layer.mlp.fc1, layer.mlp.fc2, prune_ratio, criterion)
        layer.mlp.fc1 = new_fc1
        layer.mlp.fc2 = new_fc2
    return encoder
