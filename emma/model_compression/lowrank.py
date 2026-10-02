"""
Low-rank approximation -- fourth and final compression family from this
project's original research question (after PTQ, QAT, structured
pruning). Unlike pruning (delete whole channels) or quantization
(reduce numerical precision), this factors each weight matrix into a
product of two smaller matrices that approximates it:

    W (shape [out, in]) ~ U_r @ diag(S_r) @ V_r  (rank r << min(out, in))

via truncated SVD -- the optimal rank-r approximation of W in the
least-squares sense (Eckart-Young theorem). One big Linear(in, out)
becomes two smaller ones in sequence: Linear(in, r, bias=False) then
Linear(r, out, bias=original's bias) -- same drop-in-replacement pattern
as pruning.py's new_fc1/new_fc2, just with the SVD factors as weights
instead of a channel subset.

Scope: same as pruning.py -- CLIP ViT's MLP blocks (fc1/fc2), the
"purely internal, nothing cascades" safe pruning target. Not yet
extended to attention projections or MobileCLIP's Conv2d layers (SVD on
a 4D conv weight needs reshaping to 2D first, e.g. per-spatial-position
or via a different factorization scheme -- out of scope for this pass).

Compression math: original Linear(in, out) has in*out parameters.
Rank-r factorization has r*(in+out). This is a net saving only when
r < in*out / (in+out) -- for CLIP ViT-B/32's fc1 (768->3072), that
breakeven is r < 614 (out of a full rank of 768), i.e. this only pays
off at ratio > ~0.2 in the convention below. Unlike pruning, where size
savings are linear in ratio, low-rank savings are back-loaded: small
ratios can net ZERO compression, or even increase parameter count.
"""

import copy

import torch
import torch.nn as nn


def compute_low_rank_factors(linear: nn.Linear, rank: int) -> tuple:
    """
    Truncated SVD of `linear`'s weight into two smaller Linear layers.

    W = U @ diag(S) @ Vh  (full SVD, U:[out,k], S:[k], Vh:[k,in], k=min(out,in))
    Keep only the top `rank` singular values/vectors (the ones capturing
    the most of W's variance) -- discarding the rest is the optimal
    rank-r approximation in Frobenius norm (Eckart-Young).

    Returns (A, B) where A = Linear(in, rank, bias=False) with weight
    Vh_r, and B = Linear(rank, out, bias=linear's bias) with weight
    U_r * S_r (columns of U_r scaled by their singular values) -- so
    B(A(x)) = U_r @ diag(S_r) @ Vh_r @ x + bias, approximating
    linear(x) = W @ x + bias.
    """
    W = linear.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    A = nn.Linear(linear.in_features, rank, bias=False)
    A.weight.data = Vh_r.clone()

    B = nn.Linear(rank, linear.out_features, bias=linear.bias is not None)
    B.weight.data = (U_r * S_r.unsqueeze(0)).clone()
    if linear.bias is not None:
        B.bias.data = linear.bias.data.clone()

    return A, B


def _rank_for_ratio(full_rank: int, ratio: float) -> int:
    """
    Converts a pruning-style ratio (fraction of capacity removed) into a
    concrete rank, for a consistent CLI convention with pruning.py's
    --ratio. full_rank = min(in_features, out_features) -- the largest
    rank that reproduces W exactly (no approximation error at all).
    """
    return max(1, int(round(full_rank * (1 - ratio))))


def lowrank_mlp_pair(fc1: nn.Linear, fc2: nn.Linear, ratio: float) -> tuple:
    """
    Low-rank analogue of pruning.py's prune_mlp_pair -- replaces both
    fc1 and fc2 with rank-reduced factorizations at the same `ratio`.
    Each becomes an nn.Sequential(A, B) of two smaller Linears, which is
    a drop-in replacement anywhere the original Linear was called
    (identical __call__ interface), same as how prune_mlp_pair's outputs
    slot into layer.mlp.fc1/fc2 unchanged.

    full_rank is computed separately for fc1 and fc2 since they have
    different (in, out) shapes in general (though for CLIP ViT-B/32's
    768<->3072 MLP, min(in,out)=768 for both, by coincidence of this
    specific architecture).
    """
    rank1 = _rank_for_ratio(min(fc1.in_features, fc1.out_features), ratio)
    rank2 = _rank_for_ratio(min(fc2.in_features, fc2.out_features), ratio)

    a1, b1 = compute_low_rank_factors(fc1, rank1)
    a2, b2 = compute_low_rank_factors(fc2, rank2)

    return nn.Sequential(a1, b1), nn.Sequential(a2, b2)


def lowrank_clip_vit_mlps(image_encoder: nn.Module, ratio: float) -> nn.Module:
    """
    Applies lowrank_mlp_pair to every transformer block's MLP in a CLIP
    ImageEncoder -- same traversal pattern as pruning.py's
    prune_clip_vit_mlps. Returns a new, smaller encoder; the input is
    left untouched.
    """
    encoder = copy.deepcopy(image_encoder)
    for layer in encoder.vision_model.encoder.layers:
        new_fc1, new_fc2 = lowrank_mlp_pair(layer.mlp.fc1, layer.mlp.fc2, ratio)
        layer.mlp.fc1 = new_fc1
        layer.mlp.fc2 = new_fc2
    return encoder
