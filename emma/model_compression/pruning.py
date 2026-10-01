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
import torch.nn.functional as F


def compute_channel_importance(fc1: nn.Linear, criterion: str = "l2") -> torch.Tensor:
    """
    Per-output-channel importance score for fc1's 3072 intermediate
    channels -- each row of fc1.weight is one channel's full set of
    incoming weights. L2 norm is the standard cheap proxy: a channel
    whose weights are all near zero contributes almost nothing to the
    layer's output regardless of the input, so it's a safe pruning
    candidate.

    Report Finding 19: two independent negative results (more recovery
    epochs, iterative scheduling) both failed to move the ~50-64%
    accuracy ceiling, pointing away from "how training is scheduled" and
    toward "what gets removed" -- L2-magnitude is a known weak criterion
    in the pruning literature precisely because it ignores how a channel
    actually affects the loss. See compute_taylor_importance below for
    the stronger alternative.
    """
    if criterion == "l2":
        return fc1.weight.data.norm(dim=1)
    elif criterion == "l1":
        return fc1.weight.data.abs().sum(dim=1)
    raise ValueError(f"unknown criterion: {criterion!r}")


def compute_taylor_importance(image_encoder: nn.Module, text_encoder: nn.Module,
                              calib_pixel_values: torch.Tensor,
                              calib_input_ids: torch.Tensor,
                              calib_attention_mask: torch.Tensor) -> list:
    """
    First-order Taylor-expansion importance (Molchanov et al.) for every
    MLP block's fc1 output channels, in ONE forward+backward pass:

        importance_c ~ mean_over_batch,tokens(|activation_c * d(loss)/d(activation_c)|)

    i.e. how much the loss would change, to first order, if channel c's
    activation were zeroed out -- unlike L2-magnitude, this actually
    looks at the channel's effect on a real loss, not just its weight size.

    The loss used is CLIP's own contrastive (InfoNCE) objective on a
    calibration batch of real COCO image-caption pairs -- the actual
    training signal CLIP was optimized on. Deliberately NOT a
    distillation loss against a frozen copy of this same model: since
    student and teacher would be identical weights, that loss is exactly
    zero and its gradient carries no information about which channels
    matter.

    Returns a list of per-block importance tensors, one per transformer
    block, in the same order as image_encoder.vision_model.encoder.layers.
    """
    image_encoder.zero_grad(set_to_none=True)
    activations = []
    hooks = []

    def make_hook(store):
        def hook(module, inp, out):
            out.retain_grad()
            store.append(out)
        return hook

    for layer in image_encoder.vision_model.encoder.layers:
        hooks.append(layer.mlp.fc1.register_forward_hook(make_hook(activations)))

    image_embeds = image_encoder(calib_pixel_values)
    text_embeds = text_encoder(input_ids=calib_input_ids, attention_mask=calib_attention_mask)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logits = image_embeds @ text_embeds.T * 100.0  # CLIP's own fixed logit scale
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    loss.backward()

    for h in hooks:
        h.remove()

    importance = []
    for act in activations:
        grad = act.grad
        reduce_dims = tuple(range(act.dim() - 1))  # keep only the channel dim
        importance.append((act * grad).abs().mean(dim=reduce_dims).detach())

    image_encoder.zero_grad(set_to_none=True)
    return importance


def compute_hessian_importance(image_encoder: nn.Module, text_encoder: nn.Module,
                               calib_pixel_values: torch.Tensor,
                               calib_input_ids: torch.Tensor,
                               calib_attention_mask: torch.Tensor) -> list:
    """
    Second-order (diagonal-Hessian-approximated) channel importance --
    more principled than compute_taylor_importance's first-order linear
    approximation, which only captures how the loss changes to FIRST
    order when a channel is zeroed. The true change also has a
    second-order term: ΔL ≈ grad·Δw + 0.5·H·Δw². Computing the exact
    Hessian is intractable (it's a matrix over all weights), so this
    uses the standard practical approximation -- the diagonal empirical
    Fisher information, H_cc ≈ E[grad_c²] -- which makes the
    second-order term reduce to (grad_c · w_c)², i.e. the SQUARE of the
    first-order Taylor term computed in compute_taylor_importance. This
    is exactly the saliency used in classic Optimal Brain
    Damage/Surgeon-style pruning, adapted here to channel-level (not
    individual-weight) granularity.

    Same calibration setup as compute_taylor_importance (one
    forward+backward pass of CLIP's own contrastive loss) -- only the
    final reduction differs (squared instead of absolute value), so this
    duplicates that function's hook/loss machinery rather than refactor
    it out, keeping both functions independently readable.
    """
    image_encoder.zero_grad(set_to_none=True)
    activations = []
    hooks = []

    def make_hook(store):
        def hook(module, inp, out):
            out.retain_grad()
            store.append(out)
        return hook

    for layer in image_encoder.vision_model.encoder.layers:
        hooks.append(layer.mlp.fc1.register_forward_hook(make_hook(activations)))

    image_embeds = image_encoder(calib_pixel_values)
    text_embeds = text_encoder(input_ids=calib_input_ids, attention_mask=calib_attention_mask)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logits = image_embeds @ text_embeds.T * 100.0
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    loss.backward()

    for h in hooks:
        h.remove()

    importance = []
    for act in activations:
        grad = act.grad
        reduce_dims = tuple(range(act.dim() - 1))
        importance.append(((act * grad) ** 2).mean(dim=reduce_dims).detach())

    image_encoder.zero_grad(set_to_none=True)
    return importance


def compute_global_keep_indices(importances: list, prune_ratio: float) -> list:
    """
    Converts per-layer importance scores into per-layer keep-index lists
    using ONE global threshold across the whole network, instead of
    pruning each layer to the same fixed ratio independently. This lets
    pruning concentrate wherever the network can actually spare
    capacity -- a layer with lots of redundant channels can lose more
    than prune_ratio's worth, a layer with little redundancy can lose
    less, as long as the TOTAL removed across the network matches
    prune_ratio.

    Each layer's scores are first normalized by that layer's own max
    importance before combining -- raw scores aren't comparable across
    layers (e.g. later transformer blocks can have systematically larger
    activation/gradient magnitudes than earlier ones for reasons
    unrelated to how prunable they are), so without this a global
    ranking would just reflect which layer happens to have the largest
    scale, not which channels are least important. Standard practice in
    global-pruning literature (e.g. "Rethinking the Value of Network
    Pruning").

    Returns a list of per-layer keep_idx tensors (sorted ascending),
    one per layer in the same order as `importances` -- feed directly
    into prune_clip_vit_mlps' keep_indices param.
    """
    normalized = [imp / imp.max().clamp(min=1e-8) for imp in importances]
    flat = torch.cat(normalized)
    n_total = flat.numel()
    n_prune = int(round(n_total * prune_ratio))
    n_keep_total = n_total - n_prune
    if n_keep_total >= n_total:
        return [torch.arange(imp.numel()) for imp in importances]
    threshold = flat.topk(max(n_keep_total, 1), largest=True).values.min()

    keep_indices = []
    for imp in normalized:
        keep = (imp >= threshold).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:  # never fully empty a layer
            keep = imp.topk(1).indices
        keep_indices.append(keep.sort().values)
    return keep_indices


def prune_mlp_pair(fc1: nn.Linear, fc2: nn.Linear, prune_ratio: float,
                   criterion: str = "l2", importance: torch.Tensor = None,
                   keep_idx: torch.Tensor = None) -> tuple:
    """
    Removes the lowest-importance prune_ratio fraction of fc1's output
    channels (and the matching fc2 input channels -- they have to move
    together, since fc2 reads exactly the channels fc1 produces).

    If `keep_idx` is given directly (e.g. from compute_global_keep_indices,
    where the number kept per layer varies and isn't simply
    round(width * (1 - prune_ratio))), it's used as-is and both
    `criterion`/`importance` are ignored. Otherwise if `importance` is
    given (e.g. from compute_taylor_importance), it's used directly and
    `criterion` is ignored. Otherwise `criterion` computes it fresh.

    Returns (new_fc1, new_fc2) -- genuinely smaller Linear layers with
    the surviving rows/columns copied over, not the original layers with
    anything zeroed out.
    """
    if keep_idx is None:
        if importance is None:
            importance = compute_channel_importance(fc1, criterion)
        n_keep = max(1, int(round(fc1.out_features * (1 - prune_ratio))))
        keep_idx = importance.topk(n_keep).indices.sort().values
    n_keep = keep_idx.numel()

    new_fc1 = nn.Linear(fc1.in_features, n_keep, bias=fc1.bias is not None)
    new_fc1.weight.data = fc1.weight.data[keep_idx].clone()
    if fc1.bias is not None:
        new_fc1.bias.data = fc1.bias.data[keep_idx].clone()

    new_fc2 = nn.Linear(n_keep, fc2.out_features, bias=fc2.bias is not None)
    new_fc2.weight.data = fc2.weight.data[:, keep_idx].clone()
    if fc2.bias is not None:
        new_fc2.bias.data = fc2.bias.data.clone()  # fc2's output width is untouched

    return new_fc1, new_fc2


def compute_conv_channel_importance(fc1: nn.Conv2d, criterion: str = "l2") -> torch.Tensor:
    """
    Conv2d analogue of compute_channel_importance -- fc1.weight is
    [out_channels, in_channels, kh, kw] instead of Linear's 2D
    [out_features, in_features], so importance reduces over dims (1,2,3)
    instead of dim=1. For MobileCLIP's ConvMlp, fc1/fc2 are 1x1 convs
    (kh=kw=1), so this is numerically identical to the Linear case --
    kept as a separate function only because the tensor shape differs.
    """
    flat = fc1.weight.data.flatten(1)  # [out_channels, in_channels*kh*kw]
    if criterion == "l2":
        return flat.norm(dim=1)
    elif criterion == "l1":
        return flat.abs().sum(dim=1)
    raise ValueError(f"unknown criterion: {criterion!r}")


def prune_conv_mlp_pair(fc1: nn.Conv2d, fc2: nn.Conv2d, prune_ratio: float,
                        criterion: str = "l2", importance: torch.Tensor = None) -> tuple:
    """
    Conv2d analogue of prune_mlp_pair -- removes the lowest-importance
    prune_ratio fraction of fc1's output channels/filters (and the
    matching fc2 input channels). MobileCLIP's ConvMlp.fc1/fc2 are 1x1
    convs, so this is structurally the same operation as the Linear
    case, just indexing dim 0 (out_channels) and dim 1 (in_channels) of a
    4D weight tensor instead of a 2D one.
    """
    if importance is None:
        importance = compute_conv_channel_importance(fc1, criterion)
    n_keep = max(1, int(round(fc1.out_channels * (1 - prune_ratio))))
    keep_idx = importance.topk(n_keep).indices.sort().values

    new_fc1 = nn.Conv2d(fc1.in_channels, n_keep, kernel_size=fc1.kernel_size,
                        stride=fc1.stride, padding=fc1.padding, bias=fc1.bias is not None)
    new_fc1.weight.data = fc1.weight.data[keep_idx].clone()
    if fc1.bias is not None:
        new_fc1.bias.data = fc1.bias.data[keep_idx].clone()

    new_fc2 = nn.Conv2d(n_keep, fc2.out_channels, kernel_size=fc2.kernel_size,
                        stride=fc2.stride, padding=fc2.padding, bias=fc2.bias is not None)
    new_fc2.weight.data = fc2.weight.data[:, keep_idx].clone()
    if fc2.bias is not None:
        new_fc2.bias.data = fc2.bias.data.clone()  # fc2's output width is untouched

    return new_fc1, new_fc2


def prune_mobileclip_mlps(image_encoder: nn.Module, prune_ratio: float,
                          criterion: str = "l2") -> nn.Module:
    """
    Applies prune_conv_mlp_pair to every FastViT block's ConvMlp
    (block.mlp.fc1/fc2) across all 4 stages of MobileCLIP's image
    trunk (mc_img.model.visual.trunk.stages[*].blocks[*]) -- both
    RepMixerBlock (stages 0-2) and AttentionBlock (stage 3) have this
    same mlp.fc1/fc2 structure. mlp.conv (a depthwise conv that runs
    BEFORE fc1, at the block's original channel width) is untouched --
    it doesn't depend on fc1/fc2's hidden width, same "purely internal"
    safety property as CLIP's MLP intermediate dimension.
    """
    encoder = copy.deepcopy(image_encoder)
    for stage in encoder.model.visual.trunk.stages:
        for block in stage.blocks:
            new_fc1, new_fc2 = prune_conv_mlp_pair(block.mlp.fc1, block.mlp.fc2,
                                                   prune_ratio, criterion)
            block.mlp.fc1 = new_fc1
            block.mlp.fc2 = new_fc2
    return encoder


def prune_clip_vit_mlps(image_encoder: nn.Module, prune_ratio: float,
                        criterion: str = "l2", importances: list = None,
                        keep_indices: list = None) -> nn.Module:
    """
    Applies prune_mlp_pair to every transformer block's MLP in a CLIP
    ImageEncoder (emma/encoders/image_encoder.py). Returns a new,
    smaller encoder -- the input is left untouched.

    `keep_indices`, if given, must be a list of per-block keep_idx
    tensors (i.e. the output of compute_global_keep_indices) -- takes
    precedence over everything else, since global ranking determines a
    different channel count per layer that isn't expressible as a
    single `prune_ratio` + per-layer `importance` score.

    `importances`, if given (and `keep_indices` isn't), must be a list
    of per-block importance tensors in the same order as
    encoder.vision_model.encoder.layers (i.e. the output of
    compute_taylor_importance or compute_hessian_importance) -- one
    entry consumed per layer, overriding `criterion` for that layer.
    """
    encoder = copy.deepcopy(image_encoder)
    for i, layer in enumerate(encoder.vision_model.encoder.layers):
        k_idx = keep_indices[i] if keep_indices is not None else None
        imp = importances[i] if (importances is not None and k_idx is None) else None
        new_fc1, new_fc2 = prune_mlp_pair(layer.mlp.fc1, layer.mlp.fc2, prune_ratio,
                                          criterion, importance=imp, keep_idx=k_idx)
        layer.mlp.fc1 = new_fc1
        layer.mlp.fc2 = new_fc2
    return encoder
