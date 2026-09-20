"""
Quantization-Aware Training (QAT) -- third item on the compression list,
after PTQ (ptq.py / static_ptq.py / onnx_static.py).

Unlike PTQ (quantize an already-finished model, optionally with a
calibration pass), QAT fine-tunes the model WITH fake-quantization in
the loop, so gradients see the rounding/clipping noise and the weights
adapt to be robust to it before quantization is ever "finalized." The
expectation, per the standard literature result, is that QAT recovers
more accuracy than PTQ at the same bit-width, at the cost of needing an
actual training loop.

Two things PTQ didn't need that QAT does:
  1. Gradients through round() -- round's gradient is zero almost
     everywhere, which would kill all learning. The standard fix is the
     Straight-Through Estimator (STE): use the rounded value in the
     forward pass, but pretend round() was the identity function in the
     backward pass. Implemented here via the `x + (round(x) - x).detach()`
     trick -- numerically equals round(x) forward, gradient is exactly 1
     backward, no custom autograd.Function needed.
  2. Training data. We don't have CLIP/MobileCLIP's original training
     data or labels, so fine-tuning uses distillation: minimize the
     distance between the fake-quantized ("student") model's output and
     the frozen original fp32 ("teacher") model's output, over real COCO
     images -- no labels needed, since the target is just "match your
     own unquantized self."

Scope note: unlike the ONNX static-quant path, this doesn't touch ONNX
export at all -- it's pure PyTorch autograd -- so it isn't blocked by the
text-encoder export bug that limited eval_onnx_ptq.py to image encoders
only. Both towers can be QAT-fine-tuned here.
"""

import copy

import torch
import torch.nn as nn


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through round: round() forward, identity gradient backward."""
    return x + (torch.round(x) - x).detach()


class QATLinear(nn.Module):
    """
    nn.Linear replacement with fake-quantized weights and activations,
    both recomputed every forward pass (unlike static_ptq.py's frozen
    calibration parameters) so gradients can push the underlying weight
    to become more quantization-robust during fine-tuning.

    Weight scale: recomputed fresh from the current (training) weight
    each forward -- no calibration needed, weights are always available.
    Activation scale: an exponential moving average of observed min/max,
    updated only in training mode -- the QAT analogue of BatchNorm's
    running statistics, and the standard way activation ranges are
    tracked during quantization-aware fine-tuning.
    """

    def __init__(self, linear: nn.Linear, momentum: float = 0.1):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.momentum = momentum
        self.register_buffer("act_min", torch.tensor(0.0))
        self.register_buffer("act_max", torch.tensor(0.0))
        self.register_buffer("_observed", torch.tensor(False))

    def _fake_quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        """
        Per-output-channel (per-row) scale/zero-point -- nn.Linear's
        weight is [out_features, in_features], so each row is one output
        channel/filter. Same fix that took PTQ's CLIP accuracy from 0.835
        to 0.946 (per-tensor forces every channel to share one scale
        calibrated to the widest range in the whole tensor).
        """
        w_min = w.min(dim=1, keepdim=True).values
        w_max = w.max(dim=1, keepdim=True).values
        w_min = torch.minimum(w_min, torch.zeros_like(w_min))
        w_max = torch.maximum(w_max, torch.zeros_like(w_max))
        scale = torch.clamp((w_max - w_min) / 255.0, min=1e-8)
        zero_point = _ste_round(-128 - w_min / scale)
        zero_point = torch.clamp(zero_point, -128, 127)
        q = torch.clamp(_ste_round(w / scale) + zero_point, -128, 127)
        return (q - zero_point) * scale

    def _fake_quant_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_min, batch_max = x.min().detach(), x.max().detach()
            if not bool(self._observed):
                self.act_min, self.act_max = batch_min, batch_max
                self._observed = torch.tensor(True)
            else:
                self.act_min = (1 - self.momentum) * self.act_min + self.momentum * batch_min
                self.act_max = (1 - self.momentum) * self.act_max + self.momentum * batch_max
        act_min = torch.minimum(self.act_min, torch.zeros_like(self.act_min))
        act_max = torch.maximum(self.act_max, torch.zeros_like(self.act_max))
        scale = torch.clamp((act_max - act_min) / 255.0, min=1e-8)
        zero_point = torch.clamp(_ste_round(-128 - act_min / scale), -128, 127)
        q = torch.clamp(_ste_round(x / scale) + zero_point, -128, 127)
        return (q - zero_point) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._fake_quant_activation(x)
        w_q = self._fake_quant_weight(self.weight)
        return nn.functional.linear(x_q, w_q, self.bias)


def convert_to_qat(encoder: nn.Module) -> nn.Module:
    """
    Replaces every nn.Linear with QATLinear, sharing (not copying) the
    original weight/bias Parameters -- so QATLinear's forward is what
    gets fine-tuned, and the underlying weight values it wraps are the
    ones that actually update.
    Source encoders in this project default to freeze_base=True (see
    emma/encoders/*.py), so deepcopy would otherwise carry
    requires_grad=False straight through -- explicitly re-enable it on
    every wrapped Linear's weight/bias, since those are specifically what
    QAT fine-tuning needs to update. Everything else (LayerNorm,
    embeddings, etc.) stays exactly as frozen/unfrozen as the source.
    """
    encoder = copy.deepcopy(encoder)
    for name, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                qat_linear = QATLinear(child)
                qat_linear.weight.requires_grad_(True)
                if qat_linear.bias is not None:
                    qat_linear.bias.requires_grad_(True)
                setattr(module, child_name, qat_linear)
    return encoder


def distillation_loss(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity, averaged over the batch -- no labels needed,
    the frozen fp32 model's own output is the training target."""
    return (1 - nn.functional.cosine_similarity(student_out, teacher_out, dim=-1)).mean()
