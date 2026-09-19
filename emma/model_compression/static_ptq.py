"""
Static PTQ (calibration-based), second of the three quantization variants
being compared -- see ptq.py for dynamic (the first).

Unlike dynamic quantization, activation scale/zero-point are computed
ONCE during a calibration pass over real data (via forward hooks that
observe each nn.Linear's input min/max), then frozen. At inference there
is no per-call min/max computation -- exactly the overhead dynamic
quantization pays on every call, and the leading suspect for why CLIP
got *slower* after dynamic quantization (see eval_ptq.py's results).

Implementation note: this simulates static quantization via a fake-quant
round-trip (quantize then immediately dequantize, in fp32 arithmetic)
rather than PyTorch's full eager-mode static-quant pipeline (QuantStub/
DeQuantStub + module fusion + prepare/convert). That pipeline assumes a
model's forward() is written in a quantization-friendly style with
explicit stub placement; arbitrary HF/open_clip transformer forward
passes aren't, and eager-mode static quant tends to fail or silently
skip most of an architecture like this. Fake-quant gives the CORRECT
accuracy/fidelity signal (the same rounding+clipping error a real int8
static kernel would introduce) without needing PyTorch's real int8
kernels to accept this architecture -- but it means the *latency* number
for this variant reflects fp32 compute plus cheap round/clamp ops, not a
true int8 kernel's speed. That's still a meaningful, honestly-labeled
data point: it isolates "what does removing dynamic's per-call
overhead do" from "what does a real int8 kernel do", which real static
kernels would only improve on further.
"""

import copy

import torch
import torch.nn as nn


class _ActivationObserver:
    def __init__(self):
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def __call__(self, module, inputs, output):
        x = inputs[0]
        self.min_val = min(self.min_val, x.min().item())
        self.max_val = max(self.max_val, x.max().item())


def _compute_qparams(min_val: float, max_val: float, qmin: int = -128, qmax: int = 127):
    # Always include 0 in the range -- otherwise float 0.0 (a very common
    # value: padding, masked positions, ReLU outputs) has no exact int8
    # representation, which introduces bias rather than just rounding noise.
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    scale = max((max_val - min_val) / (qmax - qmin), 1e-8)
    zero_point = int(min(max(round(qmin - min_val / scale), qmin), qmax))
    return scale, zero_point


def _fake_quant(x: torch.Tensor, scale: float, zero_point: int,
                qmin: int = -128, qmax: int = 127) -> torch.Tensor:
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return (q - zero_point) * scale


class StaticQuantLinear(nn.Module):
    """
    nn.Linear replacement using frozen, calibration-derived quantization
    on both the input activation (scale/zero-point from calibration) and
    the weight (scale/zero-point from the weight tensor itself -- no
    calibration data needed for that half, it's static and known ahead
    of time either way).

    The weight is stored as a genuine int8 buffer (not just fake-quantized
    fp32) so this actually shrinks on disk/in memory, not just in the
    forward-pass rounding error -- dequantized back to fp32 each forward
    call before the matmul, which is the "simulated" part (a real static
    kernel would matmul directly in int8; see module docstring for why
    that's not what's being measured here).
    """

    def __init__(self, linear: nn.Linear, act_scale: float, act_zero_point: int):
        super().__init__()
        self.act_scale, self.act_zero_point = act_scale, act_zero_point
        self.bias = linear.bias

        w = linear.weight.data
        self.w_scale, self.w_zero_point = _compute_qparams(w.min().item(), w.max().item())
        w_int8 = torch.clamp(torch.round(w / self.w_scale) + self.w_zero_point, -128, 127)
        self.register_buffer("w_int8", w_int8.to(torch.int8))

        # open_clip's transformer introspects a Linear's original dtype
        # via `self.mlp.c_fc.weight.dtype` (see ResidualAttentionBlock.
        # get_weight_dtype in ptq.py's docstring for the full story) --
        # same compatibility shim as the dynamic-quant path.
        self.int8_original_dtype = w.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = _fake_quant(x, self.act_scale, self.act_zero_point)
        w = (self.w_int8.float() - self.w_zero_point) * self.w_scale
        return nn.functional.linear(x_q, w, self.bias)


@torch.no_grad()
def calibrate_and_quantize(encoder: nn.Module, run_calibration) -> nn.Module:
    """
    run_calibration: callable(copied_encoder) -> None. Must call the
    encoder (or its submodules directly, e.g. `.model.encode_text(...)`
    for MobileCLIP's shared backbone -- see eval_static_ptq.py) with
    representative real data at least once; return value is ignored,
    this only exists to trigger the observer hooks below.

    The input encoder is left untouched; returns a new, frozen-static-
    quantized module.
    """
    encoder = copy.deepcopy(encoder).eval()

    linears = [(name, m) for name, m in encoder.named_modules() if isinstance(m, nn.Linear)]
    observers, handles = {}, []
    for name, m in linears:
        obs = _ActivationObserver()
        observers[name] = obs
        handles.append(m.register_forward_hook(obs))

    run_calibration(encoder)

    for handle in handles:
        handle.remove()

    for name, m in linears:
        obs = observers[name]
        if obs.min_val == float("inf"):
            continue  # never exercised during calibration -- leave as fp32, don't guess
        act_scale, act_zp = _compute_qparams(obs.min_val, obs.max_val)
        parent = encoder
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, StaticQuantLinear(m, act_scale, act_zp))

    return encoder
