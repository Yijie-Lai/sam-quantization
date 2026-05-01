# This file is modified based on the EfficientQAT project.

import torch
import torch.nn as nn

CLIPMIN = 1e-4


# =======================
# Rounding Strategy API
# =======================

class BaseRound(nn.Module):
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class STE_Round(BaseRound):
    """Straight-Through Estimator rounding"""
    def forward(self, x):
        return (x.round() - x).detach() + x


class Identity_Round(BaseRound):
    """Identity mapping (no rounding)"""
    def forward(self, x):
        return x


class TanhRound(BaseRound):
    def __init__(self, t=1.0):
        super().__init__()
        self.t = t

    def forward(self, x):
        return TanhBoundaryRound.apply(x, self.t)


class TanhBoundaryRound(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, t):
        ctx.save_for_backward(x)
        ctx.t = t
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        t = ctx.t

        # floor(x)
        k = torch.floor(x)

        diff = x - k - 0.5

        # tanh boundary
        tanh_val = torch.tanh(t * diff)

        # dy/dx = 0.5 * t * (1 - tanh^2)
        grad = 0.5 * t * (1 - tanh_val ** 2)

        return grad_output * grad, None
    

def clamp_ste(x: torch.Tensor, min_val, max_val):
    """STE version of clamp"""
    return (x.clamp(min_val, max_val) - x).detach() + x


# =======================
# Quantizer
# =======================

class AffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        round_fn: BaseRound = None,
    ):
        super().__init__()

        assert 2 <= n_bits <= 16, "bitwidth not supported"

        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1

        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % self.group_size == 0

        self.enable = True

        # Default rounding is STE
        self.round_fn = round_fn if round_fn is not None else STE_Round()

        # Initialize scale and zero point using min-max quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1, self.group_size)
                xmin = x.amin([-1], keepdim=True)
                xmax = x.amax([-1], keepdim=True)

                _range = xmax - xmin
                scale = _range / (2**self.n_bits - 1)
                scale = scale.clamp(min=1e-4, max=1e4)

                zero_point = -(xmin / scale).clamp(min=-1e4, max=1e4)

                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())

    # =======================
    # API
    # =======================

    def set_round_fn(self, round_fn: BaseRound):
        """Dynamically set rounding strategy"""
        self.round_fn = round_fn

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** n_bits - 1)


    def fake_quant(self, x):
        scale = clamp_ste(self.scale, 1e-4, 1e4)

        round_zero_point = clamp_ste(
            self.round_fn(self.zero_point), self.qmin, self.qmax
        )

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)

        x_int = self.round_fn(x / scale)

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)

        x_int = x_int.clamp(self.qmin, self.qmax)

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)

        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)

        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        return self.fake_quant(x)