# This file is modified based on the EfficientQAT project.

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize.quantizer import AffineQuantizer


class QuantLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64,
    ):
        super().__init__()

        self.fwd_func = F.linear

        # parameters
        self.register_parameter("weight", org_module.weight)
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias)
        else:
            self.bias = None

        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.group_size = group_size

        # ===== Base quant =====
        self.use_weight_quant = False

        self.weight_quantizer = AffineQuantizer(
            n_bits=wbits,
            group_size=group_size,
            weight=org_module.weight,
        )

        # ===== Progressive =====
        self.use_progressive = False
        self.secondary_quantizer = None

        self.r_shape_mode = None
        self.rho = None
        self.ema_momentum = None

        self.delta_w = None
        self.progressive_finalized = False

    def forward(self, x):
        weight = self.weight

        if self.use_weight_quant:
            if self.use_progressive:
                weight = self._forward_progressive(weight)
            else:
                weight = self.weight_quantizer(weight)

        return F.linear(x, weight, self.bias)


    def _forward_progressive(self, weight):
        if self.secondary_quantizer is None:
            raise RuntimeError("Progressive not initialized")

        if self.progressive_ratio is None:
            raise RuntimeError("progressive_ratio missing")

        w_source = self.weight_quantizer(weight)
        w_target = self.secondary_quantizer(weight)

        self.delta_w = w_target - w_source

        r = self._get_r_tensor()
        return w_source + r * self.delta_w

    def _get_r_tensor(self):
        if self.r_shape_mode == "scalar":
            return self.progressive_ratio

        elif self.r_shape_mode == "channel":
            return self.progressive_ratio

        elif self.r_shape_mode == "group":
            return self.progressive_ratio.repeat_interleave(
                self.group_size, dim=1
            )[:, : self.in_features]

        else:
            raise ValueError

    def set_quant_state(self, weight_quant=False):
        self.use_weight_quant = weight_quant


    def enable_progressive(
        self,
        target_bits=2,
        r_shape_mode="scalar",
        rho=0.05,
        ema_momentum=0.05,
    ):
        self.use_weight_quant = True
        self.use_progressive = True
        self.progressive_finalized = False

        self.r_shape_mode = r_shape_mode
        self.rho = rho
        self.ema_momentum = ema_momentum

        self.secondary_quantizer = AffineQuantizer(
            n_bits=target_bits,
            group_size=self.group_size,
            weight=self.weight,
        )

        device = self.weight.device

        if hasattr(self, "progressive_ratio"):
            delattr(self, "progressive_ratio")

        if r_shape_mode == "scalar":
            self.register_buffer("progressive_ratio", torch.tensor(0.0, device=device))

        elif r_shape_mode == "channel":
            self.register_buffer(
                "progressive_ratio",
                torch.zeros(self.out_features, 1, device=device),
            )

        elif r_shape_mode == "group":
            n_groups = (self.in_features + self.group_size - 1) // self.group_size
            self.register_buffer(
                "progressive_ratio",
                torch.zeros(self.out_features, n_groups, device=device),
            )

    def disable_progressive(self):
        self.use_progressive = False
        self.secondary_quantizer = None
        self.delta_w = None

        if "progressive_ratio" in self._buffers:
            del self._buffers["progressive_ratio"]

        self.progressive_ratio = None
        self.progressive_finalized = False

    @torch.no_grad()
    def finalize_progressive(self):
        """Switch the training forward to the deploy-time target quantizer."""
        if not self.use_progressive or self.progressive_ratio is None:
            return
        self.progressive_ratio.fill_(1.0)
        self.progressive_finalized = True


    @torch.no_grad()
    def update_progressive_ratio(self):
        if (
            not self.use_progressive
            or self.progressive_finalized
            or self.delta_w is None
            or self.weight.grad is None
            or self.progressive_ratio is None
        ):
            return

        grad = self.weight.grad

        if self.r_shape_mode == "scalar":
            directional_grad = torch.sum(grad * self.delta_w)

        elif self.r_shape_mode == "channel":
            directional_grad = torch.sum(grad * self.delta_w, dim=1, keepdim=True)

        elif self.r_shape_mode == "group":
            out, in_feat = grad.shape
            g = self.group_size
            n_groups = (in_feat + g - 1) // g
            directional_values = grad * self.delta_w
            padding = n_groups * g - in_feat
            if padding:
                directional_values = F.pad(directional_values, (0, padding))
            directional_grad = directional_values.reshape(
                out, n_groups, g
            ).sum(dim=2).to(self.progressive_ratio.dtype)

        r_target = torch.clamp(
            self.rho / (torch.abs(directional_grad) + 1e-8),
            max=1.0,
        )

        self.progressive_ratio.mul_(1 - self.ema_momentum).add_(
            self.ema_momentum * r_target
        )

    @torch.no_grad()
    def get_r_metric(self):
        if self.progressive_ratio is None:
            return None

        r = self.progressive_ratio

        stats = {
            "r_mean": r.mean().item(),
            "r_std": r.std().item(),
            "r_norm": r.norm().item(),
            "r_max": r.max().item(),
            "r_min": r.min().item(),
        }

        eps = 1e-8
        dir_grad_est = self.rho / (r + eps)

        stats.update({
            "dir_grad_mean": dir_grad_est.mean().item(),
            "dir_grad_max": dir_grad_est.max().item(),
            "dir_grad_min": dir_grad_est.min().item(),
            "r_saturated_ratio": (r > 0.999).float().mean().item(),
        })

        return stats

    @torch.no_grad()
    def get_directional_grad_stats(self):
        if (
            not self.use_progressive
            or self.delta_w is None
            or self.weight.grad is None
        ):
            return None

        grad = self.weight.grad

        if self.r_shape_mode == "scalar":
            dg = torch.sum(grad * self.delta_w)

        elif self.r_shape_mode == "channel":
            dg = torch.sum(grad * self.delta_w, dim=1, keepdim=True)

        elif self.r_shape_mode == "group":
            out, in_feat = grad.shape
            g = self.group_size
            n_groups = (in_feat + g - 1) // g

            dg = torch.zeros(out, n_groups, device=grad.device)

            for i in range(n_groups):
                s = i * g
                e = min((i + 1) * g, in_feat)
                dg[:, i] = torch.sum(
                    grad[:, s:e] * self.delta_w[:, s:e], dim=1
                )

        denom = dg.abs()

        return {
            "denom_mean": denom.mean().item(),
            "denom_std": denom.std().item(),
            "denom_max": denom.max().item(),
            "denom_min": denom.min().item(),
            "denom_p50": denom.median().item(),
            "denom_p90": denom.quantile(0.9).item(),
            "denom_p95": denom.quantile(0.95).item(),
            "denom_p99": denom.quantile(0.99).item(),
        }
    
