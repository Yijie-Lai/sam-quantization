import torch


def ASAM(params, base_optimizer, rho=0.05, adaptive=False, **kwargs):

    class _SAM(base_optimizer):

        def __init__(self, params, rho=0.05, adaptive=False, **kwargs):
            super().__init__(params, **kwargs)

            assert rho >= 0.0

            self.rho = rho
            self.adaptive = adaptive

            self._old_params = {}

        @torch.no_grad()
        def first_step(self, zero_grad=True):

            self._old_params = {}

            for group in self.param_groups:

                group_norm = self._group_grad_norm(group)

                if not torch.isfinite(group_norm):
                    continue

                scale = self.rho / (group_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    self._old_params[p] = p.data.clone()

                    grad = p.grad

                    # ASAM
                    if self.adaptive:
                        e_w = torch.pow(p, 2) * grad * scale.to(p)
                    else:
                        e_w = grad * scale.to(p)
                    if not torch.isfinite(e_w).all():
                        continue

                    p.add_(e_w)

            if zero_grad:
                self.zero_grad(set_to_none=True)

        @torch.no_grad()
        def second_step(self, zero_grad=False):

            for group in self.param_groups:

                for p in group["params"]:
                    if p in self._old_params:
                        p.data.copy_(self._old_params[p])

            if zero_grad:
                self.zero_grad(set_to_none=True)

        def _group_grad_norm(self, group):

            device = None

            for p in group["params"]:
                if p.grad is not None:
                    device = p.grad.device
                    break

            if device is None:
                return torch.tensor(0.0)

            norms = []

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                # ASAM norm
                if self.adaptive:
                    grad = torch.abs(p) * grad

                norms.append(grad.norm(p=2).to(device))

            if len(norms) == 0:
                return torch.tensor(0.0, device=device)

            norm = torch.norm(torch.stack(norms), p=2)

            if not torch.isfinite(norm):
                return torch.tensor(0.0, device=device)

            return norm

    return _SAM(
        params,
        rho=rho,
        adaptive=adaptive,
        **kwargs,
    )