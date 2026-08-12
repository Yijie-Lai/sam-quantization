import torch


def LightSAMAdam(
    params,
    base_optimizer,
    rho=0.05,
    eps=1e-8,
    **kwargs,
):

    class _LightSAM(base_optimizer):

        def __init__(self, params, rho=0.05, eps=1e-8, **kwargs):
            super().__init__(params, **kwargs)

            assert rho >= 0.0

            self.rho = rho
            self.eps = eps

            self._old_params = {}
            self._sam_state = {}

        @torch.no_grad()
        def first_step(self, zero_grad=True):

            self._old_params = {}

            for group in self.param_groups:

                beta1, beta2 = group.get("betas", (0.9, 0.999))

                device = None
                norms = []

                # ===== pass 1: update r/u and compute norm =====
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    if device is None:
                        device = grad.device

                    if p not in self._sam_state:
                        self._sam_state[p] = {
                            "exp_avg": torch.zeros_like(p),
                            "exp_avg_sq": torch.zeros_like(p),
                        }

                    sam_state = self._sam_state[p]

                    r = sam_state["exp_avg"]
                    u = sam_state["exp_avg_sq"]

                    r.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    u.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    d = r / (u.sqrt() + self.eps)

                    norms.append(d.norm(p=2).to(device))

                if device is None or len(norms) == 0:
                    continue

                group_norm = torch.norm(torch.stack(norms), p=2)

                if not torch.isfinite(group_norm) or group_norm <= 0:
                    continue

                scale = self.rho / (group_norm + 1e-12)

                # ===== pass 2: apply normalized LightSAM perturbation =====
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    if p not in self._sam_state:
                        continue

                    sam_state = self._sam_state[p]

                    r = sam_state["exp_avg"]
                    u = sam_state["exp_avg_sq"]

                    d = r / (u.sqrt() + self.eps)
                    e_w = d * scale.to(p)

                    self._old_params[p] = p.data.clone()
                    p.add_(e_w)

            if zero_grad:
                self.zero_grad(set_to_none=True)

        @torch.no_grad()
        def second_step(self, zero_grad=False):

            for group in self.param_groups:

                for p in group["params"]:

                    if p.grad is None:
                        continue

                    if p in self._old_params:
                        p.data.copy_(self._old_params[p])

            if zero_grad:
                self.zero_grad(set_to_none=True)

    return _LightSAM(
        params,
        rho=rho,
        eps=eps,
        **kwargs,
    )