import torch


def ASAM(params, base_optimizer, rho=0.05, adaptive=False, auto_rho=False, auto_rho_eps=1e-12, **kwargs):

    class _SAM(base_optimizer):

        def __init__(self, params, rho=0.01, adaptive=False, auto_rho=False, auto_rho_eps=1e-12, **kwargs):
            super().__init__(params, **kwargs)

            assert rho >= 0.0

            self.rho = rho
            self.adaptive = adaptive
            self.auto_rho = auto_rho
            self.auto_rho_eps = auto_rho_eps
            self._old_params = {}

            for group in self.param_groups:
                group.setdefault("rho", rho)

        @torch.no_grad()
        def first_step(self, zero_grad=True):
            self._old_params = {}

            for group in self.param_groups:
                rho = group.get("rho", self.rho)

                group_norm = self._group_grad_norm(group)
                scale = rho / (group_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    self._old_params[p] = p.data.clone()

                    grad = p.grad

                    if self.adaptive:
                        e_w = torch.pow(p, 2) * grad * scale
                    else:
                        e_w = grad * scale

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

                if self.adaptive:
                    grad = torch.abs(p) * grad

                norms.append(grad.norm(p=2).to(device))

            if len(norms) == 0:
                return torch.tensor(0.0, device=device)

            norm = torch.norm(torch.stack(norms), p=2)

            if not torch.isfinite(norm):
                return torch.tensor(0.0, device=device)

            return norm

        def compute_auto_rho(self, loss, trainable_params):
            params = [
                p for p in trainable_params
                if p.requires_grad
            ]

            if len(params) == 0:
                return

            grads = torch.autograd.grad(
                loss.float(),
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            valid = [
                (p, g)
                for p, g in zip(params, grads)
                if g is not None and torch.isfinite(g).all()
            ]

            if len(valid) == 0:
                return

            params, grads = zip(*valid)

            g_detach = [g.detach() for g in grads]

            # u = H g
            dot_g = sum((g * gd).sum() for g, gd in zip(grads, g_detach))

            u = torch.autograd.grad(
                dot_g,
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            u_detach = [
                x.detach() if x is not None else None
                for x in u
            ]

            # v = H u = H^2 g
            dot_u = sum(
                (g * ud).sum()
                for g, ud in zip(grads, u_detach)
                if ud is not None
            )

            v = torch.autograd.grad(
                dot_u,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            stat = {}

            for p, gd, ud, vd in zip(params, g_detach, u_detach, v):
                if ud is None or vd is None:
                    continue
                if not torch.isfinite(ud).all() or not torch.isfinite(vd).all():
                    continue

                stat[p] = (
                    (gd * ud).sum(),           # gHg
                    (ud * vd.detach()).sum(),  # gH3g
                )

            for group in self.param_groups:
                device = None

                for p in group["params"]:
                    if p in stat:
                        device = stat[p][0].device
                        break

                if device is None:
                    group["rho"] = 0.0
                    continue

                gHg = torch.tensor(0.0, device=device)
                gH3g = torch.tensor(0.0, device=device)

                for p in group["params"]:
                    if p not in stat:
                        continue

                    a, c = stat[p]
                    gHg += a
                    gH3g += c

                rho = gHg / (gH3g + self.auto_rho_eps)
                rho = torch.clamp(rho, min=0.0)

                group["rho"] = float(rho.detach().cpu()) if torch.isfinite(rho) else 0.0

                # print(
                #     f"gHg={gHg.item():.4e} "
                #     f"gH3g={gH3g.item():.4e} "
                #     f"rho={float(rho.detach().cpu()):.4e}"
                # )

    return _SAM(
        params,
        rho=rho,
        adaptive=adaptive,
        auto_rho=auto_rho,
        auto_rho_eps=auto_rho_eps,
        **kwargs,
    )