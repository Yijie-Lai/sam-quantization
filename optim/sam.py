import torch

def SAM(params, base_optimizer, rho=0.05, **kwargs):

    class _SAM(base_optimizer):
        def __init__(self, params, rho=0.05, **kwargs):
            super().__init__(params, **kwargs)
            assert rho >= 0.0
            self.rho = rho

            self._e_ws = {} 

        @torch.no_grad()
        def first_step(self, zero_grad=True):

            self._e_ws = {} 

            for group in self.param_groups:
                group_norm = self._group_grad_norm(group)
                scale = self.rho / (group_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    e_w = p.grad * scale.to(p)

                    p.add_(e_w)

                    self._e_ws[p] = e_w

            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    p.sub_(self._e_ws[p])

            if zero_grad:
                self.zero_grad()

        def _group_grad_norm(self, group):
            norms = []

            device = None
            for p in group["params"]:
                if p.grad is not None:
                    device = p.grad.device
                    break

            if device is None:
                return torch.tensor(0.0)

            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(device))

            if len(norms) == 0:
                return torch.tensor(0.0, device=device)

            return torch.norm(torch.stack(norms), p=2)

    return _SAM(params, rho=rho, **kwargs)