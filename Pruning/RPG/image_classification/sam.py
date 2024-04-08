# 2023.11-Modified some parts in the code
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import math


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, v2=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.v2 = v2

    @torch.no_grad()
    def first_step(self, args, epoch, iter, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            if args.decay_rho:
                # rho = math.sqrt(group["rho"] / (iter + 1)) # decay by math derivation
                if epoch % args.decay_freq == 0 and epoch != 0 and iter == 0:
                    group["rho"] = group["rho"] / args.decay_factor
                if epoch >= args.decay_stop and iter == 0:
                    group["rho"] = 0

            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if self.v2 and "prev_grad" in self.state[p]:
                    grad = self.state[p]["prev_grad"]
                    prev_p = self.state[p]["prev_u"]
                else:
                    grad = p.grad
                    prev_p = p
                if p.grad is None: continue

                if epoch >= args.sam_start:
                    e_w = (torch.pow(prev_p, 2) if group["adaptive"] else 1.0) * grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"
                    self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if self.v2:
                    self.state[p]["prev_grad"] = torch.clone(p.grad)
                    self.state[p]["prev_u"] = torch.clone(p)
                if p.grad is None: continue
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        # scaler.step(self.base_optimizer)
        # scaler.update()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism

        grad_list = [
            ((torch.abs((self.state[p]["prev_u"] if (self.v2 and "prev_u" in self.state[p]) else p)) if group["adaptive"] else 1.0) *
             (self.state[p]["prev_grad"] if (self.v2 and "prev_grad" in self.state[p]) else p.grad)).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]

        if len(grad_list) == 0:
            return 0
        stack = torch.stack(grad_list)
        norm = torch.norm(stack, p=2)

        return norm