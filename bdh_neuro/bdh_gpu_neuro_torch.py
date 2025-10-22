#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDH-Neuro Reference Implementation (PyTorch)
--------------------------------------------
Neuro-inspired variant of BDH-GPU (Def.4) with biological dynamics.

✔ Multi-timescale synaptic decay (STDP-like)
✔ Local forgetting for inactive pre-synapses
✔ Homeostatic regulation of output activity
✔ k-WTA lateral inhibition on y (and adaptive k-WTA on x)
✔ Dendritic subunits with branch nonlinearities
✔ Neuromodulated rho updates via surprisal gain
✔ Stochastic spikes
✔ x-decay + normalization for realistic sparsity
✔ NEW: adaptive x-sparsity (gain-modulated Option C)
"""

import torch
from typing import Optional, List, Dict


def relu(z):
    return torch.clamp_min(z, 0.0)


def layernorm_row(z, eps: float = 1e-6):
    m = z.mean(dim=-1, keepdim=True)
    s = z.std(dim=-1, keepdim=True)
    return (z - m) / (s + eps)


def effective_rank(mat: torch.Tensor, eps: float = 1e-12) -> float:
    with torch.no_grad():
        s = torch.linalg.svdvals(mat)
        ps = (s**2) / (s.pow(2).sum() + eps)
        H = -(ps * (ps.add(eps).log())).sum()
        return float(torch.exp(H))


class BDHGPURefTorch(torch.nn.Module):
    """Baseline BDH-GPU (Definition 4)."""
    def __init__(self, n=256, d=32, V=4096, seed=0, u_decay=0.97,
                 ln_before_Dy=True, use_relu_lowrank=True, device="cpu"):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(seed)
        self.n, self.d, self.V = n, d, V
        self.u_decay = float(u_decay)
        self.ln_before_Dy = bool(ln_before_Dy)
        self.use_relu_lowrank = bool(use_relu_lowrank)
        self.device = device

        self.E  = torch.randn(d, n, generator=g, device=device) / (n**0.5)
        self.Dx = torch.randn(n, d, generator=g, device=device) / (d**0.5)
        self.Dy = torch.randn(n, d, generator=g, device=device) / (d**0.5)
        self.token_emb = torch.randn(V, d, generator=g, device=device) / (d**0.5)

        self.register_buffer("x", torch.zeros(n, device=device))
        self.register_buffer("y", torch.zeros(n, device=device))
        self.register_buffer("v", torch.zeros(d, device=device))
        self.register_buffer("rho", torch.zeros(d, n, device=device))
        self._rng = g

    def step(self, token_index: int) -> Dict[str, float]:
        v_prev = self.token_emb[int(token_index)]
        x_t = self.x + (relu(self.Dx @ v_prev) if self.use_relu_lowrank else (self.Dx @ v_prev))
        a_star = self.rho @ x_t
        if self.ln_before_Dy:
            y_core = self.Dy @ layernorm_row(a_star)
        else:
            y_core = layernorm_row(self.Dy @ a_star)
        y_t = relu(y_core) * torch.clamp_min(x_t, 0.0)
        v_star = layernorm_row(self.E @ y_t)
        self.rho = self.u_decay * (self.rho + v_prev.view(self.d,1) @ x_t.view(1,self.n))
        self.x, self.y, self.v = x_t, y_t, v_star

        with torch.no_grad():
            spars_x = 1.0 - float((self.x != 0).float().mean().item())
            spars_y = 1.0 - float((self.y != 0).float().mean().item())
            rho_F = float(torch.linalg.norm(self.rho).item())
            rho_er = effective_rank(self.rho)
        return dict(sparsity_x=spars_x, sparsity_y=spars_y,
                    rho_F=rho_F, rho_eff_rank=rho_er)

    @torch.no_grad()
    def run(self, T: int) -> Dict[str, float]:
        out = {}
        for _ in range(T):
            idx = torch.randint(0, self.V, (1,), generator=self._rng, device=self.device).item()
            out = self.step(idx)
        return out


class BDHNeuroRefTorch(BDHGPURefTorch):
    """Neuro-inspired BDH variant with biologically plausible extensions."""
    def __init__(self, *args,
                 U_kernels: Optional[List[float]] = None, U_weights: Optional[List[float]] = None,
                 local_forget_eta: float = 0.0, homeostasis_tau: Optional[float] = None,
                 k_wta: Optional[int] = None, branches: int = 0, branch_nl: str = "softplus",
                 mod_gamma_max: float = 1.0, spike_rate: float = 0.0,
                 x_decay: float = 0.97,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.U_kernels = U_kernels
        self.U_weights = U_weights
        self.local_forget_eta = float(local_forget_eta)
        self.homeostasis_tau = homeostasis_tau
        self.k_wta = k_wta
        self.branches = int(branches)
        self.branch_nl = branch_nl
        self.mod_gamma_max = float(mod_gamma_max)
        self.spike_rate = float(spike_rate)
        self.x_decay = x_decay

    def _branch_nl(self, z):
        if self.branch_nl == "softplus":
            return torch.nn.functional.softplus(z)
        return relu(z)

    def _surprisal_gain(self, v_star: torch.Tensor) -> float:
        if self.mod_gamma_max <= 0.0:
            return 1.0
        p = torch.softmax(v_star, dim=-1)
        H = -(p * (p + 1e-12).log()).sum()
        return float(min(self.mod_gamma_max,
                         (H / torch.log(torch.tensor(float(self.d) + 1e-6))).item()
                         * self.mod_gamma_max))

    def step(self, token_index: int) -> Dict[str, float]:
        v_prev = self.token_emb[int(token_index)]

        # --- x dynamics: decay + normalization
        x_t = self.x_decay * self.x + (relu(self.Dx @ v_prev)
                                       if self.use_relu_lowrank
                                       else (self.Dx @ v_prev))
        x_t = x_t / (x_t.norm(p=1) + 1e-6)

        # temporary gain placeholder (computed later)
        gain = 0.7

        # --- core drive before computing gain
        a_star = self.rho @ x_t

        # Dendritic subunits
        if self.branches and self.branches > 0:
            a_hat = layernorm_row(a_star) if self.ln_before_Dy else a_star
            splits = torch.tensor_split(self.Dy, self.branches, dim=0)
            parts = [self._branch_nl(Dy_b @ a_hat) for Dy_b in splits]
            y_core = torch.cat(parts, dim=0)
            if not self.ln_before_Dy:
                y_core = layernorm_row(y_core)
        else:
            if self.ln_before_Dy:
                y_core = self.Dy @ layernorm_row(a_star)
            else:
                y_core = layernorm_row(self.Dy @ a_star)

        # Spikes
        if self.spike_rate > 0.0:
            u = torch.rand_like(y_core)
            y_core = y_core + (u < self.spike_rate).float()

        # Lateral inhibition on y
        if self.k_wta is not None and 0 < self.k_wta < self.n:
            vals, idx = torch.topk(y_core, self.k_wta)
            mask = torch.zeros_like(y_core, dtype=torch.bool)
            mask[idx] = True
            y_core = y_core * mask.float()

        y_t = relu(y_core) * torch.clamp_min(x_t, 0.0)

        # Homeostasis
        if self.homeostasis_tau is not None:
            s = y_t.sum().item()
            if s > 1e-8:
                scale = min(1.0, self.homeostasis_tau / (s + 1e-8))
                y_t = y_t * scale

        v_star = layernorm_row(self.E @ y_t)

        # --- recompute gain now that v_star is known
        gain = self._surprisal_gain(v_star)

        # --- adaptive k-WTA on x (Option C: gain-modulated)
        k_dynamic = int(self.n * (0.05 + 0.25 * gain))
        k_dynamic = max(1, min(k_dynamic, self.n))
        vals, idx = torch.topk(x_t, k_dynamic)
        mask = torch.zeros_like(x_t, dtype=torch.bool)
        mask[idx] = True
        x_t = x_t * mask.float()

        # Local forgetting
        rho_next = self.rho
        if self.local_forget_eta > 0.0:
            inactive = (x_t <= 0).float().view(1, self.n)
            rho_next = rho_next * (1.0 - self.local_forget_eta * inactive)

        # Neuromodulated multi-timescale update
        inc = gain * (v_prev.view(self.d,1) @ x_t.view(1,self.n))
        if self.U_kernels is None:
            rho_next = self.u_decay * (rho_next + inc)
        else:
            wk = torch.tensor(self.U_weights if self.U_weights is not None else
                              [1.0]*len(self.U_kernels),
                              dtype=torch.float32, device=rho_next.device)
            wk = wk / wk.sum()
            rho_mix = torch.zeros_like(rho_next)
            for w, u in zip(wk, self.U_kernels):
                rho_mix += w * (u * (rho_next + inc))
            rho_next = rho_mix

        self.rho = rho_next
        self.x, self.y, self.v = x_t, y_t, v_star

        with torch.no_grad():
            spars_x = 1.0 - float((self.x != 0).float().mean().item())
            spars_y = 1.0 - float((self.y != 0).float().mean().item())
            rho_F = float(torch.linalg.norm(self.rho).item())
            rho_er = effective_rank(self.rho)
        return dict(sparsity_x=spars_x, sparsity_y=spars_y,
                    rho_F=rho_F, rho_eff_rank=rho_er, gain=gain)
