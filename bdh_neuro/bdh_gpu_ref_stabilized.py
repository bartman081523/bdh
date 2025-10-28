#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bdh_gpu_ref_stabilized.py — Stable BDH-GPU Reference Model
-----------------------------------------------------------
A minimal, numerically stable BDH-GPU baseline that adds:

✔ x-state decay (leaky integration)
✔ L1 normalization of x
✔ Soft thresholding for sparsity

No neuromodulation, no branches, no homeostasis — pure BDH core.
Intended for direct comparison to BDH-Neuro variant.
"""

import torch
from typing import Dict


def relu(z):
    return torch.clamp_min(z, 0.0)


def layernorm_row(z, eps: float = 1e-6):
    """Row-wise layer normalization."""
    m = z.mean(dim=-1, keepdim=True)
    s = z.std(dim=-1, keepdim=True)
    return (z - m) / (s + eps)


def effective_rank(mat: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank based on singular-value entropy."""
    with torch.no_grad():
        s = torch.linalg.svdvals(mat)
        ps = (s**2) / (s.pow(2).sum() + eps)
        H = -(ps * (ps.add(eps).log())).sum()
        return float(torch.exp(H))


class BDHGPURefStabilized(torch.nn.Module):
    """Minimal BDH-GPU (Definition 4) with stabilizing sparsity mechanics."""

    def __init__(self, n=256, d=32, V=4096, seed=0,
                 u_decay=0.97, ln_before_Dy=True, device="cpu",
                 x_decay: float = 0.97, threshold_ratio: float = 0.02):
        """
        Args:
            n, d, V: model dimensions
            seed: RNG seed
            u_decay: forgetting factor for rho
            ln_before_Dy: layernorm position flag
            device: cpu or cuda
            x_decay: leak factor for x state
            threshold_ratio: threshold relative to x_t.max() for sparsity
        """
        super().__init__()
        g = torch.Generator(device=device).manual_seed(seed)
        self.n, self.d, self.V = n, d, V
        self.device = device
        self.u_decay = float(u_decay)
        self.ln_before_Dy = bool(ln_before_Dy)
        self.x_decay = float(x_decay)
        self.threshold_ratio = float(threshold_ratio)

        # Weight matrices (non-trainable)
        self.E  = torch.randn(d, n, generator=g, device=device) / (n**0.5)
        self.Dx = torch.randn(n, d, generator=g, device=device) / (d**0.5)
        self.Dy = torch.randn(n, d, generator=g, device=device) / (d**0.5)
        self.token_emb = torch.randn(V, d, generator=g, device=device) / (d**0.5)

        # State variables
        self.register_buffer("x", torch.zeros(n, device=device))
        self.register_buffer("y", torch.zeros(n, device=device))
        self.register_buffer("v", torch.zeros(d, device=device))
        self.register_buffer("rho", torch.zeros(d, n, device=device))
        self._rng = g

    # ------------------------------------------------------------
    # Core BDH update
    # ------------------------------------------------------------
    def step(self, token_index: int) -> Dict[str, float]:
        v_prev = self.token_emb[int(token_index)]

        # --- Stabilized x dynamics: decay, normalize, threshold ---
        x_t = self.x_decay * self.x + (self.Dx @ v_prev)
        x_t = x_t / (x_t.norm(p=1) + 1e-6)  # energy normalization

        # soft thresholding for sparsity
        threshold = self.threshold_ratio * x_t.max()
        x_t = torch.where(x_t > threshold, x_t, torch.zeros_like(x_t))

        # --- y dynamics (standard BDH flow) ---
        a_star = self.rho @ x_t
        if self.ln_before_Dy:
            y_core = self.Dy @ layernorm_row(a_star)
        else:
            y_core = layernorm_row(self.Dy @ a_star)

        y_t = relu(y_core) * torch.clamp_min(x_t, 0.0)
        v_star = layernorm_row(self.E @ y_t)

        # --- Hebbian update for rho ---
        self.rho = self.u_decay * (self.rho + v_prev.view(self.d, 1) @ x_t.view(1, self.n))

        # --- commit new state ---
        self.x, self.y, self.v = x_t, y_t, v_star

        # --- diagnostics ---
        with torch.no_grad():
            spars_x = 1.0 - float((self.x != 0).float().mean().item())
            spars_y = 1.0 - float((self.y != 0).float().mean().item())
            rho_F = float(torch.linalg.norm(self.rho).item())
            rho_er = effective_rank(self.rho)

        return dict(sparsity_x=spars_x, sparsity_y=spars_y,
                    rho_F=rho_F, rho_eff_rank=rho_er)

    # ------------------------------------------------------------
    # Run multiple tokens sequentially
    # ------------------------------------------------------------
    @torch.no_grad()
    def run(self, T: int) -> Dict[str, float]:
        out = {}
        for _ in range(T):
            idx = torch.randint(0, self.V, (1,), generator=self._rng, device=self.device).item()
            out = self.step(idx)
        return out
