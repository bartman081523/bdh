
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

Array = np.ndarray

def relu(z: Array) -> Array:
    return np.maximum(0.0, z)

def layernorm_row(v: Array, eps: float = 1e-6) -> Array:
    m = v.mean(axis=-1, keepdims=True)
    s = v.std(axis=-1, keepdims=True)
    return (v - m) / (s + eps)

def softmax(z: Array, axis: int = -1) -> Array:
    z = z - np.max(z, axis=axis, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=axis, keepdims=True)

def effective_rank(mat: Array, eps: float = 1e-12) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    ps = (s**2) / (np.sum(s**2) + eps)
    H = -np.sum(ps * (np.log(ps + eps)))
    return float(np.exp(H))

# ----------------------------- Baseline (Definition 4) -----------------------------

@dataclass
class BDHGPURef:
    """
    Minimal, faithful implementation of BDH-GPU (Definition 4) in NumPy.

    States:
        x_t in R^n_{>=0}, y_t in R^n_{>=0}, v*_t in R^d, rho_t in R^{d x n}

    Update (per token):
        x_t = x_{t-1} + (Dx v*_{t-1})_+                          [ReLU-lowrank path]
        a*_t = rho_{t-1} x_t
        y_t = (Dy LN(a*_t))_+ ⊙ x_t                               [LN in d-space]
        v*_t = LN(E y_t)
        rho_t = (rho_{t-1} + v*_{t-1} x_t^T) * U                  [rank-1 increment]

    This mirrors the paper's Definition 4 up to the placement of LN, which can be toggled.
    """
    n: int = 256
    d: int = 32
    V: int = 4096
    seed: int = 0
    u_decay: float = 0.97
    ln_before_Dy: bool = True
    use_relu_lowrank: bool = True

    E: Array = field(init=False)
    Dx: Array = field(init=False)
    Dy: Array = field(init=False)
    token_emb: Array = field(init=False)

    x: Array = field(init=False)
    y: Array = field(init=False)
    v: Array = field(init=False)
    rho: Array = field(init=False)

    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.E = self.rng.standard_normal((self.d, self.n)) * (1.0/np.sqrt(self.n))
        self.Dx = self.rng.standard_normal((self.n, self.d)) * (1.0/np.sqrt(self.d))
        self.Dy = self.rng.standard_normal((self.n, self.d)) * (1.0/np.sqrt(self.d))
        self.token_emb = self.rng.standard_normal((self.V, self.d)) * (1.0/np.sqrt(self.d))

        self.x = np.zeros((self.n,), dtype=float)
        self.y = np.zeros((self.n,), dtype=float)
        self.v = np.zeros((self.d,), dtype=float)
        self.rho = np.zeros((self.d, self.n), dtype=float)

    def _LN_d(self, z: Array) -> Array:
        return layernorm_row(z.reshape(1, -1)).ravel()

    def step(self, token_index: int) -> Dict[str, float]:
        v_prev = self.token_emb[token_index]
        x_in = self.Dx @ v_prev
        # >>> Uses ReLU-lowrank as in the paper.
        x_t = self.x + (relu(x_in) if self.use_relu_lowrank else x_in)

        a_star = self.rho @ x_t

        if self.ln_before_Dy:
            y_core = self.Dy @ self._LN_d(a_star)
        else:
            y_core = self._LN_d(self.Dy @ a_star)
        y_t = relu(y_core) * np.maximum(0.0, x_t)

        v_star = self._LN_d(self.E @ y_t)

        # >>> Rank-1 outer product increment on rho, then scalar decay U (baseline).
        self.rho = self.u_decay * (self.rho + v_prev.reshape(self.d,1) @ x_t.reshape(1,self.n))

        self.x, self.y, self.v = x_t, y_t, v_star

        spars_x = 1.0 - (np.count_nonzero(self.x) / self.n)
        spars_y = 1.0 - (np.count_nonzero(self.y) / self.n)
        return dict(sparsity_x=spars_x, sparsity_y=spars_y,
                    rho_F=float(np.linalg.norm(self.rho, ord='fro')),
                    rho_eff_rank=effective_rank(self.rho))

    def run(self, T: int) -> Dict[str, float]:
        out = {}
        for _ in range(T):
            idx = int(self.rng.integers(0, self.V))
            out = self.step(idx)
        return out

# ----------------------------- Neuro‑inspired extension -----------------------------

@dataclass
class BDHNeuroRef(BDHGPURef):
    """
    Drop-in replacement that adds neuro-inspired mechanisms.
    Each block below is annotated with:
        >>> DIFFERENCE vs BDH (Def.4): <what changed>
    """
    # (1) Multi-timescale STDP-like kernels for U (mixture of decays)
    U_kernels: Optional[List[float]] = None
    U_weights: Optional[List[float]] = None

    # (2) Local, activity-dependent forgetting in rho
    local_forget_eta: float = 0.0

    # (3) Homeostatic target activity for y
    homeostasis_tau: Optional[float] = None
    homeostasis_beta: float = 0.05

    # (4) k-WTA / lateral inhibition in n-space
    k_wta: Optional[int] = None

    # (5) Dendritic subunits (branch nonlinearity across row-splits of Dy)
    branches: int = 0
    branch_nl: str = "softplus"  # "softplus" or "relu"

    # (6) Neuromodulation via surprisal (entropy of v*)
    mod_gamma_max: float = 1.0

    # (7) Stochastic spiking (Bernoulli noise before ReLU)
    spike_rate: float = 0.0

    def _branch_nonlin(self, z: Array) -> Array:
        if self.branch_nl == "softplus":
            return np.log1p(np.exp(z))
        return np.maximum(0.0, z)

    def _surprisal_gain(self, v_star: Array) -> float:
        if self.mod_gamma_max <= 0.0:
            return 1.0
        p = softmax(v_star.reshape(1,-1), axis=-1).ravel()
        H = -np.sum(p * np.log(p + 1e-12))
        return float(min(self.mod_gamma_max, H / np.log(self.d + 1e-6) * self.mod_gamma_max))

    def step(self, token_index: int) -> Dict[str, float]:
        v_prev = self.token_emb[token_index]

        # Same x-path as baseline (optional ReLU-lowrank).
        x_in = self.Dx @ v_prev
        x_t = self.x + (relu(x_in) if self.use_relu_lowrank else x_in)

        # Linear attention readout.
        a_star = self.rho @ x_t

        # >>> DIFFERENCE vs BDH (Def.4): dendritic subunits across Dy rows.
        if self.branches > 0:
            a_hat = layernorm_row(a_star.reshape(1,-1)).ravel() if self.ln_before_Dy else a_star
            splits = np.array_split(self.Dy, self.branches, axis=0)
            parts = [self._branch_nonlin(Dy_b @ a_hat) for Dy_b in splits]
            y_core = np.concatenate(parts, axis=0)  # back to length n
            if not self.ln_before_Dy:
                y_core = layernorm_row(y_core.reshape(1,-1)).ravel()
        else:
            if self.ln_before_Dy:
                y_core = self.Dy @ layernorm_row(a_star.reshape(1,-1)).ravel()
            else:
                y_core = layernorm_row((self.Dy @ a_star).reshape(1,-1)).ravel()

        # >>> DIFFERENCE: stochastic spikes before thresholding.
        if self.spike_rate > 0.0:
            spikes = (np.random.random(size=y_core.shape) < self.spike_rate).astype(y_core.dtype)
            y_core = y_core + spikes

        # >>> DIFFERENCE: k-WTA lateral inhibition (n-space).
        if self.k_wta is not None and 0 < self.k_wta < self.n:
            idx = np.argpartition(y_core, -self.k_wta)[-self.k_wta:]
            mask = np.zeros_like(y_core, dtype=bool)
            mask[idx] = True
            y_core = y_core * mask

        # Nonnegativity & gating by x≥0 (as in baseline).
        y_t = relu(y_core) * np.maximum(0.0, x_t)

        # >>> DIFFERENCE: homeostatic scaling towards target L1(y).
        if self.homeostasis_tau is not None:
            s = float(np.sum(y_t))
            if s > 1e-8:
                scale = min(1.0, self.homeostasis_tau / (s + 1e-8))
                y_t = y_t * scale

        v_star = layernorm_row((self.E @ y_t).reshape(1,-1)).ravel()

        # >>> DIFFERENCE: local forgetting for inactive presynapses on rho.
        rho_next = self.rho.copy()
        if self.local_forget_eta > 0.0:
            inactive = (x_t <= 0.0).astype(rho_next.dtype)
            rho_next = rho_next * (1.0 - self.local_forget_eta * inactive.reshape(1, self.n))

        # >>> DIFFERENCE: neuromodulated, multi-timescale rho update.
        gain = self._surprisal_gain(v_star)  # ∈ (0, mod_gamma_max]
        inc = gain * (v_prev.reshape(self.d,1) @ x_t.reshape(1,self.n))  # rank-1

        if self.U_kernels is None:
            rho_next = self.u_decay * (rho_next + inc)  # baseline scalar U
        else:
            wk = np.array(self.U_weights if self.U_weights is not None else [1.0]*len(self.U_kernels), dtype=float)
            wk = wk / np.sum(wk)
            rho_mix = np.zeros_like(rho_next)
            for w, u in zip(wk, self.U_kernels):
                rho_mix += w * (u * (rho_next + inc))
            rho_next = rho_mix

        self.rho = rho_next
        self.x, self.y, self.v = x_t, y_t, v_star

        spars_x = 1.0 - (np.count_nonzero(self.x) / self.n)
        spars_y = 1.0 - (np.count_nonzero(self.y) / self.n)
        return dict(sparsity_x=spars_x, sparsity_y=spars_y,
                    rho_F=float(np.linalg.norm(self.rho, ord='fro')),
                    rho_eff_rank=effective_rank(self.rho), gain=gain)

if __name__ == "__main__":
    # Tiny smoke test
    base = BDHGPURef(n=64, d=16, V=512, seed=3)
    print("baseline:", base.run(T=8))
    neuro = BDHNeuroRef(n=64, d=16, V=512, seed=3,
                        U_kernels=[0.99,0.97,0.94], U_weights=[0.5,0.3,0.2],
                        local_forget_eta=0.02, homeostasis_tau=0.15*64,
                        k_wta=8, branches=2, branch_nl="softplus",
                        mod_gamma_max=0.8, spike_rate=0.01)
    print("neuro:", neuro.run(T=8))
