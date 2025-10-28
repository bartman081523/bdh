# bdh_paper.py
# Eine getreue und numerisch stabile PyTorch-Implementierung der BDH-GPU-Architektur.
# Fügt interne Normalisierungen hinzu, um die positive Rückkopplungsschleife zu durchbrechen.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def relu(z: torch.Tensor) -> torch.Tensor:
    """Rectified Linear Unit Aktivierungsfunktion."""
    return torch.clamp_min(z, 0.0)

def layernorm_row(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Wendet Layer Normalization auf die letzte Dimension eines Tensors an.
    """
    m = z.mean(dim=-1, keepdim=True)
    s = z.std(dim=-1, keepdim=True)
    return (z - m) / (s + eps)

class BDH_GPU(nn.Module):
    """
    BDH-GPU Zustandsraummodell (Definition 4), mit Stabilisierungen für robustes Training.
    """
    def __init__(self, n: int, d: int, V: int, u_decay: float = 0.97, x_decay: float = 0.97):
        super().__init__()
        self.n = n
        self.d = d
        self.V = V
        self.u_decay = u_decay
        self.x_decay = x_decay

        self.token_emb = nn.Parameter(torch.randn(V, d) / (d**0.5))
        self.E = nn.Parameter(torch.randn(d, n) / (n**0.5))
        self.Dx = nn.Parameter(torch.randn(n, d) / (d**0.5))
        self.Dy = nn.Parameter(torch.randn(n, d) / (d**0.5))

        self.register_buffer("x", torch.zeros(1, n))
        self.register_buffer("y", torch.zeros(1, n))
        self.register_buffer("v", torch.zeros(1, d))
        self.register_buffer("rho", torch.zeros(1, d, n))

    def reset_state(self, batch_size: int, device: torch.device):
        """Setzt die rekurrenten Zustandsvariablen für eine neue Sequenz zurück."""
        self.x = torch.zeros(batch_size, self.n, device=device)
        self.y = torch.zeros(batch_size, self.n, device=device)
        self.v = torch.zeros(batch_size, self.d, device=device)
        self.rho = torch.zeros(batch_size, self.d, self.n, device=device)

    def step(self, token_idx: torch.Tensor) -> torch.Tensor:
        """Führt einen einzelnen, stabilisierten rekurrenten Schritt aus."""
        v_prev = self.token_emb[token_idx]

        x_update = relu(v_prev @ self.Dx.T)
        x_t = self.x_decay * self.x + x_update

        # --- STABILITÄTS-FIX 1: L1-Normalisierung von x_t ---
        # Dies ist der wichtigste Schritt, um die Explosion von x_t zu verhindern.
        # Es bricht die positive Rückkopplungsschleife.
        x_t = x_t / (x_t.norm(p=1, dim=-1, keepdim=True) + 1e-6)

        a_star = torch.einsum('bdn,bn->bd', self.rho, x_t)

        y_core = layernorm_row(a_star) @ self.Dy.T
        y_t = relu(y_core) * relu(x_t)

        v_star = layernorm_row(y_t @ self.E.T)

        # --- STABILITÄTS-FIX 2: Normalisierung von v_prev vor dem rho-Update ---
        # Stellt sicher, dass beide Vektoren im äußeren Produkt eine kontrollierte Magnitude haben.
        v_prev_normed = layernorm_row(v_prev)
        increment = torch.einsum('bd,bn->bdn', v_prev_normed, x_t)
        self.rho = self.u_decay * (self.rho + increment)

        self.x, self.y, self.v = x_t, y_t, v_star

        return v_star

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Verarbeitet eine Sequenz von Tokens und gibt die Ausgabevektoren für jeden Schritt zurück."""
        B, T = idx.shape
        self.rho = self.rho.detach()
        self.x = self.x.detach()
        self.y = self.y.detach()
        self.v = self.v.detach()

        outputs = []
        for t in range(T):
            token_idx = idx[:, t]
            v_out = self.step(token_idx)
            outputs.append(v_out.unsqueeze(1))

        return torch.cat(outputs, dim=1)
