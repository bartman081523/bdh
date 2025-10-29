# bdh_paper.py
# Eine getreue und numerisch stabile PyTorch-Implementierung der BDH-GPU-Architektur.

import torch
import torch.nn as nn

def relu(z: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(z, 0.0)

def layernorm_row(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = z.mean(dim=-1, keepdim=True)
    s = z.std(dim=-1, keepdim=True)
    return (z - m) / (s + eps)

class BDH_GPU(nn.Module):
    def __init__(self, n: int, d: int, V: int, u_decay: float = 0.97, x_decay: float = 0.97):
        super().__init__()
        self.n, self.d, self.V = n, d, V
        self.u_decay, self.x_decay = u_decay, x_decay

        self.token_emb = nn.Parameter(torch.randn(V, d) / (d**0.5))
        self.E = nn.Parameter(torch.randn(d, n) / (n**0.5))
        self.Dx = nn.Parameter(torch.randn(n, d) / (d**0.5))
        self.Dy = nn.Parameter(torch.randn(n, d) / (d**0.5))

        self.register_buffer("x", torch.zeros(1, n))
        self.register_buffer("y", torch.zeros(1, n))
        self.register_buffer("v", torch.zeros(1, d))
        self.register_buffer("rho", torch.zeros(1, d, n))

    def reset_state(self, batch_size: int, device: torch.device):
        self.x = torch.zeros(batch_size, self.n, device=device)
        self.y = torch.zeros(batch_size, self.n, device=device)
        self.v = torch.zeros(batch_size, self.d, device=device)
        self.rho = torch.zeros(batch_size, self.d, self.n, device=device)

    def step(self, token_idx: torch.Tensor) -> torch.Tensor:
        v_prev = self.token_emb[token_idx]
        x_update = relu(v_prev @ self.Dx.T)
        x_t = self.x_decay * self.x + x_update
        x_t = x_t / (x_t.norm(p=1, dim=-1, keepdim=True) + 1e-6)
        a_star = torch.einsum('bdn,bn->bd', self.rho, x_t)
        y_core = layernorm_row(a_star) @ self.Dy.T
        y_t = relu(y_core) * relu(x_t)
        v_star = layernorm_row(y_t @ self.E.T)
        v_prev_normed = layernorm_row(v_prev)
        increment = torch.einsum('bd,bn->bdn', v_prev_normed, x_t)
        self.rho = self.u_decay * (self.rho + increment)
        self.x, self.y, self.v = x_t, y_t, v_star
        return v_star

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.rho = self.rho.detach()
        self.x = self.x.detach()
        self.y = self.y.detach()
        self.v = self.v.detach()
        outputs = []
        for t in range(idx.shape[1]):
            v_out = self.step(idx[:, t])
            outputs.append(v_out.unsqueeze(1))
        return torch.cat(outputs, dim=1)
