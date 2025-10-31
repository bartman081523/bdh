# bdh_paper.py (VERSION 15 - Final & Lauffähig)
# Enthält die korrigierte, stabile Implementierung von RoPE und der BDH-Kernlogik.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Globale Variable, die durch den Parser in train.py gesteuert wird
DIM_DEBUG = False

def rotate_half(x):
    """ Rotiert die Hälften des Tensors. [x1, x2] -> [-x2, x1] """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """ Wendet die Rotation an, mit korrektem Broadcasting. """
    # cos/sin haben die Form (seq_len, dim) oder (1, dim)
    # x hat die Form (batch, seq_len, dim)
    # Unsqueeze fügt eine Batch-Dimension für Broadcasting hinzu
    return (x * cos.unsqueeze(0)) + (rotate_half(x) * sin.unsqueeze(0))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension muss gerade sein, ist aber {dim}")

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute the sin and cos waves
        self.max_seq_len = max_seq_len
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        if seq_len + offset > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len + offset} exceeds max_seq_len {self.max_seq_len}")

        # Hole den relevanten Teil aus dem Cache
        cos = self.cos_cached[offset : offset + seq_len, :]
        sin = self.sin_cached[offset : offset + seq_len, :]

        if DIM_DEBUG:
            print("\n--- Inside RotaryEmbedding ---")
            print(f"  Input x shape: {x.shape}")
            print(f"  cos/sin shape: {cos.shape}")
            print("--- Exiting RotaryEmbedding ---")

        return apply_rotary_pos_emb(x, cos, sin)

class BDH_GPU(nn.Module):
    def __init__(self, n: int, d: int, V: int, u_decay: float = 0.97, x_decay: float = 0.97):
        super().__init__()
        self.n, self.d, self.V = n, d, V
        self.u_decay, self.x_decay = u_decay, x_decay

        self.token_emb = nn.Embedding(V, d)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=1.0 / (d**0.5))

        self.E = nn.Parameter(torch.randn(d, n) / (n**0.5))
        self.Dx = nn.Parameter(torch.randn(n, d) / (d**0.5))
        self.Dy = nn.Parameter(torch.randn(n, d) / (d**0.5))

        self.register_buffer("x_state", torch.zeros(1, n))
        self.register_buffer("rho_state", torch.zeros(1, d, n))

    def reset_state(self, batch_size: int, device: torch.device):
        self.x_state = torch.zeros(batch_size, self.n, device=device)
        self.rho_state = torch.zeros(batch_size, self.d, self.n, device=device)

    def step(self, rotated_v_prev: torch.Tensor, x_state_in, rho_state_in):
        x_update = F.relu(rotated_v_prev @ self.Dx.T)
        x_t = self.x_decay * x_state_in + x_update
        x_t = F.normalize(x_t, p=1, dim=-1)

        a_star = torch.einsum('bdn,bn->bd', rho_state_in, x_t)
        y_core = F.layer_norm(a_star, [self.d]) @ self.Dy.T
        y_t = F.relu(y_core) * F.relu(x_t)

        v_star = F.layer_norm(y_t @ self.E.T, [self.d])

        v_prev_normed = F.layer_norm(rotated_v_prev, [self.d])
        increment = torch.einsum('bd,bn->bdn', v_prev_normed, x_t)
        rho_t = self.u_decay * rho_state_in + increment

        return v_star, x_t, rho_t

    def forward(self, embeddings: torch.Tensor, x_state_in=None, rho_state_in=None) -> torch.Tensor:
        B, T, D = embeddings.shape
        x_state = self.x_state if x_state_in is None else x_state_in
        rho_state = self.rho_state if rho_state_in is None else rho_state_in

        x_state, rho_state = x_state.detach(), rho_state.detach()

        outputs = []
        for t in range(T):
            v_out, x_state, rho_state = self.step(embeddings[:, t, :], x_state, rho_state)
            outputs.append(v_out.unsqueeze(1))

        self.x_state = x_state
        self.rho_state = rho_state

        return torch.cat(outputs, dim=1)
