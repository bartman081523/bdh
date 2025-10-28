# bdh_paper.py
# A faithful PyTorch implementation of the BDH-GPU architecture from the paper:
# "THE DRAGON HATCHLING: THE MISSING LINK BETWEEN THE TRANSFORMER AND MODELS OF THE BRAIN"
# (Kosowski et al., 2025), based on Definition 4 and Equation (8).

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def relu(z: torch.Tensor) -> torch.Tensor:
    """Rectified Linear Unit activation function."""
    return torch.clamp_min(z, 0.0)

def layernorm_row(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Applies Layer Normalization to the last dimension of a tensor.
    This is a non-parametric version, calculating mean and std on the fly.
    """
    m = z.mean(dim=-1, keepdim=True)
    s = z.std(dim=-1, keepdim=True)
    return (z - m) / (s + eps)

class BDH_GPU(nn.Module):
    """
    BDH-GPU state-space model (Definition 4).

    This module implements the core recurrent dynamics of the Dragon Hatchling.
    It processes sequences token-by-token, updating a persistent state matrix 'rho'
    which represents the model's working memory.

    Args:
        n (int): The dimension of the neuronal space (number of "neurons").
        d (int): The dimension of the latent, low-rank space (embedding size).
        V (int): The vocabulary size.
        u_decay (float): The forgetting factor for the state matrix rho (U in the paper).
        x_decay (float): The decay factor for the x-state (leaky integration).
    """
    def __init__(self, n: int, d: int, V: int, u_decay: float = 0.97, x_decay: float = 0.97):
        super().__init__()
        self.n = n
        self.d = d
        self.V = V
        self.u_decay = u_decay
        self.x_decay = x_decay

        # --- Trainable Parameters ---
        # Token embeddings (maps token ID to a latent vector)
        self.token_emb = nn.Parameter(torch.randn(V, d) / (d**0.5))

        # Encoder matrix E: R^n -> R^d
        self.E = nn.Parameter(torch.randn(d, n) / (n**0.5))

        # Decoder matrix Dx: R^d -> R^n
        self.Dx = nn.Parameter(torch.randn(n, d) / (d**0.5))

        # Decoder matrix Dy: R^d -> R^n
        self.Dy = nn.Parameter(torch.randn(n, d) / (d**0.5))

        # --- State Variables (non-trainable buffers) ---
        # These hold the recurrent state of the model.
        # They are initialized here but will be managed by a `reset_state` method.
        self.register_buffer("x", torch.zeros(1, n))
        self.register_buffer("y", torch.zeros(1, n))
        self.register_buffer("v", torch.zeros(1, d))
        self.register_buffer("rho", torch.zeros(1, d, n)) # The state matrix

    def reset_state(self, batch_size: int, device: torch.device):
        """
        Resets the recurrent state variables to zero for a new batch.
        """
        self.x = torch.zeros(batch_size, self.n, device=device)
        self.y = torch.zeros(batch_size, self.n, device=device)
        self.v = torch.zeros(batch_size, self.d, device=device)
        self.rho = torch.zeros(batch_size, self.d, self.n, device=device)

    def step(self, token_idx: torch.Tensor) -> torch.Tensor:
        """
        Performs a single recurrent step of the BDH-GPU model.

        Args:
            token_idx (torch.Tensor): A tensor of token indices of shape (B,).

        Returns:
            torch.Tensor: The output vector `v*` of shape (B, d).
        """
        # (B, d) - Get the embedding for the current token. This is v_{t-1} in the paper.
        v_prev = self.token_emb[token_idx]

        # --- Update x-state (Equation 8, x_t line) ---
        # Leaky integration of the projected input vector.
        x_update = relu(v_prev @ self.Dx.T) # (B,n)
        x_t = self.x_decay * self.x + x_update

        # --- Attention Readout (Linear Attention) ---
        # rho is (B, d, n), x_t is (B, n) -> a_star is (B, d)
        a_star = torch.einsum('b...dn,bn->bd', self.rho, x_t)

        # --- Update y-state (Equation 8, y_t line) ---
        # This is the "feed-forward" part of the step.
        y_core = self.Dy @ layernorm_row(a_star) # (B, n)
        y_t = relu(y_core) * relu(x_t) # Gated by positive part of x

        # --- Compute Output Vector (v*) ---
        # This is the output of the current step, to be used for prediction.
        v_star = layernorm_row(y_t @ self.E.T) # (B, d)

        # --- Update State Matrix rho (Equation 8, rho_t line) ---
        # Hebbian-style rank-1 update: "neurons that fire together, wire together".
        # v_prev is pre-synaptic, x_t is post-synaptic.
        # v_prev: (B,d), x_t: (B,n) -> outer product: (B,d,n)
        increment = torch.einsum('bd,bn->bdn', v_prev, x_t)
        self.rho = self.u_decay * (self.rho + increment)

        # --- Commit new state for the next step ---
        self.x, self.y, self.v = x_t, y_t, v_star

        return v_star

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Processes a sequence of tokens and returns the output vectors for each step.

        Args:
            idx (torch.Tensor): A tensor of token indices of shape (B, T).

        Returns:
            torch.Tensor: A tensor of output vectors of shape (B, T, d).
        """
        B, T = idx.shape
        self.reset_state(B, idx.device)

        outputs = []
        for t in range(T):
            token_idx = idx[:, t]
            v_out = self.step(token_idx)
            outputs.append(v_out.unsqueeze(1))

        return torch.cat(outputs, dim=1)
