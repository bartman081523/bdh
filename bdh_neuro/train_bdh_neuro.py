#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Train the BDHNeuroRefTorch model on a text corpus
------------------------------------------------------------

This script trains the biologically-inspired BDH-GPU architecture (Definition 4 + all
neuro extensions) on a given text file (e.g., input.txt). The model learns
next-character prediction (a simple language modeling task).

All neuro-inspired options are active:
- Multi-timescale STDP-like decay (U_kernels)
- Local synaptic forgetting
- Homeostatic activity target
- k-WTA lateral inhibition
- Dendritic subunits (branch nonlinearities)
- Neuromodulation via surprisal
- Stochastic spikes

Usage:
    python train.py --file input.txt --epochs 5 --device cuda
"""

import argparse, torch, torch.nn.functional as F
from torch import nn, optim
from bdh_gpu_neuro_torch import BDHNeuroRefTorch


# ------------------------------ Simple text tokenizer ------------------------------
def load_text_as_ids(path: str):
    """Reads a text file and maps each unique character to an integer ID."""
    text = open(path, "r", encoding="utf-8").read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return ids, stoi, {i: ch for ch, i in stoi.items()}


# ------------------------------ Wrapper model ------------------------------
class BDHLanguageModel(nn.Module):
    """A simple wrapper that predicts next-token logits from the BDHNeuroRefTorch core."""
    def __init__(self, model: BDHNeuroRefTorch):
        super().__init__()
        self.core = model
        self.head = nn.Linear(model.d, model.V)  # maps v* to next-token logits

    def forward(self, token_idx):
        """Processes one token and returns logits + internal metrics."""
        metrics = self.core.step(int(token_idx))
        logits = self.head(self.core.v)
        return logits, metrics


# ------------------------------ Training loop ------------------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ids, stoi, itos = load_text_as_ids(args.file)
    print(f"Loaded text of length {len(ids)} with vocab size {len(stoi)}")

    # Build model with ALL neuro options enabled
    model = BDHNeuroRefTorch(
        n=args.n, d=args.d, V=len(stoi), seed=3, device=device,
        U_kernels=[0.99, 0.97, 0.94], U_weights=[0.5, 0.3, 0.2],
        local_forget_eta=0.02,
        homeostasis_tau=0.15 * args.n,
        k_wta=max(1, args.n // 8),
        branches=2, branch_nl="softplus",
        mod_gamma_max=0.8, spike_rate=0.01,
        ln_before_Dy=True, use_relu_lowrank=True,
        x_decay=0.97
    ).to(device)

    lm = BDHLanguageModel(model).to(device)
    opt = optim.AdamW(lm.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    ids = ids.to(device)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        # Reset internal states before each epoch
        lm.core.x.zero_(); lm.core.y.zero_(); lm.core.v.zero_(); lm.core.rho.zero_()

        for t in range(len(ids) - 1):
            x_idx, y_idx = ids[t], ids[t + 1]
            logits, metrics = lm(x_idx)
            loss = loss_fn(logits.view(1, -1), y_idx.view(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

            if (t + 1) % 500 == 0:
                print(f"[epoch {epoch}] step {t+1}/{len(ids)} "
                      f"loss={loss.item():.4f} "
                      f"sx={metrics['sparsity_x']:.2f} sy={metrics['sparsity_y']:.2f} "
                      f"ρF={metrics['rho_F']:.2f} gain={metrics['gain']:.2f}")

        print(f"Epoch {epoch}: mean loss = {total_loss / len(ids):.5f}")

    # Save trained model
    torch.save(lm.state_dict(), "bdh_neuro_model.pt")
    print("Model saved to bdh_neuro_model.pt")


# ------------------------------ CLI entry point ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="input.txt", help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n", type=int, default=128, help="Neuron dimension")
    parser.add_argument("--d", type=int, default=32, help="Latent (embedding) dimension")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    args = parser.parse_args()
    train(args)
