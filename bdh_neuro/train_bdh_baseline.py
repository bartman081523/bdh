#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bdh_baseline.py — Baseline (self-organizing) BDH-GPU Reference
---------------------------------------------------------------------
Evaluates the purely dynamical BDHGPURefTorch model on a text corpus.

No gradient descent is used — this baseline tests intrinsic BDH dynamics.
Outputs diagnostics every N steps.

Usage:
    python train_bdh_baseline.py --file data/divine_comedy_full.txt --device cuda
"""

import argparse
import torch
import torch.nn.functional as F
from bdh_gpu_ref_stabilized import BDHGPURefTorch


# ---------------------------------------------------------------
# Utility: load text and create vocabulary
# ---------------------------------------------------------------

def load_text_and_vocab(path):
    text = open(path, encoding="utf-8").read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return text, stoi, itos


# ---------------------------------------------------------------
# Training (self-organizing evaluation) loop
# ---------------------------------------------------------------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text, stoi, itos = load_text_and_vocab(args.file)
    print(f"Loaded text of length {len(text)} with vocab size {len(stoi)}")

    # Hyperparameters (same as other experiments)
    n, d = 128, 32
    seq_len = 64
    batch_size = 64

    # Initialize BDH-GPU reference model
    model = BDHGPURefTorch(n=n, d=d, V=len(stoi), seed=42, device=device).to(device)
    model.eval()  # no learnable parameters, so no training mode needed

    # Batch sampling utility
    def sample_batch():
        ix = torch.randint(0, len(text) - seq_len - 1, (batch_size,))
        x = torch.tensor([[stoi[c] for c in text[i:i+seq_len]] for i in ix], device=device)
        y = torch.tensor([[stoi[c] for c in text[i+1:i+seq_len+1]] for i in ix], device=device)
        return x, y

    # Main loop
    for step in range(1, args.steps + 1):
        x, y = sample_batch()
        logits = []

        # Feed the sequence token by token
        for t in range(seq_len):
            metrics = model.step(int(x[0, t]))  # one token per step
            v = model.v
            logit = v @ model.token_emb.T
            logits.append(logit.unsqueeze(0))

        # Diagnostic cross-entropy loss (not backpropagated)
        logits = torch.cat(logits, dim=0)  # (seq_len, vocab)
        targets = y[0]
        loss = F.cross_entropy(logits, targets)

        # Logging
        if step % args.log_interval == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"sx={metrics['sparsity_x']:.2f} sy={metrics['sparsity_y']:.2f} "
                f"ρF={metrics['rho_F']:.2f} rank={metrics['rho_eff_rank']:.2f}"
            )

    print("BDH baseline dynamics evaluation complete.")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to text file for training")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=20000, help="Total number of iterations")
    parser.add_argument("--log_interval", type=int, default=500, help="Steps between logs")
    args = parser.parse_args()
    train(args)
