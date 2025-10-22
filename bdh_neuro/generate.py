#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py — Text generation using BDHNeuroRefTorch
----------------------------------------------------

Uses the trained model (bdh_neuro_model.pt) produced by train.py
to generate text character-by-character.

Usage:
    python generate.py --file input.txt --seed_text "Hello " --length 500 --device cuda
"""

import torch, torch.nn.functional as F
from bdh_gpu_neuro_torch import BDHNeuroRefTorch


# ------------------------------ Vocabulary utils ------------------------------
def load_vocab(path: str):
    """Load the vocabulary (character to ID mappings) from a text file."""
    text = open(path, "r", encoding="utf-8").read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


# ------------------------------ Generation logic ------------------------------
def generate_text(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stoi, itos = load_vocab(args.file)
    V = len(stoi)
    print(f"Loaded vocabulary (|V|={V})")

    # Rebuild model with identical parameters as training
    model = BDHNeuroRefTorch(
        n=args.n, d=args.d, V=V, seed=3, device=device,
        U_kernels=[0.99, 0.97, 0.94], U_weights=[0.5, 0.3, 0.2],
        local_forget_eta=0.02,
        homeostasis_tau=0.15 * args.n,
        k_wta=max(1, args.n // 8),
        branches=2, branch_nl="softplus",
        mod_gamma_max=0.8, spike_rate=0.01,
        ln_before_Dy=True, use_relu_lowrank=True
    ).to(device)

    # Prediction head (same as in train.py)
    head = torch.nn.Linear(model.d, V).to(device)

    # Load saved weights
    state = torch.load("bdh_neuro_model.pt", map_location=device)
    model.load_state_dict({k.replace("core.", ""): v for k, v in state.items() if k.startswith("core.")}, strict=False)
    head.load_state_dict({k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}, strict=False)
    model.eval(); head.eval()

    # Reset internal states
    model.x.zero_(); model.y.zero_(); model.v.zero_(); model.rho.zero_()

    # Initialize with seed text
    seed_text = args.seed_text
    for ch in seed_text[:-1]:
        idx = torch.tensor([stoi.get(ch, 0)], device=device)
        model.step(int(idx))

    current_char = seed_text[-1]
    output = seed_text

    print(f"Generating text starting with: '{seed_text}'")
    for t in range(args.length):
        idx = torch.tensor([stoi.get(current_char, 0)], device=device)
        model.step(int(idx))
        logits = head(model.v).view(1, -1)
        probs = F.softmax(logits / args.temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_char = itos[next_id]
        output += next_char
        current_char = next_char

    print("\n--- GENERATED TEXT ---")
    print(output)
    with open("generated.txt", "w", encoding="utf-8") as f:
        f.write(output)
    print("\nSaved to generated.txt")


# ------------------------------ CLI entry point ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="input.txt", help="Training text file (for vocab reconstruction)")
    parser.add_argument("--seed_text", type=str, default="Hello ", help="Initial text to start generation")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    generate_text(args)
