#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_baseline.py — text generation for BDH-GPU Reference Model
-------------------------------------------------------------------
Loads the BDHGPURefTorch baseline and generates text autoregressively
from a starting prompt.

Usage:
    python generate_baseline.py --file data/divine_comedy_full.txt --start "Hello " --steps 500
"""

import argparse
import torch
from bdh_gpu_neuro_torch import BDHGPURefTorch

# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def load_text_and_vocab(path):
    text = open(path, encoding="utf-8").read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return text, stoi, itos


# ---------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text, stoi, itos = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded vocabulary (|V|={vocab_size})")

    # Initialize model (same architecture as training)
    model = BDHGPURefTorch(n=128, d=32, V=vocab_size, seed=42, device=device).to(device)
    model.eval()

    # Optionally: load trained weights if you saved them
    # Example:
    # model.load_state_dict(torch.load("baseline_model.pt"))

    prompt = args.start
    print(f"Generating text starting with: '{prompt}'\n")

    generated = list(prompt)
    for ch in prompt:
        idx = torch.tensor([stoi.get(ch, 0)], device=device)
        _ = model.step(int(idx))  # warm-up

    # Generate continuation
    for _ in range(args.steps):
        # Linear readout from v to vocab logits
        v = model.v
        logits = v @ model.token_emb.T
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        next_ch = itos[next_idx]
        generated.append(next_ch)

        # Feed back token
        _ = model.step(next_idx)

    text_out = "".join(generated)
    print("--- GENERATED TEXT ---")
    print(text_out)
    with open("generated_baseline.txt", "w", encoding="utf-8") as f:
        f.write(text_out)
    print("\nSaved to generated_baseline.txt")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to text file (same as training)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--start", type=str, default="Hello ", help="Prompt to begin generation")
    parser.add_argument("--steps", type=int, default=500, help="Number of tokens to generate")
    args = parser.parse_args()
    generate(args)
