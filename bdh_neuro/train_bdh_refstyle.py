#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bdh_refstyle.py — Reference-style trainer for stabilized BDH baseline
----------------------------------------------------------------------------
- Byte-level dataset via memmap (same as your reference train.py)
- Autocast/GradScaler for mixed precision
- AdamW optimizer
- Trains ONLY a small readout head on top of the nonparametric BDH core
  (core dynamics: self-organizing; readout: supervised)
"""

import os
from contextlib import nullcontext
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the stabilized baseline (from the file you saved earlier)
from bdh_gpu_ref_stabilized import BDHGPURefStabilized

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="input.txt", help="path to byte text file")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n", type=int, default=128, help="BDH n (neurons)")
    p.add_argument("--d", type=int, default=32, help="BDH latent dim")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dtype", type=str, default="float16", choices=["float32","bfloat16","float16"])
    p.add_argument("--x_decay", type=float, default=0.97)
    p.add_argument("--threshold_ratio", type=float, default=0.02)
    return p.parse_args()

# ----------------------- Data -----------------------
def get_batch(memmap_bytes, block_size, batch_size, device):
    ix = torch.randint(len(memmap_bytes) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(memmap_bytes[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(memmap_bytes[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ----------------------- Model wrapper -----------------------
class BDHReadout(nn.Module):
    """
    Wrap BDHGPURefStabilized core with a trainable linear head only.
    Core remains nonparametric (no nn.Parameters).
    """
    def __init__(self, core: BDHGPURefStabilized, vocab_size: int):
        super().__init__()
        self.core = core.eval()  # core is self-organizing; no gradients
        self.lm_head = nn.Linear(core.d, vocab_size)  # trainable readout

    def forward_seq(self, ids: torch.Tensor):
        """
        ids: (T,) int64 sequence for one stream.
        Returns logits: (T, vocab), and last-step metrics from core.
        """
        T = ids.size(0)
        logits = []
        metrics = {}
        for t in range(T):
            metrics = self.core.step(int(ids[t]))
            v = self.core.v  # (d,)
            logits.append(self.lm_head(v).unsqueeze(0))
        return torch.cat(logits, dim=0), metrics

# ----------------------- Train -----------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # dtype/AMP context
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = (torch.amp.autocast(device_type=device.type, dtype=ptdtype)
           if device.type == "cuda" else nullcontext())
    scaler = torch.amp.GradScaler(device=device.type, enabled=(args.dtype == "float16"))

    # load bytes as memmap (like the reference)
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Data file not found: {args.file}")
    data_mm = np.memmap(args.file, dtype=np.uint8, mode="r")
    vocab_size = 256  # byte-level

    # core (nonparametric) + readout head
    core = BDHGPURefStabilized(n=args.n, d=args.d, V=vocab_size,
                               seed=args.seed, device=str(device),
                               x_decay=args.x_decay, threshold_ratio=args.threshold_ratio)
    model = BDHReadout(core, vocab_size=vocab_size).to(device)
    model = torch.compile(model) if hasattr(torch, "compile") else model

    # optimize head params only
    optimizer = torch.optim.AdamW(model.lm_head.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_acc = 0.0
    loss_steps = 0

    for step in range(1, args.iters + 1):
        x, y = get_batch(data_mm, args.block_size, args.batch_size, device)

        # For simplicity and to stay faithful to BDH core being single-stream stateful,
        # train on the first sequence in the batch (you can loop over batch if desired).
        seq = x[0]    # (T,)
        tgt = y[0]    # (T,)

        # reset core state each iteration (so streams don't leak)
        with torch.no_grad():
            core.x.zero_(); core.y.zero_(); core.v.zero_(); core.rho.zero_()

        with ctx:
            logits, metrics = model.forward_seq(seq)  # (T, vocab)
            loss = F.cross_entropy(logits, tgt)

        loss_acc += float(loss.item()); loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_freq == 0:
            print(f"[{step}/{args.iters}] loss={loss_acc/loss_steps:.3f} "
                  f"sx={metrics['sparsity_x']:.2f} sy={metrics['sparsity_y']:.2f} "
                  f"ρF={metrics['rho_F']:.2f} rank={metrics['rho_eff_rank']:.2f}")
            loss_acc = 0.0; loss_steps = 0

    # Save head weights
    torch.save(model.state_dict(), "bdh_refstyle_head.pt")
    print("Done. Saved readout to bdh_refstyle_head.pt")

if __name__ == "__main__":
    main()
