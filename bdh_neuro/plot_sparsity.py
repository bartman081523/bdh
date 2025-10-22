#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sparsity.py — visualize BDH-Neuro dynamics
-----------------------------------------------
Reads training logs printed by train.py (from stdout or a saved log file)
and plots sx, sy, ρF, gain, and loss over time.

Usage:
    python plot_sparsity.py --file train.log
"""

import re, argparse
import matplotlib.pyplot as plt

pattern = re.compile(
    r"loss=([\d\.]+)\s+sx=([\d\.]+)\s+sy=([\d\.]+)\s+ρF=([\d\.]+)\s+gain=([\d\.]+)"
)

def parse_log(file):
    losses, sx, sy, rhoF, gain = [], [], [], [], []
    for line in open(file, "r", encoding="utf-8"):
        m = pattern.search(line)
        if m:
            l, x, y, r, g = map(float, m.groups())
            losses.append(l); sx.append(x); sy.append(y); rhoF.append(r); gain.append(g)
    return losses, sx, sy, rhoF, gain

def plot_metrics(losses, sx, sy, rhoF, gain):
    fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    axs[0].plot(losses); axs[0].set_ylabel("Loss")
    axs[1].plot(sx); axs[1].set_ylabel("Sparsity x")
    axs[2].plot(sy); axs[2].set_ylabel("Sparsity y")
    axs[3].plot(rhoF); axs[3].set_ylabel("ρF (norm)")
    axs[4].plot(gain); axs[4].set_ylabel("Gain")
    axs[-1].set_xlabel("Step (per 500 updates)")
    for ax in axs: ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="train.log",
                        help="Path to log file containing training prints")
    args = parser.parse_args()
    metrics = parse_log(args.file)
    plot_metrics(*metrics)
