# train.py
# Kombinierte Version mit SOTA-Trainingspraktiken für robuste und effiziente Ergebnisse.
# VERSION 11: Implementiert Early Stopping basierend auf Validierungs-Intervallen in Steps
#             statt Epochen für eine deutlich granularere und intelligentere Steuerung.

import os
import argparse
import random
import sys
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import requests

# Importiere beide Modellarchitekturen
import bdh
import gpt

# --- Globale Konfiguration & Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
PTDTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
CTX = torch.amp.autocast(device_type=DEVICE.type, dtype=PTDTYPE) if "cuda" in str(DEVICE) else nullcontext()
SCALER = torch.amp.GradScaler(enabled=(DTYPE == "float16"))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed):
    print(f"Setting random seed to {seed}")
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_text_and_vocab(path: str):
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare dataset to {path}...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(path, "w", encoding="utf-8") as f: f.write(requests.get(data_url).text)
    with open(path, "r", encoding="utf-8") as f: text = f.read()
    vocab = sorted(list(set(text))); stoi = {ch: i for i, ch in enumerate(vocab)}; itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos, text

class SlidingWindowDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens; self.seq_len = seq_len
    def __len__(self): return len(self.tokens) - self.seq_len
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]; x = chunk[:-1]; y = chunk[1:]; return x, y

@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.eval(); losses = []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with CTX:
            logits, loss = model(x, y)
        if torch.isfinite(loss): losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def train(args):
    set_seed(args.seed)
    effective_batch_size = args.batch_size; gradient_accumulation_steps = args.gradient_accumulation_steps
    micro_batch_size = effective_batch_size // gradient_accumulation_steps
    assert effective_batch_size % gradient_accumulation_steps == 0, "batch_size muss durch gradient_accumulation_steps teilbar sein."
    print(f"Using device: {DEVICE} with dtype: {DTYPE}")
    print(f"Effective batch size: {effective_batch_size} | Micro batch size: {micro_batch_size} | Grad accumulation steps: {gradient_accumulation_steps}")

    data, stoi, itos, full_text = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded text '{args.file}' with {len(data)} characters, vocabulary size {vocab_size}.")

    full_dataset = SlidingWindowDataset(data, args.block_size)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset aufgeteilt: {len(train_dataset)} Trainings-Samples, {len(val_dataset)} Validierungs-Samples.")

    if args.model_type == 'bdh':
        model_config = bdh.BDHConfig(n_embd=args.d, n_layer=args.n_layer, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_multiplier, vocab_size=vocab_size, dropout=args.dropout)
        model = bdh.BDH(model_config).to(DEVICE)
    elif args.model_type == 'gpt':
        model_config = gpt.GPTConfig(n_layer=8, n_head=8, n_embd=512, block_size=args.block_size, vocab_size=vocab_size)
        model = gpt.GPT(model_config).to(DEVICE)

    print(f"Model '{args.model_type.upper()}' initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    if args.compile:
        print("Compiling the model...")
        try:
            model = torch.compile(model, backend="aot_eager")
            print("Model compiled successfully.")
        except Exception as e: print(f"Warning: torch.compile failed: {e}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim >= 2: decay.append(p)
            else: no_decay.append(p)
    optim_groups = [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.25)

    best_val_loss = float('inf'); patience_counter = 0; global_step = 0; early_stop_triggered = False
    checkpoint_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training for max {args.epochs} epochs... Checkpoints saved in '{checkpoint_dir}'")
    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"\n--- Starting Epoch {epoch}/{args.epochs} ---")

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with CTX:
                logits, loss = model(x, y); loss = loss / gradient_accumulation_steps
            SCALER.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                SCALER.step(optimizer); SCALER.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()

                if global_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch} | Step {global_step} | LR: {lr:.2e} | Loss: {loss.item()*gradient_accumulation_steps:.4f}")

                # --- NEU: Validierung basierend auf Step-Intervall ---
                if args.val_interval > 0 and global_step % args.val_interval == 0:
                    val_loss = evaluate_loss(model, val_loader, DEVICE)
                    print(f"--- Validation @ Step {global_step} --- | Val Loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss; patience_counter = 0
                        print(f"New best val_loss: {best_val_loss:.4f}. Saving best checkpoint...")
                        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                        checkpoint = {'model_state_dict': model_to_save.state_dict(), 'config': model_to_save.config, 'stoi': stoi, 'itos': itos}
                        torch.save(checkpoint, os.path.join(checkpoint_dir, f"{args.model_type}_best.pt"))
                    else:
                        patience_counter += 1

                    if patience_counter >= args.early_stopping_patience:
                        print(f"Early stopping triggered after {patience_counter} validation intervals without improvement.")
                        early_stop_triggered = True; break

        if early_stop_triggered: break

    print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BDH or GPT model with SOTA practices.")
    parser.add_argument("--run_name", type=str, default="run_" + str(int(time.time())))
    parser.add_argument("--file", type=str, default="input.txt")
    parser.add_argument('--model_type', type=str, default='bdh', choices=['bdh', 'gpt'])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    # --- NEU: Argumente für Step-basierte Validierung und Geduld ---
    parser.add_argument("--val_interval", type=int, default=10, help="Validate every N training steps. 0 to disable interval validation.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Stop if val loss doesn't improve for N validation intervals.")

    parser.add_argument('--no-compile', action='store_false', dest='compile')
    parser.set_defaults(compile=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--mlp_multiplier", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device == 'cpu': DEVICE = torch.device('cpu')
    train(args)
