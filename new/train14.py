# train.py
# Basiert auf der stabilen und effektiven Version 9 und fügt intelligentes Checkpointing hinzu.
# VERSION 14: Die Synthese aus der Lerndynamik von V9 und robustem Speichern.

import os
import argparse
import random
import sys
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import requests

# Importiere beide Modellarchitekturen
import bdh
import gpt

# --- Globale Konfiguration & Setup (wie in V9) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
PTDTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
CTX = torch.amp.autocast(device_type=DEVICE.type, dtype=PTDTYPE) if "cuda" in str(DEVICE) else nullcontext()
SCALER = torch.amp.GradScaler(enabled=(DTYPE == "float16"))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed):
    print(f"Setting random seed to {seed}")
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); torch.manual_seed(seed)
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

class EpochDataLoader:
    def __init__(self, data, block_size, batch_size):
        self.data = data; self.block_size = block_size; self.batch_size = batch_size
        self.num_sequences = len(data) - block_size
        self.indices = torch.arange(self.num_sequences)
    def __len__(self): return self.num_sequences // self.batch_size
    def __iter__(self):
        shuffled_indices = self.indices[torch.randperm(len(self.indices))]
        for b in range(len(self)):
            start_idx = b * self.batch_size; end_idx = start_idx + self.batch_size
            batch_indices = shuffled_indices[start_idx:end_idx]
            x = torch.stack([self.data[j : j + self.block_size] for j in batch_indices])
            y = torch.stack([self.data[j + 1 : j + 1 + self.block_size] for j in batch_indices])
            yield x, y

@torch.no_grad()
def evaluate_robust_memorization(model, full_text: str, stoi: dict, itos: dict, device: torch.device, epoch: int, step: int):
    print("\n" + "-"*60 + f"\n RUNNING ROBUST MEMORIZATION EVALUATION | EPOCH {epoch} | STEP {step}\n" + "-"*60)
    model.eval()
    NUM_PROBES, PROBE_LEN, PROMPT_RATIO = 10, 256, 0.25
    total_correct, total_predictions, probe_accuracies = 0, 0, []
    for i in range(NUM_PROBES):
        start_index = int((i / NUM_PROBES) * (len(full_text) - PROBE_LEN - 1))
        eval_text = full_text[start_index : start_index + PROBE_LEN]
        prompt_len = int(PROBE_LEN * PROMPT_RATIO)
        prompt, ground_truth = eval_text[:prompt_len], eval_text[prompt_len:]
        if not prompt or not ground_truth: continue
        prompt_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
        generated_ids = model.generate(prompt_ids, max_new_tokens=len(ground_truth), top_k=1)
        generated_text = "".join([itos.get(idx.item(), '') for idx in generated_ids[0, len(prompt_ids[0]):]])
        probe_correct = sum(1 for (pred, true) in zip(generated_text, ground_truth) if pred == true)
        total_correct += probe_correct; total_predictions += len(ground_truth)
        probe_accuracy = probe_correct / len(ground_truth) if len(ground_truth) > 0 else 0
        probe_accuracies.append(probe_accuracy)
        sys.stdout.write(f"\r  Running probes... {i+1}/{NUM_PROBES}"); sys.stdout.flush()
    print()
    if total_predictions > 0:
        formatted_accuracies = [f"{acc:.2%}" for acc in probe_accuracies]
        print(f"  Individual Probe Accuracies: [{', '.join(formatted_accuracies)}]")
        final_accuracy = total_correct / total_predictions
        print(f"\nAggregierte Memorisierungs-Genauigkeit: {total_correct}/{total_predictions} ({final_accuracy:.2%})")
    else: print("  Nicht genügend Daten für Memorisierungs-Probe.")
    print("-"*60 + "\n")
    model.train()

def train(args):
    if args.seed is not None: set_seed(args.seed)

    effective_batch_size = args.batch_size; gradient_accumulation_steps = args.gradient_accumulation_steps
    micro_batch_size = effective_batch_size // gradient_accumulation_steps
    assert effective_batch_size % gradient_accumulation_steps == 0
    print(f"Using device: {DEVICE} with dtype: {DTYPE}")
    print(f"Effective batch size: {effective_batch_size} | Micro batch size: {micro_batch_size} | Grad accumulation steps: {gradient_accumulation_steps}")

    data, stoi, itos, full_text = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded text '{args.file}' with {len(data)} characters, vocabulary size {vocab_size}.")

    if args.model_type == 'bdh':
        model_config = bdh.BDHConfig(n_embd=args.d, n_layer=args.n_layer, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_multiplier, vocab_size=vocab_size, dropout=args.dropout)
        model = bdh.BDH(model_config).to(DEVICE)
    elif args.model_type == 'gpt':
        model_config = gpt.GPTConfig(n_layer=8, n_head=8, n_embd=512, block_size=args.block_size, vocab_size=vocab_size)
        model = gpt.GPT(model_config).to(DEVICE)

    print(f"Model '{args.model_type.upper()}' initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    if args.compile:
        print("Compiling the model...")
        model = torch.compile(model, backend="aot_eager")
        print("Model compiled successfully.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # KEIN TRAIN/VAL SPLIT - wir nutzen den gesamten Datensatz zum Trainieren
    data_loader = EpochDataLoader(data, args.block_size, effective_batch_size)
    steps_per_epoch = len(data_loader)
    print(f"Dataset has {steps_per_epoch} steps per epoch (training on full data).")

    # NEU: Logic für intelligentes Checkpointing
    best_loss = float('inf')
    checkpoint_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training for {args.epochs} epochs... Checkpoints saved in '{checkpoint_dir}'")
    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"\n--- Starting Epoch {epoch}/{args.epochs} ---")

        for step, (x_batch, y_batch) in enumerate(data_loader):
            current_step = step + 1
            optimizer.zero_grad(set_to_none=True)
            total_loss_in_batch = 0.0

            for i in range(0, x_batch.size(0), micro_batch_size):
                x = x_batch[i : i + micro_batch_size].to(DEVICE)
                y = y_batch[i : i + micro_batch_size].to(DEVICE)
                with CTX:
                    _, loss = model(x, y); loss = loss / gradient_accumulation_steps
                total_loss_in_batch += loss.item() * gradient_accumulation_steps
                SCALER.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            SCALER.step(optimizer); SCALER.update()

            if current_step % args.log_interval == 0 or current_step == steps_per_epoch:
                print(f"Epoch {epoch} | Step [{current_step}/{steps_per_epoch}] | Loss: {total_loss_in_batch:.4f}")

            # NEU: Speichere den besten Checkpoint basierend auf dem Trainings-Loss
            if total_loss_in_batch < best_loss:
                best_loss = total_loss_in_batch
                print(f"  -> New best train loss: {best_loss:.4f}. Saving best checkpoint...")
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                checkpoint = {'model_state_dict': model_to_save.state_dict(), 'config': model_to_save.config, 'stoi': stoi, 'itos': itos}
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"{args.model_type}_best.pt"))

            if args.eval_interval > 0 and (current_step % args.eval_interval == 0 or current_step == steps_per_epoch):
                 if hasattr(model, 'generate'):
                    evaluate_robust_memorization(model, full_text, stoi, itos, DEVICE, epoch, current_step)

    print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BDH or GPT model based on the stable and effective Version 9.")
    parser.add_argument("--run_name", type=str, default="run_" + str(int(time.time())), help="A name for the run, used for checkpointing.")
    parser.add_argument("--file", type=str, default="input.txt")
    parser.add_argument('--model_type', type=str, default='bdh', choices=['bdh', 'gpt'])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument('--no-compile', action='store_false', dest='compile'); parser.set_defaults(compile=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10, help="Run memorization probes every N steps. 0 to disable.")
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
