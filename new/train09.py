# train.py
# Kombinierte Version mit Fokus auf einem robusten und konsistenten Epochen-basierten Training.
# VERSION 9: Fügt einen flexiblen --evaluate_interval Parameter für granulare Evaluierung hinzu.

import os
import argparse
import random
import sys
from contextlib import nullcontext
import torch
import torch.nn as nn
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
    """Setzt den Random Seed für Reproduzierbarkeit."""
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_text_and_vocab(path: str):
    """Lädt Textdaten, erstellt ein Vokabular und tokenisiert den Text."""
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare dataset to {path}...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)
        print("Dataset downloaded.")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos, text

class EpochDataLoader:
    """Ein einfacher Datenlader, der für Epochen-basiertes Training sorgt."""
    def __init__(self, data, block_size, batch_size):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_sequences = len(data) - block_size
        self.indices = torch.arange(self.num_sequences)

    def __len__(self):
        return self.num_sequences // self.batch_size

    def __iter__(self):
        shuffled_indices = self.indices[torch.randperm(len(self.indices))]
        for b in range(len(self)):
            start_idx = b * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = shuffled_indices[start_idx:end_idx]
            x = torch.stack([self.data[j : j + self.block_size] for j in batch_indices])
            y = torch.stack([self.data[j + 1 : j + 1 + self.block_size] for j in batch_indices])
            yield x, y

@torch.no_grad()
def evaluate_robust_memorization(model, full_text: str, stoi: dict, itos: dict, device: torch.device, epoch: int, step: int):
    """Evaluiert, wie gut das Modell den Trainings-Text auswendig gelernt hat."""
    print("\n" + "-"*60 + f"\n RUNNING ROBUST MEMORIZATION EVALUATION | EPOCH {epoch} | STEP {step}\n" + "-"*60)
    model.eval()
    NUM_PROBES, PROBE_LEN, PROMPT_RATIO = 10, 256, 0.25
    total_correct, total_predictions, probe_accuracies = 0, 0, []
    for i in range(NUM_PROBES):
        start_index = int((i / NUM_PROBES) * (len(full_text) - PROBE_LEN - 1))
        eval_text = full_text[start_index : start_index + PROBE_LEN]
        prompt_len = int(PROBE_LEN * PROMPT_RATIO)
        prompt, ground_truth = eval_text[:prompt_len], eval_text[prompt_len:]
        prompt_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
        generated_ids = model.generate(prompt_ids, max_new_tokens=len(ground_truth), top_k=1)
        generated_text = "".join([itos.get(idx.item(), '') for idx in generated_ids[0, prompt_len:]])
        probe_correct = sum(1 for (pred, true) in zip(generated_text, ground_truth) if pred == true)
        total_correct += probe_correct
        total_predictions += len(ground_truth)
        probe_accuracy = probe_correct / len(ground_truth) if len(ground_truth) > 0 else 0
        probe_accuracies.append(probe_accuracy)
        sys.stdout.write(f"\r  Running probes... {i+1}/{NUM_PROBES}")
        sys.stdout.flush()
    print()
    formatted_accuracies = [f"{acc:.2%}" for acc in probe_accuracies]
    print(f"  Individual Probe Accuracies: [{', '.join(formatted_accuracies)}]")
    final_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    print(f"\nAggregierte Memorisierungs-Genauigkeit: {total_correct}/{total_predictions} ({final_accuracy:.2%})")
    print("-"*60 + "\n")
    model.train()
    return final_accuracy

def train(args):
    """Haupt-Trainingsfunktion."""
    if args.seed is not None:
        set_seed(args.seed)

    effective_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    micro_batch_size = effective_batch_size // gradient_accumulation_steps
    assert effective_batch_size % gradient_accumulation_steps == 0, "batch_size muss durch gradient_accumulation_steps teilbar sein."
    print(f"Using device: {DEVICE} with dtype: {DTYPE}")
    print(f"Effective batch size: {effective_batch_size} | Micro batch size: {micro_batch_size} | Grad accumulation steps: {gradient_accumulation_steps}")

    data, stoi, itos, full_text = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded text '{args.file}' with {len(data)} characters, vocabulary size {vocab_size}.")

    if args.model_type == 'bdh':
        model_config = bdh.BDHConfig(
            n_embd=args.d, n_layer=args.n_layer, n_head=args.n_head,
            mlp_internal_dim_multiplier=args.mlp_multiplier, vocab_size=vocab_size, dropout=args.dropout
        )
        model = bdh.BDH(model_config).to(DEVICE)
    elif args.model_type == 'gpt':
        model_config = gpt.GPTConfig(
            n_layer=8, n_head=8, n_embd=512, block_size=args.block_size, vocab_size=vocab_size
        )
        model = gpt.GPT(model_config).to(DEVICE)

    print(f"Model '{args.model_type.upper()}' initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    if args.compile:
        print("Compiling the model...")
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, backend="aot_eager")
            print("Model compiled successfully with 'aot_eager' backend.")
        except Exception as e:
            print(f"Warning: torch.compile failed with error: {e}\nContinuing without compilation.")
    else:
        print("Compilation disabled, running in eager mode.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    data_loader = EpochDataLoader(data, args.block_size, effective_batch_size)
    steps_per_epoch = len(data_loader)
    print(f"Dataset has {steps_per_epoch} steps per epoch.")

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"\n--- Starting Epoch {epoch}/{args.epochs} ---")

        for step, (x_batch, y_batch) in enumerate(data_loader):
            current_step = step + 1 # Startet bei 1
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0

            for i in range(0, x_batch.size(0), micro_batch_size):
                x = x_batch[i : i + micro_batch_size].to(DEVICE)
                y = y_batch[i : i + micro_batch_size].to(DEVICE)

                with CTX:
                    _, loss = model(x, y)
                    loss = loss / gradient_accumulation_steps

                total_loss += loss.item() * gradient_accumulation_steps
                SCALER.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            SCALER.step(optimizer)
            SCALER.update()

            if current_step % args.log_interval == 0 or current_step == steps_per_epoch:
                print(f"Epoch {epoch} | Step [{current_step}/{steps_per_epoch}] | Loss: {total_loss:.4f}")

            # NEU: Evaluierung nach dem festgelegten Intervall
            if args.eval_interval > 0 and (current_step % args.eval_interval == 0 or current_step == steps_per_epoch):
                 if hasattr(model, 'generate'):
                    evaluate_robust_memorization(model, full_text, stoi, itos, DEVICE, epoch, current_step)

    print("\nTraining complete. Saving checkpoint...")
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    checkpoint_path = f"{args.model_type}_final_model.pt"
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(), 'config': model_to_save.config,
        'stoi': stoi, 'itos': itos
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Final model and vocab saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BDH or GPT language model with a consistent epoch-based approach.")

    # KERN-ARGUMENTE
    parser.add_argument("--file", type=str, default="input.txt", help="Path to the training text file.")
    parser.add_argument('--model_type', type=str, default='bdh', choices=['bdh', 'gpt'], help='The type of model to train.')
    parser.add_argument("--epochs", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    # BATCH & PERFORMANCE
    parser.add_argument("--batch_size", type=int, default=32, help="Effective batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps to accumulate gradients.")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence length for training.")
    parser.add_argument('--no-compile', action='store_false', dest='compile', help='Disable torch.compile.')
    parser.set_defaults(compile=True)

    # LOGGING & EVALUIERUNG
    parser.add_argument("--log_interval", type=int, default=10, help="Log loss every N steps within an epoch.")
    # NEU: evaluate_interval Parameter
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate every N steps. Set to 0 to disable interval evaluation.")

    # BDH HYPERPARAMETER
    parser.add_argument("--d", type=int, default=256, help="BDH: Embedding dimension (n_embd).")
    parser.add_argument("--n_layer", type=int, default=6, help="BDH: Number of layers.")
    parser.add_argument("--n_head", type=int, default=4, help="BDH: Number of attention heads.")
    parser.add_argument("--mlp_multiplier", type=int, default=128, help="BDH: Multiplier for internal MLP dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # SONSTIGES
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    if args.device == 'cpu':
        DEVICE = torch.device('cpu')

    train(args)
