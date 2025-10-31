# train.py (VERSION 15 - Final & Lauffähig)
# Nutzt die korrekte RoPE Implementierung und eine saubere Architektur.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import bdh_paper
from bdh_paper import BDH_GPU, RotaryEmbedding
import random
import sys
import os

def load_text_and_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f: text = f.read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos, text

def get_batch(data, block_size, batch_size, device, is_sequential=False, step=0):
    if is_sequential:
        total_len = len(data)
        if total_len <= block_size: return None, None
        num_sequences_per_batch = (total_len - 1) // block_size // batch_size
        if num_sequences_per_batch == 0: return None, None
        start_index = step * block_size
        if start_index >= num_sequences_per_batch * block_size: return None, None
        ix_starts = [start_index + i * (num_sequences_per_batch * block_size) for i in range(batch_size)]
        if any(i + block_size + 1 > len(data) for i in ix_starts): return None, None
        x = torch.stack([data[i:i+block_size] for i in ix_starts])
        y = torch.stack([data[i+1:i+1+block_size] for i in ix_starts])
    else:
        if len(data) <= block_size: return None, None
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def debug_model_state(model, epoch):
    print("\n" + "="*50 + f"\nDEBUGGING MODEL STATE AFTER EPOCH {epoch}\n" + "="*50)
    tensors = {"PARAM_E": model.core.E, "PARAM_Dx": model.core.Dx, "PARAM_Dy": model.core.Dy,"PARAM_Head": model.head.weight, "STATE_rho": model.core.rho_state, "STATE_x": model.core.x_state}
    for name, t in tensors.items():
        if t is None: continue
        if torch.isnan(t).any() or torch.isinf(t).any(): print(f"!!! WARNING: {name} contains NaN/Inf!")
        norm, mean, std, min_val, max_val = t.norm().item(), t.mean().item(), t.std().item(), t.min().item(), t.max().item()
        print(f"  {name:<12} | Shape: {str(list(t.shape)):<20} | Norm: {norm:<8.4f} | Mean: {mean:<8.4f} | Std: {std:<8.4f} | Min: {min_val:<8.4f} | Max: {max_val:<8.4f}")
    print("="*50 + "\n")

class BDHLanguageModel(nn.Module):
    def __init__(self, core: BDH_GPU):
        super().__init__()
        self.core = core
        self.head = nn.Linear(core.d, core.V, bias=False)
        self.rope = RotaryEmbedding(core.d)

    def forward(self, idx: torch.Tensor, x_state=None, rho_state=None):
        seq_len = idx.shape[1]
        token_embeddings = self.core.token_emb(idx)
        rotated_embeddings = self.rope(token_embeddings, seq_len=seq_len)
        v_out = self.core(rotated_embeddings, x_state, rho_state)
        return self.head(v_out), self.core.x_state, self.core.rho_state

@torch.no_grad()
def evaluate_robust_memorization(model: BDHLanguageModel, full_text: str, stoi: dict, itos: dict, device: torch.device):
    print("\n" + "-"*60)
    print(" RUNNING ROBUST MEMORIZATION EVALUATION")
    print("-"*60)

    model.eval()

    NUM_PROBES = 10
    PROBE_LEN = 128
    PROMPT_RATIO = 0.25

    total_correct = 0
    total_predictions = 0
    probe_accuracies = []

    for i in range(NUM_PROBES):
        start_index = int((i / NUM_PROBES) * (len(full_text) - PROBE_LEN - 1))
        eval_text = full_text[start_index : start_index + PROBE_LEN]

        prompt_len = int(PROBE_LEN * PROMPT_RATIO)
        prompt = eval_text[:prompt_len]
        ground_truth = eval_text[prompt_len:]

        prompt_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)

        model.core.reset_state(1, device)
        x_state, rho_state = model.core.x_state, model.core.rho_state

        if prompt_ids.shape[1] > 1:
            _, x_state, rho_state = model(prompt_ids[:, :-1], x_state, rho_state)

        current_token_idx = prompt_ids[:, -1]

        probe_correct = 0
        for t, expected_char in enumerate(ground_truth):
            position = prompt_len + t

            token_emb = model.core.token_emb(current_token_idx)
            rotated_emb = model.rope(token_emb.unsqueeze(1), seq_len=1, offset=position)

            v_out, x_state, rho_state = model.core.step(rotated_emb.squeeze(1), x_state, rho_state)

            logits = model.head(v_out)
            next_token_idx = torch.argmax(logits, dim=-1)
            predicted_char = itos.get(next_token_idx.item())

            if predicted_char == expected_char:
                probe_correct += 1
            current_token_idx = next_token_idx

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
    global DIM_DEBUG
    if args.dim_debug:
        bdh_paper.DIM_DEBUG = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    data, stoi, itos, full_text = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded text with {len(data)} characters, vocabulary size {vocab_size}.")

    core_model = BDH_GPU(n=args.n, d=args.d, V=vocab_size, u_decay=args.u_decay, x_decay=args.x_decay)
    model = BDHLanguageModel(core_model).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting training...")

    is_small_dataset = len(data) < args.block_size * args.batch_size * 5
    if is_small_dataset:
        print("Small dataset detected. Using random sampling for training.")
        steps_per_epoch = 100
    else:
        print("Large dataset detected. Using sequential batching.")
        steps_per_epoch = (len(data) - 1) // (args.block_size * args.batch_size)

    x_state, rho_state = None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"\n--- Starting Epoch {epoch}/{args.epochs} ---")
        if not is_small_dataset:
            model.core.reset_state(args.batch_size, device)
            x_state, rho_state = model.core.x_state, model.core.rho_state

        for step in range(steps_per_epoch):
            if is_small_dataset:
                 model.core.reset_state(args.batch_size, device)
                 x_state, rho_state = model.core.x_state, model.core.rho_state

            x, y = get_batch(data, args.block_size, args.batch_size, device, is_sequential=not is_small_dataset, step=step)
            if x is None: break

            logits, x_state, rho_state = model(x, x_state, rho_state)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % args.log_interval == 0 or (step == steps_per_epoch - 1):
                print(f"Epoch {epoch} | Step [{step+1}/{steps_per_epoch}] | Loss: {loss.item():.4f}")

        if args.debug:
            debug_model_state(model, epoch, x_state, rho_state)

        if args.evaluate:
            evaluate_robust_memorization(model, full_text, stoi, itos, device)

    print("\nTraining complete. Saving checkpoint to bdh_model_checkpoint.pt")

    state_dict = model.state_dict()
    config = {'n': args.n, 'd': args.d, 'V': vocab_size, 'u_decay': args.u_decay, 'x_decay': args.x_decay}
    checkpoint = {'model_state_dict': state_dict, 'stoi': stoi, 'itos': itos, 'config': config}
    torch.save(checkpoint, "bdh_model_checkpoint.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BDH-GPU language model with optional evaluation.")
    parser.add_argument("--file", type=str, required=True, help="Path to the training text file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs.")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length for TBPTT.")
    parser.add_argument("--n", type=int, default=2048, help="Neuronal dimension.")
    parser.add_argument("--d", type=int, default=128, help="Latent dimension.")
    parser.add_argument("--u_decay", type=float, default=0.97, help="Decay for rho-state.")
    parser.add_argument("--x_decay", type=float, default=0.97, help="Decay for x-state.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--debug", action='store_true', help="Enable detailed logging after each epoch.")
    parser.add_argument("--evaluate", action='store_true', help="Run robust memorization evaluation after each epoch.")
    parser.add_argument("--dim-debug", action='store_true', help="Print tensor dimensions inside RoPE for debugging.")

    args = parser.parse_args()
    train(args)
