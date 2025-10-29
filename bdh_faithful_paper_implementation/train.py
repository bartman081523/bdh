# train.py (VERSION 3 - mit --evaluate Schalter)
# Fügt eine Evaluierungsphase nach jeder Epoche hinzu, um die
# Memorisierungs-Genauigkeit auf einem Teil der Trainingsdaten zu messen.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from bdh_paper import BDH_GPU
import random

def load_text_and_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f: text = f.read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos, text # Gebe auch den rohen Text zurück

def get_sequential_batch(data, block_size, batch_size, step, device):
    start_index = step * block_size * batch_size
    ix = [start_index + i * block_size for i in range(batch_size)]
    if any(i + block_size + 1 > len(data) for i in ix): return None, None
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def debug_model_state(model, epoch):
    print("\n" + "="*50 + f"\nDEBUGGING MODEL STATE AFTER EPOCH {epoch}\n" + "="*50)
    tensors = {
        "PARAM_E": model.core.E, "PARAM_Dx": model.core.Dx, "PARAM_Dy": model.core.Dy,
        "PARAM_Head": model.head.weight, "STATE_rho": model.core.rho, "STATE_x": model.core.x,
        "STATE_y": model.core.y, "STATE_v": model.core.v
    }
    for name, t in tensors.items():
        if torch.isnan(t).any() or torch.isinf(t).any(): print(f"!!! WARNING: {name} contains NaN/Inf!")
        norm, mean, std, min_val, max_val = t.norm().item(), t.mean().item(), t.std().item(), t.min().item(), t.max().item()
        print(f"  {name:<12} | Shape: {str(list(t.shape)):<20} | Norm: {norm:<8.4f} | Mean: {mean:<8.4f} | Std: {std:<8.4f} | Min: {min_val:<8.4f} | Max: {max_val:<8.4f}")
    print("="*50 + "\n")

# ==============================================================================
# NEUE EVALUIERUNGSFUNKTION
# ==============================================================================
@torch.no_grad()
def evaluate_memorization(model: nn.Module, full_text: str, stoi: dict, itos: dict, device: torch.device):
    print("\n" + "-"*60)
    print(" RUNNING MEMORIZATION EVALUATION")
    print("-"*60)

    model.eval() # Schalte das Modell in den Evaluationsmodus

    # Wähle eine zufällige, lange Sequenz aus den Trainingsdaten
    eval_seq_len = 256
    start_index = random.randint(0, len(full_text) - eval_seq_len - 1)
    eval_text = full_text[start_index : start_index + eval_seq_len]

    # Teile die Sequenz in Prompt und Ground Truth
    prompt_len = eval_seq_len // 4
    prompt = eval_text[:prompt_len]
    ground_truth = eval_text[prompt_len:]

    prompt_ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

    # Modellzustand aufwärmen
    model.core.reset_state(1, device)
    if prompt_ids.shape[1] > 1:
        _ = model(prompt_ids[:, :-1])

    current_token_idx = prompt_ids[:, -1]

    correct_predictions = 0
    total_predictions = len(ground_truth)

    for expected_char in ground_truth:
        v_out = model.core.step(current_token_idx)
        logits = model.head(v_out)

        # Greedy-Decoding für strikte Memorisierungsprüfung
        next_token_idx = torch.argmax(logits, dim=-1)
        predicted_char = itos.get(next_token_idx.item())

        if predicted_char == expected_char:
            correct_predictions += 1

        current_token_idx = next_token_idx

    accuracy = correct_predictions / total_predictions

    print(f"Evaluierungssequenz (Ausschnitt): '{prompt.replace(chr(10), ' ')}[...]'")
    print(f"Memorisierungs-Genauigkeit: {correct_predictions}/{total_predictions} ({accuracy:.2%})")
    print("-"*60 + "\n")

    model.train() # Schalte das Modell zurück in den Trainingsmodus
    return accuracy

class BDHLanguageModel(nn.Module):
    def __init__(self, core: BDH_GPU):
        super().__init__()
        self.core, self.head = core, nn.Linear(core.d, core.V)
    def forward(self, idx: torch.Tensor):
        return self.head(self.core(idx))

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data, stoi, itos, full_text = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded text with {len(data)} characters, vocabulary size {vocab_size}.")

    core_model = BDH_GPU(n=args.n, d=args.d, V=vocab_size, u_decay=args.u_decay, x_decay=args.x_decay)
    model = BDHLanguageModel(core_model).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting training...")
    num_batches_per_epoch = len(data) // (args.block_size * args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"\n--- Starting Epoch {epoch}/{args.epochs} ---")
        model.core.reset_state(args.batch_size, device)

        for step in range(num_batches_per_epoch):
            x, y = get_sequential_batch(data, args.block_size, args.batch_size, step, device)
            if x is None: break
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if step % args.log_interval == 0:
                print(f"Epoch {epoch} | Step [{step}/{num_batches_per_epoch}] | Loss: {loss.item():.4f}")

        if args.debug:
            debug_model_state(model, epoch)

        # --- INTEGRATION DER EVALUIERUNG ---
        if args.evaluate:
            evaluate_memorization(model, full_text, stoi, itos, device)

    print("\nTraining complete. Saving checkpoint to bdh_model_checkpoint.pt")

    state_dict = model.state_dict()
    buffers_to_remove = [name for name, _ in model.named_buffers()]
    params_to_save = {k: v for k, v in state_dict.items() if k not in buffers_to_remove}

    config = {'n': args.n, 'd': args.d, 'V': vocab_size, 'u_decay': args.u_decay, 'x_decay': args.x_decay}
    checkpoint = {'model_state_dict': params_to_save, 'stoi': stoi, 'itos': itos, 'config': config}
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
    parser.add_argument("--debug", action='store_true', help="Enable detailed logging after each epoch.")
    # --- NEUER SCHALTER ---
    parser.add_argument("--evaluate", action='store_true', help="Run memorization evaluation after each epoch.")

    args = parser.parse_args()
    train(args)
