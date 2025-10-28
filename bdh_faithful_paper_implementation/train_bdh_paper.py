# train_with_debug.py
# Ein verbessertes Trainings-Skript, das Gradient Clipping zur Stabilisierung
# und detailliertes Debug-Logging nach jeder Epoche hinzufügt.

import argparse
import torch
import torch.nn as nn
from torch import optim
from bdh_paper import BDH_GPU

# --- Hilfsfunktionen ---

def load_text_and_vocab(path: str):
    """Liest eine Textdatei, erstellt ein Zeichen-Vokabular und gibt die Daten als IDs zurück."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in stoi.items()}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos

def get_sequential_batch(data, block_size, batch_size, step, device):
    """Erzeugt einen Batch von aufeinanderfolgenden Datenblöcken."""
    start_index = step * block_size * batch_size
    ix = [start_index + i * block_size for i in range(batch_size)]

    if any(i + block_size + 1 > len(data) for i in ix):
        return None, None

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def debug_model_state(model, epoch):
    """Gibt den "Gesundheitszustand" der Modellparameter und Zustände aus."""
    print("\n" + "="*50)
    print(f"DEBUGGING MODEL STATE AFTER EPOCH {epoch}")
    print("="*50)

    tensors_to_check = {
        "PARAM_E": model.core.E,
        "PARAM_Dx": model.core.Dx,
        "PARAM_Dy": model.core.Dy,
        "PARAM_Head": model.head.weight,
        "STATE_rho": model.core.rho,
        "STATE_x": model.core.x,
        "STATE_y": model.core.y,
        "STATE_v": model.core.v
    }

    for name, tensor in tensors_to_check.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"!!! WARNING: Tensor '{name}' contains NaN or Inf values!")

        norm = torch.linalg.norm(tensor.float()).item()
        mean = tensor.float().mean().item()
        std = tensor.float().std().item()
        min_val = tensor.float().min().item()
        max_val = tensor.float().max().item()

        print(f"  {name:<12} | Shape: {str(list(tensor.shape)):<20} | Norm: {norm:<8.4f} | Mean: {mean:<8.4f} | Std: {std:<8.4f} | Min: {min_val:<8.4f} | Max: {max_val:<8.4f}")

    print("="*50 + "\n")


# --- Modell-Wrapper ---

class BDHLanguageModel(nn.Module):
    def __init__(self, core: BDH_GPU):
        super().__init__()
        self.core = core
        self.head = nn.Linear(core.d, core.V)

    def forward(self, idx: torch.Tensor):
        v_out = self.core(idx)
        logits = self.head(v_out)
        return logits

# --- Trainingslogik ---

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, stoi, itos = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded text with {len(data)} characters, vocabulary size {vocab_size}.")

    core_model = BDH_GPU(n=args.n, d=args.d, V=vocab_size, u_decay=args.u_decay, x_decay=args.x_decay)
    model = BDHLanguageModel(core_model).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params/1e6:.2f}M trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting training...")
    num_batches_per_epoch = len(data) // (args.block_size * args.batch_size)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Starting Epoch {epoch}/{args.epochs} ---")

        model.core.reset_state(args.batch_size, device)

        for step in range(num_batches_per_epoch):
            x, y = get_sequential_batch(data, args.block_size, args.batch_size, step, device)

            if x is None:
                break

            logits = model(x)

            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), y.view(B * T))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # --- STABILITÄTS-FIX: GRADIENT CLIPPING ---
            # Verhindert, dass die Gradienten explodieren und NaN-Werte verursachen.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if step % args.log_interval == 0:
                print(f"Epoch {epoch} | Step [{step}/{num_batches_per_epoch}] | Loss: {loss.item():.4f}")

        # Führe den Debug-Check am Ende jeder Epoche aus
        debug_model_state(model, epoch)

    print("\nTraining complete. Saving model parameters to bdh_paper_model.pt")
    model_parameters = {k: v for k, v in model.state_dict().items() if not k.startswith('core.x') and not k.startswith('core.y') and not k.startswith('core.v') and not k.startswith('core.rho')}
    torch.save(model_parameters, "bdh_paper_model.pt")

# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BDH-GPU language model with persistent state and debugging.")
    parser.add_argument("--file", type=str, required=True, help="Path to the training text file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length for backpropagation (TBPTT).")
    parser.add_argument("--n", type=int, default=2048, help="Neuronal dimension.")
    parser.add_argument("--d", type=int, default=128, help="Latent (embedding) dimension.")
    parser.add_argument("--u_decay", type=float, default=0.97, help="Decay for rho-state (memory).")
    parser.add_argument("--x_decay", type=float, default=0.97, help="Decay for x-state (leaky integration).")

    args = parser.parse_args()
    train(args)
