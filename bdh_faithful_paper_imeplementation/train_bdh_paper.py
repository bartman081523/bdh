# train_bdh_paper.py
# Training script for the BDH-GPU model.

import argparse
import torch
import torch.nn as nn
from torch import optim
from bdh_paper import BDH_GPU # Import the model

# --- Utility Functions ---

def load_text_and_vocab(path: str):
    """Reads a text file, creates character-level vocabulary, and returns data as IDs."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in stoi.items()}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos

def get_batch(data, block_size, batch_size, device):
    """Generates a random batch of data (x) and targets (y)."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

# --- Model Wrapper ---

class BDHLanguageModel(nn.Module):
    """
    A wrapper for the BDH_GPU core that adds a trainable linear head
    to predict next-token logits from the core's output vector `v`.
    """
    def __init__(self, core: BDH_GPU):
        super().__init__()
        self.core = core
        # The head maps the latent dimension `d` to the vocabulary size `V`
        self.head = nn.Linear(core.d, core.V)

    def forward(self, idx: torch.Tensor):
        """
        Processes a sequence through the core and then the prediction head.

        Args:
            idx (torch.Tensor): Input token sequence of shape (B, T).

        Returns:
            torch.Tensor: Logits for next-token prediction of shape (B, T, V).
        """
        # v_out has shape (B, T, d)
        v_out = self.core(idx)
        # logits has shape (B, T, V)
        logits = self.head(v_out)
        return logits

# --- Training Logic ---

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, stoi, itos = load_text_and_vocab(args.file)
    vocab_size = len(stoi)
    print(f"Loaded text with {len(data)} characters, vocabulary size {vocab_size}.")

    # --- Model Initialization ---
    core_model = BDH_GPU(n=args.n, d=args.d, V=vocab_size)
    model = BDHLanguageModel(core_model).to(device)

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params/1e6:.2f}M trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training...")
    for step in range(1, args.steps + 1):
        # Get a batch of data
        x, y = get_batch(data, args.block_size, args.batch_size, device)

        # Forward pass
        logits = model(x)

        # Calculate loss
        # Reshape for CrossEntropyLoss: (B, T, V) -> (B*T, V) and (B, T) -> (B*T)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        if step % args.log_interval == 0:
            print(f"Step [{step}/{args.steps}], Loss: {loss.item():.4f}")

    # --- Save the trained model ---
    print("Training complete. Saving model to bdh_paper_model.pt")
    torch.save(model.state_dict(), "bdh_paper_model.pt")

# --- CLI Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BDH-GPU language model.")
    parser.add_argument("--file", type=str, required=True, help="Path to the training text file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length (context size).")
    parser.add_argument("--n", type=int, default=2048, help="Neuronal dimension.")
    parser.add_argument("--d", type=int, default=128, help="Latent (embedding) dimension.")

    args = parser.parse_args()
    train(args)
