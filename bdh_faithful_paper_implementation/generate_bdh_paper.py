# generate_bdh_paper.py
# Textgenerierungs-Skript mit korrekter Logik zum Laden und Aufwärmen des Modells.

import argparse
import torch
import torch.nn.functional as F
from bdh_paper import BDH_GPU
from train_with_debug import BDHLanguageModel # Importiere aus dem neuen Trainingsskript

def load_vocab_from_file(path: str):
    """Lädt nur die Vokabular-Mappings aus einer Textdatei."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in stoi.items()}
    return stoi, itos

@torch.no_grad()
def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stoi, itos = load_vocab_from_file(args.file)
    vocab_size = len(stoi)

    core_model = BDH_GPU(n=args.n, d=args.d, V=vocab_size)
    model = BDHLanguageModel(core_model).to(device)

    print("Loading trained model parameters from bdh_paper_model.pt...")
    model.load_state_dict(torch.load("bdh_paper_model.pt", map_location=device), strict=False)
    model.eval()

    # --- Generierungsprozess ---
    prompt_ids = [stoi.get(c, 0) for c in args.prompt]

    # 1. Zustand mit dem Prompt initialisieren (aufwärmen) - SAUBERE VERSION
    print("Warming up model state with prompt...")
    model.core.reset_state(1, device) # Starte mit einem leeren Zustand für die Generierung (batch_size=1)

    # Wir verarbeiten den Prompt Token für Token, um den Zustand `rho` aufzubauen.
    # Der letzte Token wird nicht verarbeitet, da er der erste Input für die Generierungsschleife ist.
    for token_id in prompt_ids[:-1]:
        idx_tensor = torch.tensor([[token_id]], device=device)
        _ = model.core.step(idx_tensor.squeeze(0))

    # Der letzte Token des Prompts ist der erste Input für die eigentliche Generierung
    current_token_idx = torch.tensor([[prompt_ids[-1]]], device=device)

    generated_text = args.prompt
    print(f"--- Starting generation from prompt: '{args.prompt}' ---")

    for _ in range(args.length):
        # Führe einen einzigen Schritt aus, um den nächsten Token vorherzusagen
        v_out = model.core.step(current_token_idx.squeeze(0))

        # Prüfe auf Instabilität
        if torch.isnan(v_out).any():
            print("\n!!! NaN detected in model state. Stopping generation. !!!")
            break

        logits = model.head(v_out)

        logits = logits / args.temperature
        probs = F.softmax(logits, dim=-1)

        next_token_idx = torch.multinomial(probs, num_samples=1)

        next_char = itos.get(next_token_idx.item(), '?')
        generated_text += next_char

        # Der generierte Token wird zum Input für den nächsten Schritt
        current_token_idx = next_token_idx

    print(generated_text)
    print("\n--- Generation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained BDH-GPU model.")
    parser.add_argument("--file", type=str, required=True, help="Path to the original training text file (for vocabulary).")
    parser.add_argument("--prompt", type=str, default="Hello, my name is ", help="The starting prompt for generation.")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher is more random).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--n", type=int, default=2048, help="Neuronal dimension (must match trained model).")
    parser.add_argument("--d", type=int, default=128, help="Latent dimension (must match trained model).")

    args = parser.parse_args()
    generate(args)
