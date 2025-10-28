# generate_bdh_paper.py
# Text generation script for the trained BDH-GPU model.

import argparse
import torch
import torch.nn.functional as F
from bdh_paper import BDH_GPU
from train_bdh_paper import BDHLanguageModel # Reuse the wrapper and vocab loader

def load_vocab_from_file(path: str):
    """Loads only the vocabulary mappings from a text file."""
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

    # --- Model Initialization and Loading ---
    core_model = BDH_GPU(n=args.n, d=args.d, V=vocab_size)
    model = BDHLanguageModel(core_model).to(device)

    print("Loading trained model from bdh_paper_model.pt...")
    model.load_state_dict(torch.load("bdh_paper_model.pt", map_location=device))
    model.eval()

    # --- Generation Process ---
    prompt_ids = torch.tensor([stoi.get(c, 0) for c in args.prompt], dtype=torch.long, device=device).unsqueeze(0)

    # Warm up the model's state with the prompt
    print("Warming up model state with prompt...")
    # We run the full forward pass on the prompt to set the state correctly.
    # The output logits are discarded.
    _ = model(prompt_ids)

    # The last token of the prompt is the first input for generation
    current_token_idx = prompt_ids[:, -1]

    generated_text = args.prompt
    print(f"--- Starting generation from prompt: '{args.prompt}' ---")

    for _ in range(args.length):
        # Get the output vector `v` for the current token
        # model.core.v is the state from the last `step` call inside `forward`
        v_out = model.core.v

        # Get logits from the head
        logits = model.head(v_out)

        # Apply temperature and get probabilities
        logits = logits / args.temperature
        probs = F.softmax(logits, dim=-1)

        # Sample the next token
        next_token_idx = torch.multinomial(probs, num_samples=1).squeeze(0)

        # Append the character to our sequence
        next_char = itos.get(next_token_idx.item(), '?')
        generated_text += next_char

        # Update the model's state with the newly generated token
        # We call `step` directly to advance the recurrence by one token
        _ = model.core.step(next_token_idx)

        # The new token becomes the current token for the next iteration
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
