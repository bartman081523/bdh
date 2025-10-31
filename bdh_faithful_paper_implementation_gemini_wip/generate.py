# generate.py (VERSION 18 - Final & Lauffähig)
# Angepasst an die finale, korrekte Architektur.

import argparse
import torch
import torch.nn.functional as F
from train import BDHLanguageModel # Hauptmodell-Klasse ist jetzt in train.py
from bdh_paper import BDH_GPU     # Kern-Klasse

@torch.no_grad()
def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading checkpoint from bdh_model_checkpoint.pt...")
    checkpoint = torch.load("bdh_model_checkpoint.pt", map_location=device)

    stoi, itos, config = checkpoint['stoi'], checkpoint['itos'], checkpoint['config']

    core_model = BDH_GPU(n=config['n'], d=config['d'], V=config['V'],
                         u_decay=config['u_decay'], x_decay=config['x_decay'])
    model = BDHLanguageModel(core_model).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model and vocabulary loaded successfully.")

    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=device)
    prompt_len = prompt_ids.shape[1]

    print("Warming up model state with prompt...")
    x_state = torch.zeros(1, model.core.n, device=device)
    rho_state = torch.zeros(1, model.core.d, model.core.n, device=device)
    if prompt_len > 1:
        _, x_state, rho_state = model(prompt_ids[:, :-1], x_state, rho_state)

    current_token_idx = prompt_ids[:, -1]

    generated_text = args.prompt
    print(f"--- Starting generation from prompt: '{args.prompt}' ---")
    print(generated_text, end='', flush=True)

    for i in range(args.length):
        position = prompt_len + i

        token_emb = model.core.token_emb(current_token_idx)
        rotated_emb = model.rope(token_emb.unsqueeze(1), seq_len=1, offset=position)

        v_out, x_state, rho_state = model.core.step(rotated_emb.squeeze(1), x_state, rho_state)
        logits = model.head(v_out)

        probs = F.softmax(logits / args.temperature, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)

        next_char = itos.get(next_token_idx.item(), '?')
        generated_text += next_char
        print(next_char, end='', flush=True)

        current_token_idx = next_token_idx

    print("\n\n--- Generation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained BDH-GPU model.")
    parser.add_argument("--prompt", type=str, default="Hello, ", help="Starting prompt.")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")

    args = parser.parse_args()
    generate(args)
