# generate.py
# Finales, stabiles Generierungs-Skript. Lädt den korrekten Checkpoint.

import argparse
import torch
import torch.nn.functional as F
from bdh_paper import BDH_GPU
from train import BDHLanguageModel

@torch.no_grad()
def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading checkpoint from bdh_model_checkpoint.pt...")
    # weights_only=False ist sicher, da wir die Checkpoint-Datei selbst erstellen
    checkpoint = torch.load("bdh_model_checkpoint.pt", map_location=device, weights_only=False)

    stoi, itos, config = checkpoint['stoi'], checkpoint['itos'], checkpoint['config']

    core_model = BDH_GPU(n=config['n'], d=config['d'], V=config['V'],
                         u_decay=config['u_decay'], x_decay=config['x_decay'])
    model = BDHLanguageModel(core_model).to(device)

    # --- KORREKTE LADELOGIK ---
    # Lade die Parameter. strict=False erlaubt es PyTorch, die fehlenden Buffer zu ignorieren.
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("Model and vocabulary loaded successfully.")

    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=device)

    print("Warming up model state with prompt...")
    model.core.reset_state(1, device)
    if prompt_ids.shape[1] > 1:
        _ = model(prompt_ids[:, :-1])

    current_token_idx = prompt_ids[:, -1]

    generated_text = args.prompt
    print(f"--- Starting generation from prompt: '{args.prompt}' ---")
    print(generated_text, end='', flush=True)

    for i in range(args.length):
        v_out = model.core.step(current_token_idx)
        logits = model.head(v_out)

        if args.debug:
            print(f"\n--- Step {i} ---")
            top_p, top_i = torch.topk(F.softmax(logits, dim=-1), 5)
            for prob, idx in zip(top_p.squeeze().tolist(), top_i.squeeze().tolist()):
                print(f"  '{itos.get(idx, 'UNK')}' (ID: {idx}) - Prob: {prob:.4f}")

        probs = F.softmax(logits / args.temperature, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)

        next_char = itos.get(next_token_idx.item(), '?')
        generated_text += next_char

        if not args.debug:
            print(next_char, end='', flush=True)

        current_token_idx = next_token_idx.squeeze(0)

    print("\n\n--- Generation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained BDH-GPU model.")
    # ... (alle Argumente bleiben gleich)
    parser.add_argument("--prompt", type=str, default="Hello, my name is ", help="Starting prompt.")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--debug", action='store_true', help="Enable detailed step-by-step prediction logging.")
    args = parser.parse_args()
    generate(args)
