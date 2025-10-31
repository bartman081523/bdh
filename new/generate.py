# generate.py
# Lädt ein trainiertes Modell (BDH oder GPT) von einem Checkpoint
# und generiert Text basierend auf einem Prompt.

import torch
import argparse
from contextlib import nullcontext

# Importiere die Modell-Definitionen
import bdh
import gpt

def generate(args):
    """Hauptfunktion zur Textgenerierung."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from '{args.checkpoint_path}'...")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{args.checkpoint_path}'")
        return

    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

    # Prüfen, welcher Modelltyp im Checkpoint gespeichert ist.
    # Wir leiten dies aus dem Typ des 'config'-Objekts ab.
    if isinstance(config, bdh.BDHConfig):
        model_type = 'BDH'
        model = bdh.BDH(config)
    elif isinstance(config, gpt.GPTConfig):
        model_type = 'GPT'
        model = gpt.GPT(config)
    else:
        print("Error: Could not determine model type from the checkpoint config.")
        return

    print(f"Successfully identified model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Wichtig: Modell in den Evaluationsmodus schalten

    print("Model and vocabulary loaded successfully.")

    # Dtype und Autocast-Kontext für Performance (wie im Training)
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if "cuda" in str(device) else nullcontext()

    # Eingabe-Prompt verarbeiten
    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=device)

    print(f"\n--- Starting generation from prompt: '{args.prompt}' ---")
    print(args.prompt, end='', flush=True)

    with torch.no_grad():
        with ctx:
            # Nutze die `generate` Methode des Modells, die bereits im Modell-Code definiert ist
            generated_ids = model.generate(
                idx=prompt_ids,
                max_new_tokens=args.length,
                temperature=args.temperature,
                top_k=args.top_k
            )

            # Dekodiere nur die neu generierten Tokens
            generated_tokens = generated_ids[0, len(prompt_ids[0]):].tolist()
            generated_text = "".join([itos.get(i, '?') for i in generated_tokens])
            print(generated_text, end='', flush=True)

    print("\n\n--- Generation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained BDH or GPT model.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file (e.g., 'bdh_final_model.pt').")
    parser.add_argument("--prompt", type=str, default="What", help="Starting prompt for the generation.")
    parser.add_argument("--length", type=int, default=500, help="Number of new characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Higher values mean more randomness.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K sampling. Consider only the K most likely tokens. Set to None to disable.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    generate(args)
