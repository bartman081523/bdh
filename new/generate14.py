# generate.py
# Lädt ein trainiertes Modell (bdh, gpt, oder bdh_paper) von einem Checkpoint
# und generiert Text. Basiert auf dem V14 Trainings-Skript.

import torch
import argparse
from contextlib import nullcontext
import torch.nn as nn

# Importiere die Modell-Definitionen
import bdh
import gpt
import bdh_paper

# --- NEU: Wrapper-Klasse für bdh_paper, muss identisch zur train.py sein ---
class BDHLanguageModel_Paper(nn.Module):
    def __init__(self, core_config: dict, vocab_size: int):
        super().__init__()
        self.config = core_config
        self.core = bdh_paper.BDH_GPU(n=core_config['n'], d=core_config['d'], V=vocab_size, u_decay=core_config.get('u_decay', 0.97), x_decay=core_config.get('x_decay', 0.97))
        self.head = nn.Linear(core_config['d'], vocab_size, bias=False)
        self.rope = bdh_paper.RotaryEmbedding(core_config['d'])
        self.generate = self._generate

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        seq_len = idx.shape[1]
        token_embeddings = self.core.token_emb(idx)
        rotated_embeddings = self.rope(token_embeddings, seq_len=seq_len)
        v_out = self.core(rotated_embeddings)
        logits = self.head(v_out)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def _generate(self, idx: torch.Tensor, max_new_tokens: int, top_k: int = 1):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.get('block_size', 256):]) # Kontext abschneiden
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint from '{args.checkpoint_path}'...")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']
    stoi = checkpoint['stoi']; itos = checkpoint['itos']
    model_type = checkpoint.get('model_type', 'bdh') # Fallback für ältere Checkpoints

    if model_type == 'bdh':
        model = bdh.BDH(config)
    elif model_type == 'gpt':
        model = gpt.GPT(config)
    elif model_type == 'bdh_paper':
        model = BDHLanguageModel_Paper(config, len(stoi))
    else:
        raise ValueError(f"Unknown model type '{model_type}' in checkpoint.")

    print(f"Successfully identified model type: {model_type.upper()}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device); model.eval()
    print("Model and vocabulary loaded successfully.")

    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if "cuda" in str(device) else nullcontext()

    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=device)

    print(f"\n--- Starting generation from prompt: '{args.prompt}' ---")
    print(args.prompt, end='', flush=True)

    with torch.no_grad():
        with ctx:
            generated_ids = model.generate(idx=prompt_ids, max_new_tokens=args.length, top_k=args.top_k)
            generated_tokens = generated_ids[0, len(prompt_ids[0]):].tolist()
            generated_text = "".join([itos.get(i, '?') for i in generated_tokens])
            print(generated_text, end='', flush=True)

    print("\n\n--- Generation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained model.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file (e.g., 'checkpoints/run_123/bdh_best.pt').")
    parser.add_argument("--prompt", type=str, default="Hello, ")
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5, help="Top-K sampling. 0 to disable.")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    generate(args)
