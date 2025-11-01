# generate.py
# VERSION 15: Angepasst, um das neue, stateful bdh_paper-Modell korrekt zu laden und zu verwenden.

import torch, argparse, torch.nn as nn
from contextlib import nullcontext
import bdh, gpt, bdh_paper

# --- ANGEPASST: Wrapper-Klasse für bdh_paper, muss identisch zur train.py sein ---
class BDHLanguageModel_Paper(nn.Module):
    def __init__(self, core_config, vocab_size):
        super().__init__()
        self.config = core_config
        self.core = bdh_paper.BDH_GPU(n=core_config['n'], d=core_config['d'], V=vocab_size)
        self.head = nn.Linear(core_config['d'], vocab_size, bias=False)
        self.rope = bdh_paper.RotaryEmbedding(core_config['d'])

    def forward(self, idx, targets=None):
        token_embeddings = self.core.token_emb(idx)
        rotated_embeddings = self.rope(token_embeddings, seq_len=idx.shape[1])
        v_out = self.core(rotated_embeddings)
        logits = self.head(v_out)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=1):
        self.eval()
        B, T = idx.shape
        self.core.reset_state(B, idx.device)
        if T > 1:
            prompt_emb = self.core.token_emb(idx[:, :-1])
            rotated_prompt_emb = self.rope(prompt_emb, seq_len=T-1)
            self.core(rotated_prompt_emb)
        current_token_idx = idx[:, -1]
        for _ in range(max_new_tokens):
            token_emb = self.core.token_emb(current_token_idx)
            rotated_emb = self.rope(token_emb.unsqueeze(1), seq_len=1, offset=idx.shape[1]-1)
            v_out, x_state, rho_state = self.core.step(rotated_emb.squeeze(1), self.core.x_state, self.core.rho_state)
            self.core.x_state, self.core.rho_state = x_state, rho_state
            logits = self.head(v_out)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            current_token_idx = idx_next.squeeze(1)
        self.train()
        return idx

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint from '{args.checkpoint_path}'...")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']; stoi = checkpoint['stoi']; itos = checkpoint['itos']
    model_type = checkpoint.get('model_type', 'bdh')

    if model_type == 'bdh': model = bdh.BDH(config)
    elif model_type == 'gpt': model = gpt.GPT(config)
    elif model_type == 'bdh_paper': model = BDHLanguageModel_Paper(config, len(stoi))
    else: raise ValueError(f"Unknown model type '{model_type}' in checkpoint.")

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

    with torch.no_grad(), ctx:
        generated_ids = model.generate(idx=prompt_ids, max_new_tokens=args.length, top_k=args.top_k)
        generated_tokens = generated_ids[0, len(prompt_ids[0]):].tolist()
        generated_text = "".join([itos.get(i, '?') for i in generated_tokens])
        print(generated_text, end='', flush=True)

    print("\n\n--- Generation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained model.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--prompt", type=str, default="Hello, ")
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    generate(args)
