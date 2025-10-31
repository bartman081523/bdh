# testbench.py
# Ein in sich geschlossenes Skript, um die Memorisierungsfähigkeit
# des 0.8M BDH-Modells auf einem minimalen Datensatz zu testen.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os

# ==============================================================================
# 1. MODELLDEFINITION (aus bdh_paper.py kopiert)
# ==============================================================================
def relu(z: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(z, 0.0)

def layernorm_row(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = z.mean(dim=-1, keepdim=True)
    s = z.std(dim=-1, keepdim=True)
    return (z - m) / (s + eps)

class BDH_GPU(nn.Module):
    def __init__(self, n: int, d: int, V: int, u_decay: float = 0.97, x_decay: float = 0.97):
        super().__init__()
        self.n, self.d, self.V = n, d, V
        self.u_decay, self.x_decay = u_decay, x_decay

        self.token_emb = nn.Parameter(torch.randn(V, d) / (d**0.5))
        self.E = nn.Parameter(torch.randn(d, n) / (n**0.5))
        self.Dx = nn.Parameter(torch.randn(n, d) / (d**0.5))
        self.Dy = nn.Parameter(torch.randn(n, d) / (d**0.5))

        self.register_buffer("x", torch.zeros(1, n))
        self.register_buffer("y", torch.zeros(1, n))
        self.register_buffer("v", torch.zeros(1, d))
        self.register_buffer("rho", torch.zeros(1, d, n))

    def reset_state(self, batch_size: int, device: torch.device):
        self.x = torch.zeros(batch_size, self.n, device=device)
        self.y = torch.zeros(batch_size, self.n, device=device)
        self.v = torch.zeros(batch_size, self.d, device=device)
        self.rho = torch.zeros(batch_size, self.d, self.n, device=device)

    def step(self, token_idx: torch.Tensor) -> torch.Tensor:
        v_prev = self.token_emb[token_idx]
        x_update = relu(v_prev @ self.Dx.T)
        x_t = self.x_decay * self.x + x_update
        x_t = x_t / (x_t.norm(p=1, dim=-1, keepdim=True) + 1e-6)
        a_star = torch.einsum('bdn,bn->bd', self.rho, x_t)
        y_core = layernorm_row(a_star) @ self.Dy.T
        y_t = relu(y_core) * relu(x_t)
        v_star = layernorm_row(y_t @ self.E.T)
        v_prev_normed = layernorm_row(v_prev)
        increment = torch.einsum('bd,bn->bdn', v_prev_normed, x_t)
        self.rho = self.u_decay * (self.rho + increment)
        self.x, self.y, self.v = x_t, y_t, v_star
        return v_star

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.rho = self.rho.detach()
        self.x = self.x.detach()
        self.y = self.y.detach()
        self.v = self.v.detach()
        outputs = []
        for t in range(idx.shape[1]):
            v_out = self.step(idx[:, t])
            outputs.append(v_out.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# ==============================================================================
# 2. HELFERFUNKTIONEN (aus train.py kopiert)
# ==============================================================================
def load_text_and_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f: text = f.read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos

def get_sequential_batch(data, block_size, batch_size, step, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

class BDHLanguageModel(nn.Module):
    def __init__(self, core: BDH_GPU):
        super().__init__()
        self.core, self.head = core, nn.Linear(core.d, core.V)
    def forward(self, idx: torch.Tensor):
        return self.head(self.core(idx))

# ==============================================================================
# 3. DIE TESTBENCH-FUNKTION
# ==============================================================================
def run_memorization_testbench():
    # --- Konfiguration ---
    TEST_TEXT_FILE = "test_memorize.txt"
    SENTENCE = "The quick brown fox jumps over the lazy dog. "

    # Modellparameter (0.8M Modell)
    N = 2048
    D = 128

    # Trainingsparameter
    EPOCHS = 1
    BATCH_SIZE = 16
    BLOCK_SIZE = 64 # Länger als der Satz
    LR = 3e-4

    # Verifikationsparameter
    PROMPT = "The quick brown fox "
    GROUND_TRUTH_CONTINUATION = "jumps over the lazy dog. "

    # --- 1. Testdaten erstellen ---
    print("="*60)
    print(" PHASE 1: ERSTELLE TESTDATEN")
    print("="*60)
    print(f"Erstelle '{TEST_TEXT_FILE}' mit dem Satz: '{SENTENCE}'")
    with open(TEST_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(SENTENCE * 2)
    print("Testdaten erstellt.\n")

    # --- 2. Trainingsphase ---
    print("="*60)
    print(f" PHASE 2: TRAINING (Modell: {N=}, {D=}, {EPOCHS=})")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, stoi, itos = load_text_and_vocab(TEST_TEXT_FILE)
    vocab_size = len(stoi)

    core_model = BDH_GPU(n=N, d=D, V=vocab_size)
    model = BDHLanguageModel(core_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        model.core.reset_state(BATCH_SIZE, device)
        # Wir machen nur wenige Schritte pro Epoche, da der Text so kurz ist
        for step in range(50):
            x, y = get_sequential_batch(data, BLOCK_SIZE, BATCH_SIZE, step, device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} | Final Loss: {loss.item():.4f}")
    print("Training abgeschlossen.\n")

    # --- 3. Verifikationsphase ---
    print("="*60)
    print(" PHASE 3: VERIFIKATION DER MEMORISIERUNG")
    print("="*60)

    model.eval()

    prompt_ids = torch.tensor([[stoi[c] for c in PROMPT]], dtype=torch.long, device=device)

    # Modellzustand mit dem Prompt aufwärmen
    model.core.reset_state(1, device)
    if prompt_ids.shape[1] > 1:
        _ = model(prompt_ids[:, :-1])

    current_token_idx = prompt_ids[:, -1]

    print(f"Prompt: '{PROMPT}'")
    print(f"Erwartete Fortsetzung: '{GROUND_TRUTH_CONTINUATION}'\n")

    generated_text = ""
    correct_predictions = 0

    with torch.no_grad():
        for i, expected_char in enumerate(GROUND_TRUTH_CONTINUATION):
            v_out = model.core.step(current_token_idx)
            logits = model.head(v_out)
            probs = F.softmax(logits, dim=-1)

            # Greedy decoding (temperature=0)
            next_token_idx = torch.argmax(probs, dim=-1)
            predicted_char = itos.get(next_token_idx.item())

            # --- Aufschlussreicher Debug-Output ---
            status = "✅ MATCH" if predicted_char == expected_char else "❌ MISMATCH"
            if predicted_char == expected_char:
                correct_predictions += 1

            confidence_in_prediction = probs[0, next_token_idx.item()].item()

            expected_idx = stoi.get(expected_char)
            confidence_in_correct = probs[0, expected_idx].item() if expected_idx is not None else 0

            print(f"Step {i+1:02d}: Prompt='...{PROMPT[-5:]+generated_text[-5:]}'")
            print(f"  - Vorhergesagt: '{predicted_char}' (Konfidenz: {confidence_in_prediction:.2%})")
            print(f"  - Erwartet:     '{expected_char}' (Modell-Konfidenz dafür: {confidence_in_correct:.2%})")
            print(f"  - Status:       {status}\n")

            generated_text += predicted_char
            current_token_idx = next_token_idx

    # --- 4. Endergebnis ---
    print("="*60)
    print(" PHASE 4: ERGEBNIS")
    print("="*60)
    accuracy = correct_predictions / len(GROUND_TRUTH_CONTINUATION)
    print(f"Vollständiger generierter Text: '{PROMPT}{generated_text}'")
    print(f"\nMemorisierungs-Genauigkeit: {correct_predictions}/{len(GROUND_TRUTH_CONTINUATION)} Zeichen ({accuracy:.2%})")

    if accuracy > 0.9:
        print("\nFazit: ✅ Erfolg! Das Modell hat die Sequenz erfolgreich memoriert.")
        print("Dies bestätigt, dass die Architektur fähig ist zu lernen, aber für komplexe Aufgaben mehr Kapazität (größere N, D) benötigt.")
    else:
        print("\nFazit: ❌ Fehlgeschlagen! Das Modell konnte die einfache Sequenz nicht memorieren.")
        print("Dies deutet darauf hin, dass selbst für diese simple Aufgabe die Kapazität des 0.8M-Modells an ihre Grenzen stößt oder mehr Training benötigt wird.")

    # --- Aufräumen ---
    os.remove(TEST_TEXT_FILE)


if __name__ == "__main__":
    run_memorization_testbench()
