# train.py
# Final comparative version based on our working script.
# Can train either BDH or GPT models.

import os
from contextlib import nullcontext
import torch
import numpy as np
import requests
import argparse # For command-line arguments

# Import both model architectures
import bdh
import gpt

# --- Configuration Section ---
# Default parameters, can be overridden if needed
BLOCK_SIZE = 512
BATCH_SIZE = 8
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
CHECKPOINT_FREQ = 1000

# Compilation settings from your version
USE_COMPILE = True
COMPILE_MODE = "default" # This isn't used, but we keep it for consistency

# --- Device and Dtype Setup (Your version) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (torch.amp.autocast(device_type=device.type, dtype=ptdtype) if "cuda" in str(device) else nullcontext())
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

# --- Performance Optimizations (Your version) ---
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype: {dtype}")

# --- Data Loading Section (Your version) ---
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
def fetch_data():
    if not os.path.exists(input_file_path):
        print("Downloading Tiny Shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)
        print("Dataset downloaded.")
def get_batch(split):
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    split_idx = int(0.9 * len(data))
    data = data[:split_idx] if split == "train" else data[split_idx:]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

# --- Main Execution ---
if __name__ == "__main__":
    # --- NEW: Argument Parser ---
    parser = argparse.ArgumentParser(description="Train either a BDH or a GPT model.")
    parser.add_argument('--model_type', type=str, default='bdh', choices=['bdh', 'gpt'],
                        help='The type of model to train (bdh or gpt).')
    args = parser.parse_args()
    print(f"\nSelected model type for training: {args.model_type.upper()}")
    
    fetch_data()

    # --- NEW: Model selection logic ---
    if args.model_type == 'bdh':
        model_config = bdh.BDHConfig() # Default config from bdh.py
        model = bdh.BDH(model_config).to(device)
        MODEL_CHECKPOINT_PATH = "bdh_shakespeare_checkpoint.pth"
        FINAL_MODEL_PATH = "bdh_shakespeare_final.pth"
    elif args.model_type == 'gpt':
        # GPT config to match BDH's ~25M parameters
        model_config = gpt.GPTConfig(n_layer=8, n_head=8, n_embd=512, block_size=BLOCK_SIZE)
        model = gpt.GPT(model_config).to(device)
        MODEL_CHECKPOINT_PATH = "gpt_shakespeare_checkpoint.pth"
        FINAL_MODEL_PATH = "gpt_shakespeare_final.pth"

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Your compilation block, unchanged
    if USE_COMPILE:
        print(f"Compiling the model...")
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, backend="aot_eager")
            print("Model compiled successfully with 'aot_eager' backend.")
        except Exception as e:
            print(f"Warning: torch.compile failed with error: {e}\nContinuing without compilation...")
    else:
        print("Compilation disabled, running in eager mode.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Your training loop, unchanged
    print(f"\nStarting training for {MAX_ITERS} iterations...")
    # ... (cała pętla treningowa, checkpointing, generowanie i zapis - bez zmian)
    loss_acc = 0.0
    loss_steps = 0
    for step in range(MAX_ITERS):
        x, y = get_batch("train")
        with ctx:
            _, loss = model(x, y)
        loss_acc += loss.item()
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if step > 0 and step % LOG_FREQ == 0:
            avg_loss = loss_acc / loss_steps
            print(f"Step: {step}/{MAX_ITERS} | loss: {avg_loss:.4f}")
            loss_acc = 0.0
            loss_steps = 0
        if step > 0 and step % CHECKPOINT_FREQ == 0:
            print(f"\n--- Saving checkpoint at step {step} ---")
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            print(f"Model checkpoint saved to {MODEL_CHECKPOINT_PATH}")
            print("-----------------------------------------")

    print("\nTraining finished. Generating a sample...")
    model.eval()
    prompt = torch.tensor(bytearray("To be or not to be", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with ctx:
            ret = model.generate(prompt, max_new_tokens=200, top_k=5)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
    print("-" * 50)
    print(ret_decoded)
    print("-" * 50)

    print(f"\nSaving final model to {FINAL_MODEL_PATH}...")
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print("Final model saved successfully.")