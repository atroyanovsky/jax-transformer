import jax
import jax.numpy as jnp
import optax
import requests
import os
from transformers import GPT2Tokenizer

# ==========================================
# 1. PREPARE DATA (Tiny Shakespeare)
# ==========================================

# Download the dataset
file_path = "tiny_shakespeare.txt"
if not os.path.exists(file_path):
    print("Downloading Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(file_path, "w") as f:
        f.write(requests.get(url).text)

# Read text
with open(file_path, "r") as f:
    text = f.read()

print(f"Dataset length: {len(text)} characters")

# Tokenize the ENTIRE dataset once
# This turns the 1MB file into ~300k integers
print("Tokenizing... (This might take 10 seconds)")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# We use encode_plus or just encode. 'text' is one giant string.
full_dataset = jnp.array(tokenizer.encode(text))
print(f"Total Tokens: {len(full_dataset)}")

# ==========================================
# 2. CONFIGURATION
# ==========================================

VOCAB_SIZE = len(tokenizer) # ~50257
D_MODEL = 256               # Bigger model for harder task
NUM_LAYERS = 4              # Deeper
NUM_HEADS = 4
D_FF = 512
MAX_LEN = 64                # Longer context window
BATCH_SIZE = 32
STEPS = 2000                # Train longer
LR = 3e-4

# ==========================================
# 3. INITIALIZATION
# ==========================================

print("Initializing Model...")
key = jax.random.PRNGKey(1337)
model = Transformer(NUM_HEADS, MAX_LEN, D_MODEL, VOCAB_SIZE, NUM_LAYERS, D_FF)

key, subkey = jax.random.split(key)
params = model.init_params(subkey)

optimizer = optax.adamw(learning_rate=LR)
opt_state = optimizer.init(params)

