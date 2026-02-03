import os
import time
import pickle
import jax
import jax.numpy as jnp
import optax
from functools import partial

# Local Imports
# Ensure your directory structure is correct:
# src/
#   data/bpe_loader.py
#   models/transformer.py
from src.data.bpe_loader import BPELoader
from src.models.transformer import Transformer

# --- 1. CONFIGURATION ---
# "Karpathy" Config (~10M Params)
# This is significantly larger than the previous model.
config = {
    "num_heads": 6,
    "max_len": 256,          # Increased context window (was 128)
    "d_model": 384,          # Triple the width (was 128)
    "vocab_size": 0,         # Will be set by loader
    "num_layers": 6,         # Deeper (was 4)
    "d_ff": 384 * 4,         # Standard 4x expansion (1536)
    "dropout_rate": 0.1
}

TRAIN_CONFIG = {
    "num_steps": 20000,      # A larger model needs fewer steps to beat the small one, 
                             # but 50k+ is ideal for convergence. Start with 20k.
    "batch_size": 32,
    "learning_rate": 3e-4,
    "log_every": 100,
    "eval_every": 1000,
    "ckpt_name": "shakespeare_big_ckpt.pkl" # New filename to avoid shape errors!
}

# --- 2. HELPER FUNCTIONS ---
print(f"Devices found: {jax.devices()}")

def loss_fn(params, model, inputs, targets, key):
    """
    Calculates the Cross-Entropy Loss for next-token prediction.
    """
    # 1. Forward Pass
    logits = model.forward(params, inputs, key, training=True)
    
    # 2. Log Softmax
    log_probs = jax.nn.log_softmax(logits)
    
    # 3. Gather probabilities of the true target tokens
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    
    # 4. Negative Log Likelihood
    return -jnp.mean(target_log_probs)

def save_checkpoint(filename, params, opt_state, config, step, loss):
    print(f"Saving checkpoint to {filename}...")
    checkpoint_data = {
        'params': jax.device_get(params),
        'opt_state': jax.device_get(opt_state),
        'config': config,
        'step': step,
        'loss': loss
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print("✅ Saved.")

def load_checkpoint(filename):
    if not os.path.exists(filename):
        return None
    print(f"Loading checkpoint from {filename}...")
    with open(filename, 'rb') as f:
        return pickle.load(f)

# --- 3. JIT COMPILED UPDATE STEP ---
@partial(jax.jit, static_argnames=['model'])
def train_step(params, opt_state, model, inputs, targets, key):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, inputs, targets, key)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, key

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # --- A. SETUP ---
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, TRAIN_CONFIG["ckpt_name"])

    print("Initializing Data Loader...")
    loader = BPELoader(
        tokenizer_path="data/shakespeare_tokenizer.json",
        data_path="data/tiny_shakespeare.txt",
        batch_size=TRAIN_CONFIG["batch_size"],
        seq_len=config["max_len"] # Use config max_len
    )
    
    # Set vocab size dynamically
    config["vocab_size"] = loader.vocab_size

    # Initialize Model
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    print(f"🚀 Initializing Model: d_model={config['d_model']}, layers={config['num_layers']}")
    model = Transformer(**config)
    
    # Initialize Scheduler & Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=TRAIN_CONFIG["learning_rate"],
        warmup_steps=1000, 
        decay_steps=TRAIN_CONFIG["num_steps"],
        end_value=TRAIN_CONFIG["learning_rate"] / 10
    )
    
    # Gradient Clipping is crucial for deeper models to prevent instability
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-2)
    )

    # --- B. INITIALIZATION / RESUME ---
    params = model.init_params(init_key)
    opt_state = optimizer.init(params)
    start_step = 0
    
    # Try to load checkpoint
    loaded_data = load_checkpoint(ckpt_path)
    if loaded_data:
        # Safety check: Ensure dimensions match before loading
        if loaded_data['config']['d_model'] == config['d_model']:
            print("♻️ Resuming training...")
            params = loaded_data['params']
            opt_state = loaded_data['opt_state']
            start_step = loaded_data['step'] + 1
        else:
            print("⚠️ Checkpoint found but config mismatch (Different d_model). Starting fresh.")
    else:
        print("🌱 Starting training from scratch...")

    # --- C. TRAINING LOOP ---
    print(f"Training for {TRAIN_CONFIG['num_steps'] - start_step} more steps...")
    start_time = time.time()
    
    # Master key for dropout
    dropout_key = jax.random.PRNGKey(99)

    for step in range(start_step, TRAIN_CONFIG["num_steps"]):
        
        # 1. Get Data
        inputs, targets = loader.get_batch('train')
        
        # 2. Train Step
        dropout_key, step_key = jax.random.split(dropout_key)
        params, opt_state, loss_val, _ = train_step(params, opt_state, model, inputs, targets, step_key)
        
        # 3. Logging
        if step % TRAIN_CONFIG["log_every"] == 0:
            elapsed = time.time() - start_time
            # Calculate tokens per second
            tokens_per_sec = (TRAIN_CONFIG["batch_size"] * config["max_len"] * TRAIN_CONFIG["log_every"]) / elapsed
            print(f"Step {step} | Loss: {loss_val:.4f} | Speed: {tokens_per_sec:.0f} tok/s")
            start_time = time.time()
            
        # 4. Evaluation
        if step % TRAIN_CONFIG["eval_every"] == 0:
            print("\n--- GENERATING TEXT ---")
            prompt_str = "The"
            context = jnp.array([loader.encode(prompt_str)])
            
            # Use a subkey for generation randomness
            key, gen_key = jax.random.split(key)
            
            output_ids = model.generate(
                params, 
                context, 
                max_new_tokens=100, 
                key=gen_key,
                top_k=40
            )
            
            decoded = loader.decode(output_ids[0].tolist())
            print(f"Result: {decoded}")
            print("-----------------------\n")
            
            save_checkpoint(ckpt_path, params, opt_state, config, step, float(loss_val))

    print("Training Complete!")