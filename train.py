import os
import time
import pickle
import jax
import jax.numpy as jnp
import optax
from functools import partial

# Local Imports
from src.data.bpe_loader import BPELoader
from src.models.transformer import Transformer

# --- 1. HELPER FUNCTIONS ---
# This should print something like: [MetalDevice(id=0, ...)]
print(jax.devices())

def loss_fn(params, model, inputs, targets, key):
    """
    Calculates the Cross-Entropy Loss for next-token prediction.
    """
    # 1. Forward Pass (Returns Logits)
    logits = model.forward(params, inputs, key, training=True)
    
    # 2. Log Softmax (Numerical stability)
    log_probs = jax.nn.log_softmax(logits)
    
    # 3. Gather probabilities of the true target tokens
    # targets shape: (batch, seq) -> (batch, seq, 1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    
    # 4. Negative Log Likelihood
    return -jnp.mean(target_log_probs)

def save_checkpoint(filename, params, opt_state, config, step, loss):
    """Saves model and optimizer state to CPU/Disk."""
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
    """Loads checkpoint if it exists."""
    if not os.path.exists(filename):
        return None
    print(f"Loading checkpoint from {filename}...")
    with open(filename, 'rb') as f:
        return pickle.load(f)

# --- 2. JIT COMPILED UPDATE STEP ---

# We freeze 'model' as a static argument so JAX doesn't trace the class instance
@partial(jax.jit, static_argnames=['model'])
def train_step(params, opt_state, model, inputs, targets, key):
    
    # 1. Calculate Loss & Gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, model, inputs, targets, key)
    
    # 2. Compute Updates (AdamW)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    
    # 3. Apply Updates
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, key

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    # --- A. SETUP ---
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "shakespeare_ckpt.pkl")

    # Initialize Tokenizer & Data Loader
    # Ensure you have run your tokenizer training script first!
    print("Initializing Data Loader...")
    loader = BPELoader(
        tokenizer_path="data/shakespeare_tokenizer.json",
        data_path="data/tiny_shakespeare.txt",
        batch_size=32,
        seq_len=64
    )

    # Model Configuration
    config = {
        "num_heads": 4,
        "max_len": 128,      # Context window
        "d_model": 128,      # Embedding dimension
        "vocab_size": loader.vocab_size,
        "num_layers": 4,
        "d_ff": 512,          # Feed-forward hidden size
        "dropout_rate": 0.1
    }

    # Initialize Model
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    model = Transformer(**config)
    
    # Initialize Optimizer
    learning_rate = 3e-4
    optimizer = optax.adamw(learning_rate)

    # --- B. INITIALIZATION / RESUME ---
    
    # Default initialization
    params = model.init_params(init_key)
    opt_state = optimizer.init(params)
    start_step = 0
    
    # Try to load checkpoint
    loaded_data = load_checkpoint(ckpt_path)
    if loaded_data:
        print("♻️ Resuming training...")
        params = loaded_data['params']
        opt_state = loaded_data['opt_state']
        start_step = loaded_data['step'] + 1
        # Optional: Validate config matches loaded_data['config']
    else:
        print("🌱 Starting training from scratch...")

    # --- C. TRAINING LOOP ---
    
    num_steps = 10000
    log_every = 100
    eval_every = 500
    
    print(f"Training for {num_steps - start_step} more steps...")
    start_time = time.time()
    
    for step in range(start_step, num_steps):
        
        # 1. Get Data
        inputs, targets = loader.get_batch('train')
        
        # 2. Train Step
        key, step_key = jax.random.split(key)
        params, opt_state, loss_val, _ = train_step(params, opt_state, model, inputs, targets, step_key)
        
        # 3. Logging
        if step % log_every == 0:
            elapsed = time.time() - start_time
            print(f"Step {step} | Loss: {loss_val:.4f} | Time: {elapsed:.2f}s")
            start_time = time.time()
            
        # 4. Evaluation & Checkpointing
        if step % eval_every == 0:
            print("\n--- GENERATING TEXT ---")
            
            # Create a prompt (e.g., "The")
            prompt_str = "The"
            context = jnp.array([loader.encode(prompt_str)])
            
            # Generate! (Pass a key if your generate method supports sampling)
            # Assuming your generate method signature is: generate(params, inputs, max_new_tokens, ...)
            # If you implemented temperature sampling, you can pass temperature=0.8 here.
            output_ids = model.generate(params, context, max_new_tokens=50)
            
            decoded = loader.decode(output_ids[0].tolist())
            print(f"Prompt: {prompt_str}")
            print(f"Result: {decoded}")
            print("-----------------------\n")
            
            # Save Checkpoint
            save_checkpoint(ckpt_path, params, opt_state, config, step, float(loss_val))

    print("Training Complete!")