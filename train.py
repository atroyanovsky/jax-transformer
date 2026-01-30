from src.data.bpe_loader import BPELoader
import jax
import jax.numpy as jnp
from src.models.transformer import Transformer
from src.training.training_loop import loss_fn_decoder
from functools import partial
import optax
import time

# --- MAIN TRAINING SETUP ---

# 1. Initialize Loader
# Make sure "data/shakespeare_tokenizer.json" exists (run your script first!)
loader = BPELoader(
    tokenizer_path="data/shakespeare_tokenizer.json",
    data_path="data/tiny_shakespeare.txt",
    batch_size=32,
    seq_len=64
)

# 2. Config
config = {
    "num_heads": 4,
    "max_len": 128,      # Context window
    "d_model": 128,      # Embedding dimension
    "vocab_size": loader.vocab_size, # <--- 5000
    "num_layers": 4,
    "d_ff": 512          # 4 * d_model
}

# 3. Initialize Model
key = jax.random.PRNGKey(0)
model = Transformer(**config)
params = model.init_params(key)

print("Model Initialized!")

# 4. Initialize Optimizer (Optax)
learning_rate = 3e-4 # Classic transformer LR
optimizer = optax.adamw(learning_rate)
opt_state = optimizer.init(params)
print("Optimizer Initialized!")

# 5. Define Step Function
# We use 'static_argnames' for 'model' so JAX doesn't try to trace the class instance
@partial(jax.jit, static_argnames=['model'])
def train_step(params, opt_state, model, inputs, targets):
    
    # 1. Calculate Loss & Gradients
    loss, grads = jax.value_and_grad(loss_fn_decoder)(params, model, inputs, targets)
    
    # 2. Get Updates from Optax
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    
    # 3. Apply Updates to Parameters
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss

# --- CONFIGURATION ---
num_steps = 5000       # How many batches to train on
log_every = 100        # Print loss every X steps
eval_every = 500       # Generate text every X steps

# --- THE LOOP ---
print("Starting Training...")
start_time = time.time()
losses = []

for step in range(num_steps):
    
    # 1. Get Data
    inputs, targets = loader.get_batch('train')
    
    # 2. Run One Step
    # FIX: Pass opt_state and model correctly. Capture the new opt_state!
    params, opt_state, loss_val = train_step(params, opt_state, model, inputs, targets)
    
    losses.append(loss_val)
    
    # 3. Logging
    if step % log_every == 0:
        print(f"Step {step} | Loss: {loss_val:.4f} | Time: {time.time() - start_time:.2f}s")
        start_time = time.time() # Reset timer
        
    # 4. Watch it Learn (Evaluation)
    if step % eval_every == 0:
        print("\n--- GENERATING TEXT ---")
        # Start with a simple prompt
        context = jnp.array([loader.encode("The")]) 
        
        # Generate 30 tokens
        output_ids = model.generate(params, context, max_new_tokens=30)
        
        # Decode and print
        print(loader.decode(output_ids[0].tolist()))
        print("-----------------------\n")

print("Training Complete!")