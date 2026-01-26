#!/usr/bin/env python3
"""
Main training script for JAX Transformer
"""

import jax
import jax.numpy as jnp
import optax
from src.models.transformer import Transformer
from src.training.training_loop import loss_fn, get_batch
from src.config import Config

def main():
    # A. Initialization
    key = jax.random.PRNGKey(42)
    model = Transformer(
        Config.NUM_HEADS, 
        Config.MAX_LEN, 
        Config.D_MODEL, 
        Config.VOCAB_SIZE, 
        Config.NUM_LAYERS, 
        Config.D_FF
    )

    key, subkey = jax.random.split(key)
    params = model.init_params(subkey)

    # B. Optimizer (AdamW)
    optimizer = optax.adamw(learning_rate=Config.LR)
    opt_state = optimizer.init(params)

    # C. The JIT-Compiled Update Step
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, model, inputs, targets)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    # D. The Training Loop
    print(f"\n--- Starting Training (Steps: {Config.STEPS}) ---")

    for step in range(Config.STEPS):
        key, subkey = jax.random.split(key)
        inputs, targets = get_batch(subkey, Config.BATCH_SIZE, Config.SEQ_LEN, Config.VOCAB_SIZE)
        
        params, opt_state, loss = train_step(params, opt_state, inputs, targets)
        
        if step % 100 == 0:
            print(f"Step {step:04d} | Loss: {loss:.4f}")

    # E. Final Test
    print("\n--- Final Test ---")
    test_input = jax.random.randint(key, (1, Config.SEQ_LEN), 0, Config.VOCAB_SIZE)
    features = model.encoder(params, test_input)
    log_probs = model.classifier_head(params, features)
    predicted = jnp.argmax(log_probs, axis=-1)

    print(f"Input:     {test_input[0]}")
    print(f"Predicted: {predicted[0]}")

    if jnp.array_equal(test_input, predicted):
        print("✅ SUCCESS: Model perfectly copied the sequence.")
    else:
        print("❌ FAILURE: Model made mistakes.")

if __name__ == "__main__":
    main()
