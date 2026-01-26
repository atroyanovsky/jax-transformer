import jax
from transformer import Transformer
import jax.numpy as jnp
import optax

def loss_fn(params, model, inputs, targets):
    features = model.encoder(params, inputs)
    log_probs = model.classifier_head(params, features)
    
    # Calculate NLL Loss
    # We select the log_prob of the target class at each position
    label_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    return -jnp.mean(label_log_probs)

def get_batch(key, batch_size, seq_len, vocab_size):
    """Generates random integers. Targets = Inputs (Copy Task)"""
    data = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    return data, data


# A. Configuration
VOCAB_SIZE = 50      # Small vocab
D_MODEL = 128        # Feature vector size
NUM_LAYERS = 5
NUM_HEADS = 4
D_FF = 256
BATCH_SIZE = 32
SEQ_LEN = 15
MAX_LEN = 50
STEPS = 6000
LR = 1e-3

# B. Initialization
key = jax.random.PRNGKey(42)
model = Transformer(NUM_HEADS, MAX_LEN, D_MODEL, VOCAB_SIZE, NUM_LAYERS, D_FF)

key, subkey = jax.random.split(key)
params = model.init_params(subkey)

# C. Optimizer (AdamW)
optimizer = optax.adamw(learning_rate=LR)
opt_state = optimizer.init(params)

# D. The JIT-Compiled Update Step
@jax.jit
def train_step(params, opt_state, inputs, targets):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, model, inputs, targets)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val

# E. The Training Loop
print(f"\n--- Starting Training (Steps: {STEPS}) ---")

for step in range(STEPS):
    key, subkey = jax.random.split(key)
    inputs, targets = get_batch(subkey, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    params, opt_state, loss = train_step(params, opt_state, inputs, targets)
    
    if step % 100 == 0:
        print(f"Step {step:04d} | Loss: {loss:.4f}")

# F. Final Test
print("\n--- Final Test ---")
test_input = jax.random.randint(key, (1, SEQ_LEN), 0, VOCAB_SIZE)
features = model.encoder(params, test_input)
log_probs = model.classifier_head(params, features)
predicted = jnp.argmax(log_probs, axis=-1)

print(f"Input:     {test_input[0]}")
print(f"Predicted: {predicted[0]}")

if jnp.array_equal(test_input, predicted):
    print("✅ SUCCESS: Model perfectly copied the sequence.")
else:
    print("❌ FAILURE: Model made mistakes.")