"""Training utilities for the copy task."""

import jax
import jax.numpy as jnp


def loss_fn(params, model, inputs, targets):
    """
    Compute cross-entropy loss.

    Args:
        params: Model parameters
        model: Transformer instance
        inputs: Input token IDs (batch, seq_len)
        targets: Target token IDs (batch, seq_len)

    Returns:
        Scalar cross-entropy loss
    """
    features = model.encoder(params, inputs)
    log_probs = model.classifier_head(params, features)
    label_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    return -jnp.mean(label_log_probs)


def get_batch(key, batch_size, seq_len, vocab_size):
    """
    Generate a random batch for the copy task.

    Args:
        key: JAX random key
        batch_size: Number of sequences
        seq_len: Length of each sequence
        vocab_size: Vocabulary size

    Returns:
        (inputs, targets) where targets == inputs
    """
    data = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    return data, data
