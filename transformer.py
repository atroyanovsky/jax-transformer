import jnp

def attention(Q, K, V):

    d_k = Q.shape[-1]
    scores = jnp.dot(Q, K.T) / jnp.sqrt(d_k)
    probabilities = jax.nn.softmax(scores)