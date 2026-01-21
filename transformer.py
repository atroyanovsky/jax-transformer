import jnp

def attention(Q, K, V):
    ###
    ### Q: Query matrix -> (batch_size, num_heads, seq_len, d_k)
    ###
    ### K: Key matrix -> (batch_size, num_heads, seq_len, d_k)
    ###
    ### V: Value matrix -> (batch_size, num_heads, seq_len, d_k)
    ###
    ### d_k: len feature vector for a specific word
    d_k = Q.shape[-1]
    scores = jnp.matmul(Q, jnp.swapaxes(K, -2,-1)) / jnp.sqrt(d_k) # -> (batch_size, num_heads, seq_len, seq_len)
    probabilities = jax.nn.softmax(scores)
    return jnp.matmul(probabilities, V) # -> (batch_size, num_heads, seq_len, d_k)

def project(x, W, b):
    return jnp.matmul(x, W) + b

def layer_norm(x, eps=1e-6):
    ###
    ### x: Input matrix -> (batch_size, seq_len, d_k)
    ###
    ### eps: Numerical stability for sigma
    ###

    mu = jnp.mean(x, 2, keepdims=True)
    sigma = jnp.std(x, 2,keepdims=True)

    return (x - mu) / (sigma + eps)

def residual_block(x, layer_norm, sublayer):
    ###
    ### x: Input matrix -> (batch_size, seq_len, d_k)
    ### layer_norm: layer norm function
    ### sublayer: sublayer func (attention func)
    return layer_norm(x + sublayer(x))

def split_heads(x, num_heads):
    head_depth = x.shape[-1] // num_heads
    split_head_vals = jnp.reshape(x, x.shape[:2] + (num_heads, head_depth))
    split_head_vals = split_head_vals.swapaxes(1, 2)
    return split_head_vals

def merge_heads(split_heads):
    ### split_heads -> (Batch, Heads, Seq_Len, Depth)
    split_heads = split_heads.swapaxes(1, 2)
    merged_heads = jnp.reshape(split_heads, (split_heads.shape[0], split_heads.shape[1], -1))

    return merged_heads

def multi_head_attention(x, params, num_heads):
    Q = project(x, params["W_q"], params["b_q"])
    K = project(x, params["W_k"], params["b_k"])
    V = project(x, params["W_v"], params["b_v"])

    Q = split_heads(Q, num_heads=num_heads)
    K = split_heads(K, num_heads=num_heads)
    V = split_heads(V, num_heads=num_heads)

    res = attention(Q, K, V)

    res = merge_heads(res)

    res = project(res, params["W_o"], params["b_o"])

    return res