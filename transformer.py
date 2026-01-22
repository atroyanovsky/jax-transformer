import jnp
import jax

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

def feed_forward(x, params):

    expanded_x = jnp.matmul(x, params["W_1"]) + params["b_1"]
    act_expanded_x = jax.nn.relu(expanded_x)
    act_x = jnp.matmul(act_expanded_x, params["W_2"]) + params["b_2"]

    return act_x

def transformer_block(x, all_params, num_heads):
    multi_head_res = multi_head_attention(x, all_params["attention"], num_heads)

    multi_head_res = layer_norm(x + multi_head_res)

    ff_res = feed_forward(multi_head_res, all_params["ffn"])

    ff_res = layer_norm(multi_head_res + ff_res)

    return ff_res

def positional_encoding(max_len, d_model):
    pe = jnp.zeros((max_len, d_model))
    position = jnp.arange(0, max_len).reshape(-1, 1)
    div_idx = jnp.arange(0, d_model, 2)
    div_term = jnp.exp(div_idx * -(jnp.log(10000.0) / d_model))

    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

    return pe

def encoder(input_tokens, params, max_len, num_heads):
    # input_tokens: (Batch, Seq_Len) - Integers
    
    # 1. Embedding Lookup
    # We assume params['embedding'] is a matrix of (vocab_size, d_model)
    # JAX allows simple indexing: embeddings[indices]
    x = params['embedding'][input_tokens] 
    
    # 2. Add Positional Encoding
    d_model = x.shape[-1]
    pe = positional_encoding(max_len, d_model)
    
    # Slice PE to matching length (handling batches with broadcasting)
    seq_len = x.shape[1]
    x = x + pe[:seq_len, :]
    
    # 3. Apply Dropout (Skipping for this simple inference version)
    
    # 4. Loop through Transformer Blocks
    # params['layers'] is a list of dictionaries, one for each block
    for layer_params in params['layers']:
        x = transformer_block(x, layer_params, num_heads)
        
    return x