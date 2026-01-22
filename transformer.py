import jax.numpy as jnp
import jax

class Transformer:
    def __init__(self, num_heads, max_len, d_model, vocab_size, num_layers, d_ff):
        self.num_heads = num_heads
        self.max_len = max_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_ff = d_ff
    
    def init_params(self, key):
        """
        Initialize all model parameters.
        
        Args:
            key: JAX random key
            
        Returns:
            dict: Nested dictionary containing all model parameters
        """
        keys = jax.random.split(key, self.num_layers * 6 + 3)  # 6 per layer + embedding + classifier
        key_idx = 0
        
        params = {}
        
        # Embedding matrix
        params['embedding'] = jax.random.normal(keys[key_idx], (self.vocab_size, self.d_model))
        key_idx += 1
        
        # Classifier head
        params['W_vocab'] = jax.random.normal(keys[key_idx], (self.d_model, self.vocab_size))
        key_idx += 1
        params['b_vocab'] = jnp.zeros(self.vocab_size)
        
        # Transformer layers
        params['layers'] = []
        for i in range(self.num_layers):
            layer_params = {}
            
            # Attention parameters
            layer_params['attention'] = {
                'W_q': jax.random.normal(keys[key_idx], (self.d_model, self.d_model)),
                'b_q': jnp.zeros(self.d_model),
                'W_k': jax.random.normal(keys[key_idx + 1], (self.d_model, self.d_model)),
                'b_k': jnp.zeros(self.d_model),
                'W_v': jax.random.normal(keys[key_idx + 2], (self.d_model, self.d_model)),
                'b_v': jnp.zeros(self.d_model),
                'W_o': jax.random.normal(keys[key_idx + 3], (self.d_model, self.d_model)),
                'b_o': jnp.zeros(self.d_model),
            }

            layer_params['norm1'] = {
                'gamma': jnp.ones(self.d_model),
                'beta': jnp.zeros(self.d_model)
            }
            
            key_idx += 4
            
            # Feed-forward parameters
            layer_params['ffn'] = {
                'W_1': jax.random.normal(keys[key_idx], (self.d_model, self.d_ff)),
                'b_1': jnp.zeros(self.d_ff),
                'W_2': jax.random.normal(keys[key_idx + 1], (self.d_ff, self.d_model)),
                'b_2': jnp.zeros(self.d_model),
            }

            layer_params['norm2'] = {
                'gamma': jnp.ones(self.d_model),
                'beta': jnp.zeros(self.d_model)
            }

            key_idx += 2
            
            params['layers'].append(layer_params)
        
        return params
    
    def attention(self, Q, K, V):
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

    def project(self, x, W, b):
        return jnp.matmul(x, W) + b

    def layer_norm(self, x, params, eps=1e-6):
        ###
        ### x: Input matrix -> (batch_size, seq_len, d_k)
        ###
        ### eps: Numerical stability for sigma
        ###

        mu = jnp.mean(x, 2, keepdims=True)
        sigma = jnp.std(x, 2,keepdims=True)
         
        norm = (x - mu) / (sigma + eps)


        return norm * params['gamma'] + params['beta']

    def residual_block(self, x, layer_norm, sublayer):
        ###
        ### x: Input matrix -> (batch_size, seq_len, d_k)
        ### layer_norm: layer norm function
        ### sublayer: sublayer func (attention func)
        return layer_norm(x + sublayer(x))

    def split_heads(self, x, num_heads):
        head_depth = x.shape[-1] // num_heads
        split_head_vals = jnp.reshape(x, x.shape[:2] + (num_heads, head_depth))
        split_head_vals = split_head_vals.swapaxes(1, 2)
        return split_head_vals

    def merge_heads(self, split_heads):
        ### split_heads -> (Batch, Heads, Seq_Len, Depth)
        split_heads = split_heads.swapaxes(1, 2)
        merged_heads = jnp.reshape(split_heads, (split_heads.shape[0], split_heads.shape[1], -1))

        return merged_heads

    def multi_head_attention(self, x, attention_params):
        Q = self.project(x, attention_params["W_q"], attention_params["b_q"])
        K = self.project(x, attention_params["W_k"], attention_params["b_k"])
        V = self.project(x, attention_params["W_v"], attention_params["b_v"])

        Q = self.split_heads(Q, num_heads=self.num_heads)
        K = self.split_heads(K, num_heads=self.num_heads)
        V = self.split_heads(V, num_heads=self.num_heads)

        res = self.attention(Q, K, V)

        res = self.merge_heads(res)

        res = self.project(res, attention_params["W_o"], attention_params["b_o"])

        return res

    def feed_forward(self, x, ffn_params):

        expanded_x = jnp.matmul(x, ffn_params["W_1"]) + ffn_params["b_1"]
        act_expanded_x = jax.nn.relu(expanded_x)
        act_x = jnp.matmul(act_expanded_x, ffn_params["W_2"]) + ffn_params["b_2"]

        return act_x

    def transformer_block(self, x, layer_params):
        multi_head_res = self.multi_head_attention(x, layer_params["attention"])

        multi_head_res = self.layer_norm(x + multi_head_res, layer_params["norm1"])

        ff_res = self.feed_forward(multi_head_res, layer_params["ffn"])

        ff_res = self.layer_norm(multi_head_res + ff_res, layer_params["norm2"])

        return ff_res

    def positional_encoding(self, max_len, d_model):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len).reshape(-1, 1)
        div_idx = jnp.arange(0, d_model, 2)
        div_term = jnp.exp(div_idx * -(jnp.log(10000.0) / d_model))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe

    def encoder(self, params, input_tokens):
        # input_tokens: (Batch, Seq_Len) - Integers
        
        # 1. Embedding Lookup
        x = params['embedding'][input_tokens] * jnp.sqrt(self.d_model)
        
        # 2. Add Positional Encoding
        d_model = x.shape[-1]
        pe = self.positional_encoding(self.max_len, d_model)
        
        # Slice PE to matching length (handling batches with broadcasting)
        seq_len = x.shape[1]
        x = x + pe[:seq_len, :]
        
        # 3. Apply Dropout (Skipping for this simple inference version)
        
        # 4. Loop through Transformer Blocks
        for layer_params in params['layers']:
            x = self.transformer_block(x, layer_params)
            
        return x

    def classifier_head(self, params, x):
        # x: (Batch, Seq_Len, d_model)
        # W_vocab: (d_model, vocab_size)
        
        # 1. Project to Vocabulary size
        logits = jnp.matmul(x, params['W_vocab']) + params['b_vocab']
        
        # 2. Convert to probabilities
        probs = jax.nn.log_softmax(logits)
        
        return probs
    
    def validate_params(self, params):
        """
        Validate that all parameters have the correct dimensions.
        Uses instance config values for expected dimensions.
        
        Args:
            params: Parameter dictionary to validate
        
        Returns:
            bool: True if all dimensions are correct
            str: Error message if validation fails
        """
        try:
            # Check embedding matrix
            if 'embedding' not in params:
                return False, "Missing embedding matrix"
            if params['embedding'].shape != (self.vocab_size, self.d_model):
                return False, f"Embedding shape: expected ({self.vocab_size}, {self.d_model}), got {params['embedding'].shape}"
            
            # Check classifier head
            if 'W_vocab' not in params or 'b_vocab' not in params:
                return False, "Missing classifier head parameters"
            if params['W_vocab'].shape != (self.d_model, self.vocab_size):
                return False, f"W_vocab shape: expected ({self.d_model}, {self.vocab_size}), got {params['W_vocab'].shape}"
            if params['b_vocab'].shape != (self.vocab_size,):
                return False, f"b_vocab shape: expected ({self.vocab_size},), got {params['b_vocab'].shape}"
            
            # Check layers
            if 'layers' not in params:
                return False, "Missing layers parameter"
            if len(params['layers']) != self.num_layers:
                return False, f"Number of layers: expected {self.num_layers}, got {len(params['layers'])}"
            
            # Check each layer
            for i, layer_params in enumerate(params['layers']):
                # Check attention parameters
                attention_params = layer_params.get('attention', {})
                for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
                    if param_name not in attention_params:
                        return False, f"Layer {i}: Missing {param_name} in attention"
                    expected_shape = (self.d_model, self.d_model)
                    if attention_params[param_name].shape != expected_shape:
                        return False, f"Layer {i}: {param_name} shape: expected {expected_shape}, got {attention_params[param_name].shape}"
                
                for param_name in ['b_q', 'b_k', 'b_v', 'b_o']:
                    if param_name not in attention_params:
                        return False, f"Layer {i}: Missing {param_name} in attention"
                    if attention_params[param_name].shape != (self.d_model,):
                        return False, f"Layer {i}: {param_name} shape: expected ({self.d_model},), got {attention_params[param_name].shape}"
                
                # Check feed-forward parameters
                ffn_params = layer_params.get('ffn', {})
                for param_name in ['W_1', 'W_2']:
                    if param_name not in ffn_params:
                        return False, f"Layer {i}: Missing {param_name} in feed-forward"
                    expected_shape = (self.d_model, self.d_ff) if param_name == 'W_1' else (self.d_ff, self.d_model)
                    if ffn_params[param_name].shape != expected_shape:
                        return False, f"Layer {i}: {param_name} shape: expected {expected_shape}, got {ffn_params[param_name].shape}"
                
                for param_name in ['b_1', 'b_2']:
                    if param_name not in ffn_params:
                        return False, f"Layer {i}: Missing {param_name} in feed-forward"
                    expected_shape = (self.d_ff,) if param_name == 'b_1' else (self.d_model,)
                    if ffn_params[param_name].shape != expected_shape:
                        return False, f"Layer {i}: {param_name} shape: expected {expected_shape}, got {ffn_params[param_name].shape}"
                
                # Check layer normalization parameters
                for norm_name in ['norm1', 'norm2']:
                    norm_params = layer_params.get(norm_name, {})
                    for param_name in ['gamma', 'beta']:
                        if param_name not in norm_params:
                            return False, f"Layer {i}: Missing {param_name} in {norm_name}"
                        if norm_params[param_name].shape != (self.d_model,):
                            return False, f"Layer {i}: {norm_name} {param_name} shape: expected ({self.d_model},), got {norm_params[param_name].shape}"
            
            return True, "All parameter dimensions are correct"
            
        except Exception as e:
            return False, f"Error during validation: {str(e)}"


def loss_fn(params, model, inputs, targets):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        params: Model parameters
        model: Transformer instance
        inputs: Input token sequences (batch_size, seq_len)
        targets: Target token sequences (batch_size, seq_len)
    
    Returns:
        float: Cross-entropy loss
    """
    # Forward pass
    logits = model.encoder(params, inputs)
    log_probs = model.classifier_head(params, logits)
    
    # Compute cross-entropy loss
    # targets are the true next tokens
    batch_size, seq_len, vocab_size = log_probs.shape
    
    # Gather log probabilities for target tokens
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    
    # Average over batch and sequence length
    loss = -jnp.mean(target_log_probs)
    
    return loss