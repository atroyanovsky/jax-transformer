import jnp
import jax

class Transformer:
    def __init__(self, num_heads, max_len, params):
        self.num_heads = num_heads
        self.max_len = max_len
        self.params = params
        
        # Validate parameters during initialization
        # Note: We'll need to infer these from params or require them as args
        # For now, we'll skip validation in __init__ and let user call it explicitly
    
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

    def layer_norm(self, x, eps=1e-6):
        ###
        ### x: Input matrix -> (batch_size, seq_len, d_k)
        ###
        ### eps: Numerical stability for sigma
        ###

        mu = jnp.mean(x, 2, keepdims=True)
        sigma = jnp.std(x, 2,keepdims=True)

        return (x - mu) / (sigma + eps)

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

        multi_head_res = self.layer_norm(x + multi_head_res)

        ff_res = self.feed_forward(multi_head_res, layer_params["ffn"])

        ff_res = self.layer_norm(multi_head_res + ff_res)

        return ff_res

    def positional_encoding(self, max_len, d_model):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len).reshape(-1, 1)
        div_idx = jnp.arange(0, d_model, 2)
        div_term = jnp.exp(div_idx * -(jnp.log(10000.0) / d_model))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe

    def encoder(self, input_tokens):
        # input_tokens: (Batch, Seq_Len) - Integers
        
        # 1. Embedding Lookup
        # We assume self.params['embedding'] is a matrix of (vocab_size, d_model)
        # JAX allows simple indexing: embeddings[indices]
        x = self.params['embedding'][input_tokens] 
        
        # 2. Add Positional Encoding
        d_model = x.shape[-1]
        pe = self.positional_encoding(self.max_len, d_model)
        
        # Slice PE to matching length (handling batches with broadcasting)
        seq_len = x.shape[1]
        x = x + pe[:seq_len, :]
        
        # 3. Apply Dropout (Skipping for this simple inference version)
        
        # 4. Loop through Transformer Blocks
        # self.params['layers'] is a list of dictionaries, one for each block
        for layer_params in self.params['layers']:
            x = self.transformer_block(x, layer_params)
            
        return x

    def classifier_head(self, x):
        # x: (Batch, Seq_Len, d_model)
        # W_vocab: (d_model, vocab_size)
        
        # 1. Project to Vocabulary size
        logits = jnp.matmul(x, self.params['W_vocab']) + self.params['b_vocab']
        
        # 2. Convert to probabilities
        probs = jax.nn.softmax(logits)
        
        return probs
    
    def validate_params(self, d_model, vocab_size, num_layers, d_ff):
        """
        Validate that all parameters have the correct dimensions.
        
        Args:
            d_model: Model dimension
            vocab_size: Vocabulary size
            num_layers: Number of transformer layers
            d_ff: Feed-forward network dimension
        
        Returns:
            bool: True if all dimensions are correct
            str: Error message if validation fails
        """
        try:
            # Check embedding matrix
            if 'embedding' not in self.params:
                return False, "Missing embedding matrix"
            if self.params['embedding'].shape != (vocab_size, d_model):
                return False, f"Embedding shape: expected ({vocab_size}, {d_model}), got {self.params['embedding'].shape}"
            
            # Check classifier head
            if 'W_vocab' not in self.params or 'b_vocab' not in self.params:
                return False, "Missing classifier head parameters"
            if self.params['W_vocab'].shape != (d_model, vocab_size):
                return False, f"W_vocab shape: expected ({d_model}, {vocab_size}), got {self.params['W_vocab'].shape}"
            if self.params['b_vocab'].shape != (vocab_size,):
                return False, f"b_vocab shape: expected ({vocab_size},), got {self.params['b_vocab'].shape}"
            
            # Check layers
            if 'layers' not in self.params:
                return False, "Missing layers parameter"
            if len(self.params['layers']) != num_layers:
                return False, f"Number of layers: expected {num_layers}, got {len(self.params['layers'])}"
            
            # Check each layer
            for i, layer_params in enumerate(self.params['layers']):
                # Check attention parameters
                attention_params = layer_params.get('attention', {})
                for param_name in ['W_q', 'W_k', 'W_v', 'W_o']:
                    if param_name not in attention_params:
                        return False, f"Layer {i}: Missing {param_name} in attention"
                    expected_shape = (d_model, d_model)
                    if attention_params[param_name].shape != expected_shape:
                        return False, f"Layer {i}: {param_name} shape: expected {expected_shape}, got {attention_params[param_name].shape}"
                
                for param_name in ['b_q', 'b_k', 'b_v', 'b_o']:
                    if param_name not in attention_params:
                        return False, f"Layer {i}: Missing {param_name} in attention"
                    if attention_params[param_name].shape != (d_model,):
                        return False, f"Layer {i}: {param_name} shape: expected ({d_model},), got {attention_params[param_name].shape}"
                
                # Check feed-forward parameters
                ffn_params = layer_params.get('ffn', {})
                for param_name in ['W_1', 'W_2']:
                    if param_name not in ffn_params:
                        return False, f"Layer {i}: Missing {param_name} in feed-forward"
                    expected_shape = (d_model, d_ff) if param_name == 'W_1' else (d_ff, d_model)
                    if ffn_params[param_name].shape != expected_shape:
                        return False, f"Layer {i}: {param_name} shape: expected {expected_shape}, got {ffn_params[param_name].shape}"
                
                for param_name in ['b_1', 'b_2']:
                    if param_name not in ffn_params:
                        return False, f"Layer {i}: Missing {param_name} in feed-forward"
                    expected_shape = (d_ff,) if param_name == 'b_1' else (d_model,)
                    if ffn_params[param_name].shape != expected_shape:
                        return False, f"Layer {i}: {param_name} shape: expected {expected_shape}, got {ffn_params[param_name].shape}"
            
            return True, "All parameter dimensions are correct"
            
        except Exception as e:
            return False, f"Error during validation: {str(e)}"