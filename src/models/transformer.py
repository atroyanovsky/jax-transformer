"""
Transformer model implementation in JAX.

This is a encoder-only transformer trained on a sequence copy task,
where the model learns to reproduce its input.
"""

from re import S
import jax.numpy as jnp
import jax


class Transformer:
    """
    Stateless Transformer model following JAX idioms.

    The class stores only configuration; all parameters are passed
    explicitly to methods for JAX compatibility.
    """

    def __init__(self, num_heads, max_len, d_model, vocab_size, num_layers, d_ff, dropout_rate):
        self.num_heads = num_heads
        self.max_len = max_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
    
    def _xavier_he(self, key, shape):
        """Xavier/He initialization: std = sqrt(2 / fan_in)."""
        fan_in = shape[0]
        std = jnp.sqrt(2.0 / fan_in)
        return jax.random.normal(key, shape) * std

    def init_params(self, key):
        """
        Initialize all model parameters using He initialization.

        Args:
            key: JAX random key

        Returns:
            dict: Nested dictionary containing all model parameters
        """
        keys = jax.random.split(key, self.num_layers * 6 + 5)
        key_idx = 0

        params = {}

        # Embedding matrix
        params['embedding'] = self._xavier_he(keys[key_idx], (self.vocab_size, self.d_model))
        key_idx += 1

        # Classifier head
        params['W_vocab'] = self._xavier_he(keys[key_idx], (self.d_model, self.vocab_size))
        key_idx += 1
        params['b_vocab'] = jnp.zeros(self.vocab_size)

        params['final_norm'] = {
            'gamma': jnp.ones(self.d_model),
            'beta': jnp.zeros(self.d_model)
        }

        # Transformer layers
        params['layers'] = []
        for i in range(self.num_layers):
            layer_params = {}

            # Attention parameters
            layer_params['attention'] = {
                'W_q': self._xavier_he(keys[key_idx], (self.d_model, self.d_model)),
                'b_q': jnp.zeros(self.d_model),
                'W_k': self._xavier_he(keys[key_idx + 1], (self.d_model, self.d_model)),
                'b_k': jnp.zeros(self.d_model),
                'W_v': self._xavier_he(keys[key_idx + 2], (self.d_model, self.d_model)),
                'b_v': jnp.zeros(self.d_model),
                'W_o': self._xavier_he(keys[key_idx + 3], (self.d_model, self.d_model)),
                'b_o': jnp.zeros(self.d_model),
            }

            layer_params['norm1'] = {
                'gamma': jnp.ones(self.d_model),
                'beta': jnp.zeros(self.d_model)
            }

            key_idx += 4

            # Feed-forward parameters
            layer_params['ffn'] = {
                'W_1': self._xavier_he(keys[key_idx], (self.d_model, self.d_ff)),
                'b_1': jnp.zeros(self.d_ff),
                'W_2': self._xavier_he(keys[key_idx + 1], (self.d_ff, self.d_model)),
                'b_2': jnp.zeros(self.d_model),
            }

            layer_params['norm2'] = {
                'gamma': jnp.ones(self.d_model),
                'beta': jnp.zeros(self.d_model)
            }

            key_idx += 2
            
            params['layers'].append(layer_params)
        
        return params
    
    def attention(self, Q, K, V, mask, key, training):
        """
        Scaled dot-product attention.

        Args:
            Q: Query matrix (batch, num_heads, seq_len, d_k)
            K: Key matrix (batch, num_heads, seq_len, d_k)
            V: Value matrix (batch, num_heads, seq_len, d_k)

        Returns:
            Attention output (batch, num_heads, seq_len, d_k)
        """
        d_k = Q.shape[-1]
        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(d_k)

        if mask is not None:
            scores += mask

        probabilities = jax.nn.softmax(scores)

        if self.dropout_rate > 0 and training:
            probabilities = self.dropout(probabilities, key)

        return jnp.matmul(probabilities, V)

    def project(self, x, W, b):
        """Linear projection: x @ W + b."""
        return jnp.matmul(x, W) + b

    def layer_norm(self, x, params, eps=1e-6):
        """
        Layer normalization with learnable scale and shift.

        Args:
            x: Input (batch, seq_len, d_model)
            params: Dict with 'gamma' and 'beta'
            eps: Small constant for numerical stability

        Returns:
            Normalized output (batch, seq_len, d_model)
        """
        mu = jnp.mean(x, 2, keepdims=True)
        sigma = jnp.std(x, 2, keepdims=True)
        norm = (x - mu) / (sigma + eps)
        return norm * params['gamma'] + params['beta']

    def residual_block(self, x, layer_norm, sublayer):
        """Apply sublayer with residual connection: layer_norm(x + sublayer(x))."""
        return layer_norm(x + sublayer(x))

    def split_heads(self, x, num_heads):
        """
        Split the last dimension into (num_heads, depth) and transpose.

        Args:
            x: Input (batch, seq_len, d_model)
            num_heads: Number of attention heads

        Returns:
            Split tensor (batch, num_heads, seq_len, depth)
        """
        head_depth = x.shape[-1] // num_heads
        split_head_vals = jnp.reshape(x, x.shape[:2] + (num_heads, head_depth))
        return split_head_vals.swapaxes(1, 2)

    def merge_heads(self, x):
        """
        Inverse of split_heads: merge attention heads back together.

        Args:
            x: Split tensor (batch, num_heads, seq_len, depth)

        Returns:
            Merged tensor (batch, seq_len, d_model)
        """
        x = x.swapaxes(1, 2)
        return jnp.reshape(x, (x.shape[0], x.shape[1], -1))

    def multi_head_attention(self, x, attention_params, mask, key, training):
        """
        Multi-head self-attention.

        Projects input to Q, K, V, splits into heads, applies attention,
        merges heads, and projects output.

        Args:
            x: Input (batch, seq_len, d_model)
            attention_params: Dict with W_q, W_k, W_v, W_o and biases

        Returns:
            Attention output (batch, seq_len, d_model)
        """
        Q = self.project(x, attention_params["W_q"], attention_params["b_q"])
        K = self.project(x, attention_params["W_k"], attention_params["b_k"])
        V = self.project(x, attention_params["W_v"], attention_params["b_v"])

        Q = self.split_heads(Q, num_heads=self.num_heads)
        K = self.split_heads(K, num_heads=self.num_heads)
        V = self.split_heads(V, num_heads=self.num_heads)

        res = self.attention(Q, K, V, mask, key, training)
        res = self.merge_heads(res)
        res = self.project(res, attention_params["W_o"], attention_params["b_o"])
        return res

    def feed_forward(self, x, ffn_params):
        """
        Position-wise feed-forward network: two linear layers with ReLU.

        Args:
            x: Input (batch, seq_len, d_model)
            ffn_params: Dict with W_1, b_1, W_2, b_2

        Returns:
            Output (batch, seq_len, d_model)
        """
        hidden = jax.nn.relu(jnp.matmul(x, ffn_params["W_1"]) + ffn_params["b_1"])
        return jnp.matmul(hidden, ffn_params["W_2"]) + ffn_params["b_2"]

    def transformer_block(self, x, layer_params, mask=None, key=None, training=True):
        """
        Single transformer block: attention + FFN, each with residual and layer norm.

        Args:
            x: Input (batch, seq_len, d_model)
            layer_params: Dict with attention, ffn, norm1, norm2 params

        Returns:
            Output (batch, seq_len, d_model)
        """

        k1, k2, k3 = jax.random.split(key, 3)

        x_norm = self.layer_norm(x, layer_params["norm1"])

        attn_out = self.multi_head_attention(x_norm, layer_params["attention"], mask, k1, training)

        if self.dropout_rate > 0 and training:
            attn_out = self.dropout(attn_out, k2)
        
        x += attn_out

        x_norm = self.layer_norm(x, layer_params["norm2"])

        ff_out = self.feed_forward(x_norm, layer_params["ffn"])
        
        if self.dropout_rate > 0 and training:
            ff_out = self.dropout(ff_out, k3)

        x += ff_out
        
        return x

    def positional_encoding(self, max_len, d_model):
        """
        Generate sinusoidal positional encodings.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding matrix (max_len, d_model)
        """
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len).reshape(-1, 1)
        div_idx = jnp.arange(0, d_model, 2)
        div_term = jnp.exp(div_idx * -(jnp.log(10000.0) / d_model))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return pe

    def encoder(self, params, input_tokens):
        """
        Full encoder forward pass.

        Args:
            params: Model parameters dict
            input_tokens: Token IDs (batch, seq_len)

        Returns:
            Encoded representations (batch, seq_len, d_model)
        """
        x = params['embedding'][input_tokens] * jnp.sqrt(self.d_model)

        pe = self.positional_encoding(self.max_len, self.d_model)
        seq_len = x.shape[1]
        x = x + pe[:seq_len, :]

        for layer_params in params['layers']:
            x = self.transformer_block(x, layer_params)

        return x

    def classifier_head(self, params, x):
        """
        Project encoder output to vocabulary log-probabilities.

        Args:
            params: Model parameters dict
            x: Encoder output (batch, seq_len, d_model)

        Returns:
            Log-probabilities over vocabulary (batch, seq_len, vocab_size)
        """
        logits = jnp.matmul(x, params['W_vocab']) + params['b_vocab']
        return jax.nn.log_softmax(logits)
    
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

    def forward(self, params, input_tokens, key, training=True):
        # Input:

        # params: Your dictionary of weights.

        # x: A batch of integer token IDs. Shape: (Batch_Size, Seq_Len).

        # Output:

        # logits: Unnormalized scores for the next token. Shape: (Batch_Size, Seq_Len, Vocab_Size).

        x = params['embedding'][input_tokens] * jnp.sqrt(self.d_model)

        pe = self.positional_encoding(self.max_len, self.d_model)
        seq_len = x.shape[1]
        x = x + pe[:seq_len, :]

        k_emb, k_layers = jax.random.split(key)
        
        k_layers = jax.random.split(k_layers, self.num_layers)


        if self.dropout_rate > 0 and training:
            x = self.dropout(x=x, key=k_emb)

        causal_mask = self.make_causal_mask(seq_len)

        for i, layer_params in enumerate(params['layers']):
            x = self.transformer_block(x, layer_params, causal_mask, k_layers[i], training)

        x_norm = self.layer_norm(x, params["final_norm"])

        logits = jnp.matmul(x_norm, params['W_vocab']) + params['b_vocab']
        return logits
    
    def make_causal_mask(self, seq_len):
        
        mask = jnp.ones((seq_len, seq_len))
        mask = jnp.triu(mask, k=1)
        mask *= -1e9
        return mask


    def generate(self, params, inputs, max_new_tokens, temperature=1.0, key=None):
        # We need a random key for sampling!
        if key is None:
            key = jax.random.PRNGKey(42)
            
        for _ in range(max_new_tokens):
            # 1. Crop Context
            inputs_cond = inputs
            if inputs.shape[1] > self.max_len:
                inputs_cond = inputs[:, -self.max_len:]
            
            # 2. Forward Pass
            logits = self.forward(params, inputs_cond, key, training=False)
            # Focus on the last token: (Batch, Vocab)
            logits = logits[:, -1, :]
            
            # 3. Apply Temperature
            # Higher T (e.g. 1.0) = More creative/random
            # Lower T (e.g. 0.1) = More confident/conservative
            logits = logits / temperature
            
            # 4. Sample instead of Argmax
            key, subkey = jax.random.split(key)
            # jax.random.categorical expects unnormalized logits (no softmax needed)
            next_token = jax.random.categorical(subkey, logits, axis=-1)
            
            # 5. Append
            next_token = next_token.reshape(1, 1)
            inputs = jnp.concatenate([inputs, next_token], axis=1)
            
        return inputs

    def dropout(self, x, key):
        dropout_mask = jax.random.bernoulli(key, p=1 - self.dropout_rate, shape=x.shape)
        x = (x * dropout_mask) / (1.0 - self.dropout_rate)

        return x
