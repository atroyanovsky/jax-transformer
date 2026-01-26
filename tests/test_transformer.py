"""
Tests for the Transformer model
"""

import jax
import jax.numpy as jnp
import pytest
from src.models.transformer import Transformer
from src.config import Config


class TestTransformer:
    def test_model_initialization(self):
        """Test that model can be initialized"""
        model = Transformer(
            Config.NUM_HEADS, 
            Config.MAX_LEN, 
            Config.D_MODEL, 
            Config.VOCAB_SIZE, 
            Config.NUM_LAYERS, 
            Config.D_FF
        )
        
        assert model.num_heads == Config.NUM_HEADS
        assert model.max_len == Config.MAX_LEN
        assert model.d_model == Config.D_MODEL
        assert model.vocab_size == Config.VOCAB_SIZE
        assert model.num_layers == Config.NUM_LAYERS
        assert model.d_ff == Config.D_FF
    
    def test_parameter_initialization(self):
        """Test that parameters can be initialized"""
        model = Transformer(
            Config.NUM_HEADS, 
            Config.MAX_LEN, 
            Config.D_MODEL, 
            Config.VOCAB_SIZE, 
            Config.NUM_LAYERS, 
            Config.D_FF
        )
        
        key = jax.random.PRNGKey(42)
        params = model.init_params(key)
        
        # Check that all required keys exist
        assert 'embedding' in params
        assert 'W_vocab' in params
        assert 'b_vocab' in params
        assert 'layers' in params
        assert len(params['layers']) == Config.NUM_LAYERS
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        model = Transformer(
            Config.NUM_HEADS, 
            Config.MAX_LEN, 
            Config.D_MODEL, 
            Config.VOCAB_SIZE, 
            Config.NUM_LAYERS, 
            Config.D_FF
        )
        
        key = jax.random.PRNGKey(42)
        params = model.init_params(key)
        
        is_valid, message = model.validate_params(params)
        assert is_valid, f"Parameter validation failed: {message}"
    
    def test_forward_pass(self):
        """Test that forward pass works"""
        model = Transformer(
            Config.NUM_HEADS, 
            Config.MAX_LEN, 
            Config.D_MODEL, 
            Config.VOCAB_SIZE, 
            Config.NUM_LAYERS, 
            Config.D_FF
        )
        
        key = jax.random.PRNGKey(42)
        params = model.init_params(key)
        
        # Create dummy input
        batch_size = 2
        seq_len = Config.SEQ_LEN
        inputs = jax.random.randint(key, (batch_size, seq_len), 0, Config.VOCAB_SIZE)
        
        # Forward pass
        output = model.encoder(params, inputs)
        logits = model.classifier_head(params, output)
        
        # Check shapes
        expected_output_shape = (batch_size, seq_len, Config.D_MODEL)
        expected_logits_shape = (batch_size, seq_len, Config.VOCAB_SIZE)
        
        assert output.shape == expected_output_shape
        assert logits.shape == expected_logits_shape


if __name__ == "__main__":
    pytest.main([__file__])
