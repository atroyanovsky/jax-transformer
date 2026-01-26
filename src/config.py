"""
Configuration settings for the JAX Transformer
"""

class Config:
    # Model configuration
    VOCAB_SIZE = 50      # Small vocab for copy task
    D_MODEL = 128        # Feature vector size
    NUM_LAYERS = 5
    NUM_HEADS = 4
    D_FF = 256
    MAX_LEN = 50
    
    # Training configuration
    BATCH_SIZE = 32
    SEQ_LEN = 15
    STEPS = 6000
    LR = 1e-3
    
    # Data configuration
    DATA_DIR = "data"
    CHECKPOINT_DIR = "checkpoints"
