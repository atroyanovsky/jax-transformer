# JAX Transformer

An object-oriented implementation of a Transformer model using JAX. This project is for self-educational purposes.

## Project Structure

```
jax-transformer/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py          # Transformer model implementation
│   ├── training/
│   │   ├── __init__.py
│   │   └── training_loop.py         # Training utilities
│   ├── config.py                    # Configuration settings
│   └── __init__.py
├── scripts/
│   └── tokenizer.py                 # Tokenizer training script
├── data/
│   ├── tiny_shakespeare.txt         # Training data
│   └── shakespeare_tokenizer.json   # Trained tokenizer
├── checkpoints/                     # Model checkpoints
├── tests/                           # Unit tests
├── train.py                        # Main training script
└── README.md
```

## Features

- **Clean JAX OOP**: Follows JAX best practices with explicit parameter passing
- **Complete Transformer**: Multi-head attention, layer normalization, feed-forward networks
- **Training Loop**: JIT-compiled training with AdamW optimizer
- **Parameter Validation**: Comprehensive shape checking for all parameters
- **Modular Design**: Separated models, training, and configuration

## Quick Start

### Installation

```bash
uv sync
```

### Training

```bash
uv run train.py
```

### Tokenizer Training

```bash
uv run scripts/tokenizer.py
```

## Model Architecture

The implementation includes:

- **Multi-Head Self-Attention**: With learnable Q, K, V projections
- **Positional Encoding**: Sinusoidal positional embeddings
- **Layer Normalization**: With learnable gamma and beta parameters
- **Feed-Forward Networks**: Two-layer MLP with ReLU activation
- **Residual Connections**: Proper residual connections around attention and FFN blocks

## Configuration

Key hyperparameters are defined in `src/config.py`:

- `VOCAB_SIZE`: Vocabulary size
- `D_MODEL`: Model dimension
- `NUM_LAYERS`: Number of transformer layers
- `NUM_HEADS`: Number of attention heads
- `D_FF`: Feed-forward network dimension
- `MAX_LEN`: Maximum sequence length

## Usage Example

```python
import jax
from src.models.transformer import Transformer
from src.config import Config

# Create model
model = Transformer(
    Config.NUM_HEADS, 
    Config.MAX_LEN, 
    Config.D_MODEL, 
    Config.VOCAB_SIZE, 
    Config.NUM_LAYERS, 
    Config.D_FF
)

# Initialize parameters
key = jax.random.PRNGKey(42)
params = model.init_params(key)

# Forward pass
batch = jax.random.randint(key, (Config.BATCH_SIZE, Config.SEQ_LEN), 0, Config.VOCAB_SIZE)
output = model.encoder(params, batch)
logits = model.classifier_head(params, output)
```

## Design Principles

1. **JAX-idiomatic OOP**: Class stores configuration, parameters passed explicitly
2. **Functional purity**: All forward methods are pure functions
3. **Type safety**: Comprehensive parameter validation
4. **Modularity**: Clear separation of concerns
5. **Performance**: JIT-compiled training steps

## Testing

Run tests with:

```bash
pytest tests/
```
