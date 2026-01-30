import pickle
import os
import jax

def save_checkpoint(filename, params, opt_state, config, step, loss):
    """
    Saves the model state, optimizer state, and training metadata.
    """
    print(f"Saving checkpoint to {filename}...")
    
    # 1. Move everything to CPU (Numpy format)
    # This prevents errors when trying to pickle GPU arrays
    checkpoint_data = {
        'params': jax.device_get(params),
        'opt_state': jax.device_get(opt_state),
        'config': config,
        'step': step,
        'loss': loss
    }
    
    # 2. Save using pickle
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
        
    print("✅ Saved successfully.")

def load_checkpoint(filename):
    """
    Loads a checkpoint and returns the unpacked data.
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return None
        
    print(f"Loading checkpoint from {filename}...")
    
    with open(filename, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # JAX will automatically convert Numpy arrays back to Device Arrays 
    # when you use them in the first train_step, so we don't need manual casting here.
    return checkpoint_data