import numpy as np
import jax.numpy as jnp
from tokenizers import Tokenizer
import os

class BPELoader:
    def __init__(self, tokenizer_path, data_path, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # 1. Load the Tokenizer you trained
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"Vocab Size: {self.vocab_size}")
        
        # 2. Load the Text
        with open(data_path, 'r') as f:
            text = f.read()
            
        # 3. Encode the entire dataset (This takes a second)
        # We get a huge list of integers: [102, 554, 12, ...]
        print("Encoding dataset into BPE tokens...")
        encoded = self.tokenizer.encode(text)
        self.data = np.array(encoded.ids)
        print(f"Total tokens in dataset: {len(self.data)}")
        
        # 4. Train/Val Split (90/10)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split='train'):
        """
        Grab a random chunk of BPE tokens.
        """
        data = self.train_data if split == 'train' else self.val_data
        
        # Randomly pick starting positions
        ix = np.random.randint(len(data) - self.seq_len - 1, size=(self.batch_size,))
        
        x_stack = []
        y_stack = []
        
        for i in ix:
            # Slice the integer array
            chunk = data[i : i + self.seq_len + 1]
            x_stack.append(chunk[:-1]) # Inputs
            y_stack.append(chunk[1:])  # Targets (Shifted)
            
        return jnp.array(x_stack), jnp.array(y_stack)

    def decode(self, token_ids):
        # Helper to turn list of ints back to string
        # We use skip_special_tokens=True to hide [PAD], [UNK]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def encode(self, text):
        return self.tokenizer.encode(text).ids