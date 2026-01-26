# train_tokenizer.py
import requests
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Download Tiny Shakespeare
file_path = "data/tiny_shakespeare.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

print("Downloading Shakespeare...")
with open(file_path, "w") as f:
    f.write(requests.get(url).text)

# 2. Initialize a blank BPE Tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 3. Train it on the file
# We choose a small vocab size (5000) because the dataset is small (1MB)
trainer = BpeTrainer(
    vocab_size=5000, 
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

print("Training Tokenizer...")
tokenizer.train(files=[file_path], trainer=trainer)

# 4. Save it
tokenizer.save("data/shakespeare_tokenizer.json")
print("✅ Tokenizer saved to 'data/shakespeare_tokenizer.json'")

# Test it
encoded = tokenizer.encode("Thou art a programmer")
print(f"Test Tokens: {encoded.tokens}")