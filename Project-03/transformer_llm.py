import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import tokenize, build_vocab, encode, decode
import os

# Load and tokenize text
if not os.path.exists("data.txt"):
    text = "hello world hello machine learning world hello transformer"
else:
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

tokens = tokenize(text)
stoi, itos = build_vocab(tokens)
vocab_size = len(stoi)
data = torch.tensor(encode(tokens, stoi), dtype=torch.long)

# Train-validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Model hyperparameters
block_size = 8
batch_size = 16
embedding_dim = 64
num_heads = 4
num_layers = 2
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Guard clause: Check if data is long enough
if len(train_data) < block_size + 1:
    raise ValueError(f"Training data too short! Got {len(train_data)} tokens. Need at least {block_size+1}.")

# Positional Encoding
def get_positional_encoding(block_size, embedding_dim):
    pos = torch.arange(block_size).unsqueeze(1)
    i = torch.arange(embedding_dim).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / embedding_dim)
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return angle_rads.unsqueeze(0)

# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)

# Transformer LLM model
class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = get_positional_encoding(block_size, embed_dim).to(device)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, heads) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        x = self.blocks(x)
        return self.linear(x)

# Batch generator
def get_batch(data):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Not enough data to get a batch.")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Train
model = TransformerLLM(vocab_size, embedding_dim, num_heads, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("\nStarting training...\n")
for epoch in range(epochs):
    model.train()
    x, y = get_batch(train_data)
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generate text
def generate(model, start_text, length=20):
    model.eval()
    tokens = tokenize(start_text)
    input_ids = torch.tensor([encode(tokens, stoi)], dtype=torch.long).to(device)

    for _ in range(length):
        input_chunk = input_ids[:, -block_size:]
        logits = model(input_chunk)
        probs = F.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    decoded = decode(input_ids[0].tolist(), itos)
    return " ".join(decoded)

print("\nGenerated text:\n")
print(generate(model, "hello world"))
