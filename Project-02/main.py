# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import tokenize, build_vocab, encode, decode

# Load and tokenize data
with open('data.txt', 'r') as f:
    text = f.read()

tokens = tokenize(text)
vocab, stoi, itos = build_vocab(tokens)
vocab_size = len(vocab)
encoded_data = torch.tensor(encode(tokens, stoi), dtype=torch.long)

# Train/Val Split
n = int(0.9 * len(encoded_data))
train_data = encoded_data[:n]
val_data = encoded_data[n:]

# Hyperparameters
context_size = 8  # number of previous words to look at
embedding_dim = 32
hidden_dim = 64

# Model
class TinyWordLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = TinyWordLLM(vocab_size, embedding_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    ix = torch.randint(0, len(train_data) - context_size - 1, (32,))
    x = torch.stack([train_data[i:i+context_size] for i in ix])
    y = torch.stack([train_data[i+1:i+context_size+1] for i in ix])

    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generate function
def generate(model, start="the", length=20):
    model.eval()
    context = tokenize(start)
    context_idx = [stoi.get(w, 0) for w in context][-context_size:]
    context_tensor = torch.tensor([context_idx], dtype=torch.long)

    result = context.copy()
    for _ in range(length):
        logits = model(context_tensor)
        probs = F.softmax(logits[0, -1], dim=0)
        next_word_idx = torch.multinomial(probs, 1).item()
        result.append(itos[next_word_idx])
        context_tensor = torch.cat([context_tensor, torch.tensor([[next_word_idx]])], dim=1)
        context_tensor = context_tensor[:, -context_size:]

    return ' '.join(result)

print(generate(model, start="the world"))
