import torch
import torch.nn as nn
import torch.nn.functional as F

# Load text
with open('data.txt', 'r') as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Encode and decode
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Split data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Model
class TinyLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 64, batch_first=True)
        self.linear = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = TinyLLM(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    ix = torch.randint(0, len(train_data) - 10, (32,))
    x = torch.stack([train_data[i:i+10] for i in ix])
    y = torch.stack([train_data[i+1:i+11] for i in ix])

    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

# Generate sample
def generate(model, start="H", length=100):
    model.eval()
    context = torch.tensor([encode(start)], dtype=torch.long)
    result = list(start)
    for _ in range(length):
        logits = model(context)
        probs = F.softmax(logits[0, -1], dim=0)
        next_char = torch.multinomial(probs, 1).item()
        result.append(itos[next_char])
        context = torch.cat([context, torch.tensor([[next_char]])], dim=1)
    return ''.join(result)

print(generate(model))
