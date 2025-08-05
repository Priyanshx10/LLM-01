def tokenize(text):
    return text.lower().split()

def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def encode(tokens, word2idx):
    return [word2idx[word] for word in tokens]

def decode(indices, idx2word):
    return [idx2word[idx] for idx in indices]
