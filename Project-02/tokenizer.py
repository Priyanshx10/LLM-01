# tokenizer.py
import re

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(tokens):
    vocab = sorted(set(tokens))
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    return vocab, stoi, itos

def encode(tokens, stoi):
    return [stoi[token] for token in tokens]

def decode(indices, itos):
    return ' '.join([itos[i] for i in indices])
