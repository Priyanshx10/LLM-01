🧠 Project 02 – Word-Level LLM from Scratch with PyTorch
This project builds a tiny language model (LLM) that learns to predict the next word in a sentence. Unlike Project 01 (character-level), this model uses word-level tokenization for more meaningful text generation.

📁 Project Structure
project-02-word-llm/
├── data.txt # Raw text file (your dataset)
├── tokenizer.py # Word-level tokenizer functions
├── tiny_llm_word.py # The main model and training loop
├── requirements.txt # Required Python packages
└── README.md # You're reading this file!

🚀 What You’ll Learn
How word-level tokenization works
How to build a vocabulary of words instead of characters
How to train an LSTM to learn from word sequences
How to generate text word-by-word from your model

📦 Installation
Make sure Python 3.8+ is installed. Then, install requirements:
pip install -r requirements.txt

📄 Files Explained
✅ tokenizer.py
Custom tokenizer that:

Splits text into words using regex
Builds vocabulary (stoi, itos)
Converts text to integers (encode)
Converts integers back to words (decode)
Example:

from tokenizer import tokenize, build_vocab, encode, decode

text = open("data.txt").read()
tokens = tokenize(text)
vocab, stoi, itos = build_vocab(tokens)
encoded_data = torch.tensor(encode(tokens, stoi), dtype=torch.long)

✅ tiny_llm_word.py
Main training script containing:
Data loading and tokenization
PyTorch LSTM model
Training loop (predict next word)
generate function for producing new text

🧠 LLM Components
Component What it does
nn.Embedding Turns each word index into a dense vector
nn.LSTM Learns to model word sequences
nn.Linear Outputs scores over the vocabulary

✅ Summary
This project covers:

Data preprocessing
Word-level tokenization
LSTM-based sequence modeling
Language generation from scratch
