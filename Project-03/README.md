# 🚀 Project 03: Transformer-Based Mini LLM

Build your own mini **Language Learning Model (LLM)** from scratch using PyTorch — powered by the **Transformer** architecture. This project upgrades your understanding beyond LSTM and takes you into modern deep learning techniques.

---

## 🧠 What is this project?

This project is a word-level language model that uses the **Transformer** architecture — the same concept behind GPT and BERT. Unlike LSTMs that process words one by one, Transformers allow parallel processing and capture **long-range dependencies** more effectively.

---

## 📚 Concepts Covered

| 🧩 Component             | 🔍 Explanation                                                                   |
| ------------------------ | -------------------------------------------------------------------------------- |
| **Tokenization**         | Converts words into numerical indices (like `["hello", "world"]` → `[3, 17]`).   |
| **Embedding**            | Turns each index into a dense vector (`[3, 17]` → `[[0.2, 0.5], [0.9, 0.1]]`).   |
| **Positional Encoding**  | Adds position info so the model knows the word order.                            |
| **Self-Attention**       | Learns to focus on relevant words in a sequence, regardless of their position.   |
| **Multi-Head Attention** | Multiple attention mechanisms working in parallel to capture different patterns. |
| **Transformer Block**    | Layer made of attention + feed-forward network + layer norm + residuals.         |
| **Text Generation**      | Predicts the next word repeatedly to generate complete sentences.                |

---

## 🗂️ Project Structure

transformer_llm/
│
├── data.txt # Raw training text
├── transformer_llm.py # Main model & training script
├── README.md # You're reading it!
└── requirements.txt # Dependencies

## 🛠️ Setup & Installation

```bash
# 1. Clone the repo or save the files
git clone https://github.com/your-username/transformer-llm
cd transformer-llm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your training text (any .txt file)
# For example, data.txt = "hello world welcome to AI"

# 4. Run the model
python transformer_llm.py
```
