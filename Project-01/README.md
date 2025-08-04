🧠 Project 01 – Character-Level Tiny LLM (Built from Scratch)
This project is a minimal implementation of a character-level language model using PyTorch. It reads raw text, learns character-level patterns using an LSTM, and generates text one character at a time.

It’s the perfect starting point to understand how LLMs really work, without relying on complex libraries or pretrained models.

🚧 Project Structure
project-01-char-llm/
│
├── data.txt # Training data (raw text file)
├── tiny_llm_char.py # Main file – Character-level LLM model and training
├── requirements.txt # Python dependencies
└── README.md # You're here!

📦 Setup
Clone this folder or create a new directory:
mkdir project-01-char-llm && cd project-01-char-llm

Add training data.
Create a file named data.txt and paste in any plain English paragraph. For example:
The quick brown fox jumps over the lazy dog.
The dog barked loudly while the fox ran away.

Install requirements:
pip install -r requirements.txt

🚀 How to Run the Model
python tiny_llm_char.py
