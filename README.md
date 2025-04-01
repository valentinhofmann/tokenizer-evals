# Tokenizer Evaluation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/transformers/)
[![Datasets](https://img.shields.io/badge/Datasets-HuggingFace-blueviolet.svg)](https://huggingface.co/datasets)

This repository provides a simple script to evaluate the **fertility** of LLM tokenizers using real user inputs from popular conversational datasets. 
Fertility is defined as the **average number of tokens per word** and reflects how efficiently a tokenizer represents text.

---

## 📂 Repository Structure

```bash
.
├── src/
│   ├── analyze_tokenizer.py   # Main script for evaluating fertility
│   └── utils.py               # Data loading and metric computation
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/valentinhofmann/tokenizer-evaluation.git
cd tokenizer-evaluation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python src/analyze_tokenizer.py --tokenizer <tokenizer-name> [--n_samples N] [--english_only]
```

Required arguments:
- `tokenizer`: Hugging Face tokenizer name (e.g., `gpt2`, `bert-base-uncased`)

Optional arguments:
- `n_samples`: Number of samples to evaluate per dataset (default: 1000)
- `english_only`: Filter only English messages

Example:

```bash
python src/analyze_tokenizer.py --tokenizer gpt2 --n_samples 10000 --english_only
```

## 📊 Output

For each dataset, the script prints fertility statistics like:

```bash
--- FERTILITY statistics on wildchat ---
Mean:     1.378
Std Dev:  0.292
Min:      0.917
Max:      2.143
25%:      1.205
Median:   1.349
75%:      1.502
```

## 📚 Supported Datasets

- WildChat (`allenai/WildChat`)

- Chatbot Arena (`lmsys/chatbot_arena_conversations`)
