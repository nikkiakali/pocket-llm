# Pocket-LLM 🪶

[![CI](https://github.com/nikkiakali/pocket-llm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nikkiakali/pocket-llm/actions/workflows/ci.yml)

Tiny GPT-style LLM from scratch: tokenizer, transformer, training & inference.  
Built for experimentation, understanding core model internals, and pushing small models to be usable in lightweight settings.

---

## 📖 Docs & Samples
- **Model Card:** [MODEL_CARD.md](MODEL_CARD.md)  
- **Experiments/Results:** [EXPERIMENTS.md](EXPERIMENTS.md)  
- **Samples (view on GitHub):**
  - BPE (small dev): [`samples/small_bpe_ROMEO.txt`](samples/small_bpe_ROMEO.txt)
  - BPE (tiny):      [`samples/tiny_bpe_ROMEO.txt`](samples/tiny_bpe_ROMEO.txt)
  - Char (micro):    [`samples/micro_char_ROMEO.txt`](samples/micro_char_ROMEO.txt)

---

## 🛠 Tech Stack & Design

- **Language**: Python  
- **Core Libraries**: PyTorch 2.2.x (CPU), NumPy (<2), PyYAML, regex 
- **Directory Layout**:  
```bash
src/pocketllm/   # model, tokenizers, training & inference CLIs
├── bigram.py
├── bpe_tokenizer.py
├── infer.py
├── model.py
├── tokenizer.py
├── train.py             # char-level trainer
└── train_bpe_gpt.py     # BPE trainer
configs/         # YAML configs (small_bpe, tiny_bpe, tiny_char)
scripts/         # dataset download, helpers
samples/         # generated text (human-readable outputs)
tests/           # tiny unit tests (smoke, tokenizer, sampling)
```
 

---

## 🚀 Quick Start (CPU Friendly)

```bash
# Clone the repo
git clone https://github.com/nikkiakali/pocket-llm.git
cd pocket-llm

# Install dependencies
pip install -r requirements.txt

# Prepare data (e.g. download tiny Shakespeare)
bash scripts/download_tinyshakespeare.sh

# Train (fast dev BPE on CPU)
python -m pocketllm.train_bpe_gpt --config configs/small_bpe.yaml

# Train (char micro on CPU)
python -m pocketllm.train --config configs/tiny_char.yaml

# Train (tiny BPE — better quality; use GPU if available)
python -m pocketllm.train_bpe_gpt --config configs/tiny_bpe.yaml
```

# Inference
```bash
# BPE (small dev)
python -m pocketllm.infer --ckpt runs/small_bpe/ckpt_best.pt --prompt "ROMEO:"

# Char (micro)
python -m pocketllm.infer --ckpt runs/micro_char/ckpt_best.pt --prompt "ROMEO:"
```

## Tests
```bash
python -m pip install -e .
python -m pytest -q tests
```
