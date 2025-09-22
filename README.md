# Pocket-LLM 🪶

[![CI](https://github.com/nikkiakali/pocket-llm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nikkiakali/pocket-llm/actions/workflows/ci.yml)

Tiny GPT-style LLM from scratch: tokenizer, transformer, training & inference.  
Built for experimentation, understanding core model internals, and pushing small models to be usable in lightweight settings.

---

## 🔍 What’s Working Now
- Custom tokenizer & text preprocessing  
- Transformer blocks implementation (self-attention, MLP, etc.)  
- Training loop + evaluation on toy datasets (e.g. Tiny Shakespeare)  
- Inference: generate text given prompt + checkpoint  

---

## 🗺 Roadmap & Future Work
- Add Byte Pair Encoding (BPE) or similar tokenization  
- Mixed precision / Automatic Mixed Precision (AMP) for faster training  
- Fine-tuning support (LoRA or similar)  
- Web demo / interactive interface  
- Ablation studies: vary context length, number of layers/heads, dropout  
- Better tooling: logging, visualization of training curves  

---

## 🛠 Tech Stack & Design

- **Language**: Python  
- **Core Libraries**: PyTorch (or your DL framework), NumPy, etc.  
- **Structure**:  
  - `tokenizer/` or `src/tokenizer` → tokenization logic  
  - `transformer/` or `src/model` → model architecture  
  - `scripts/` or `notebooks/` → training / data prep / experiments  
  - `configs/` → configuration files for experiments/datasets  

---

## 🚀 Quick Start (Early Demo)

> 💡 *This is experimental — expect rough edges*

```bash
# Clone the repo
git clone https://github.com/nikkiakali/pocket-llm.git
cd pocket-llm

# Install dependencies
pip install -r requirements.txt

# Prepare data (e.g. download tiny Shakespeare)
bash scripts/download_tinyshakespeare.sh

# Train model with default config
python -m pocketllm.train --config configs/tiny_char.yaml

# Inference
python -m pocketllm.infer --ckpt runs/tiny_char/ckpt_best.pt --prompt "To be, or not to be…"
