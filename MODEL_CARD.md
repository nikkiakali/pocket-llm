# Pocket-LLM Model Card

**What**: Minimal GPTs trained on Tiny Shakespeare with two tokenizers: char and BPE.  
**Why**: Educational, CPU-friendly training, clear engineering hygiene (tests/CI/configs).

## Model Details
- Architectures: Bigram baseline, Char-GPT, BPE-GPT
- Example configs: see `configs/` (`micro_char`, `tiny_bpe`, `small_bpe`)
- Framework: PyTorch 2.2.x (CPU)

## Training Data
- Tiny Shakespeare (public domain). See `data/tinyshakespeare.txt`.
- Dev runs use a 200k-char slice for speed.

## Intended Use
- Learning and demonstration.
- **Not** for production or safety-critical use.

## Metrics
- Perplexity (tokenizer-specific; not directly comparable across tokenizers).
- Qualitative samples in `samples/`.

## How to Reproduce
```bash
python -m pocketllm.train --config configs/tiny_char.yaml
python -m pocketllm.train_bpe_gpt --config configs/tiny_bpe.yaml
python -m pocketllm.infer --ckpt runs/tiny_bpe/ckpt_best.pt --prompt "ROMEO:"

