What it is: 1–2 lines on from-scratch LLM with clean architecture, reproducible results.

Features: tokenizer, GPT blocks, training loop, evals, inference, ablations.

Quickstart:

pip install -e .
bash scripts/download_tinyshakespeare.sh
python -m pocketllm.train --config configs/tiny_char.yaml
python -m pocketllm.infer --ckpt runs/tiny_char/ckpt_best.pt --prompt "To be, or not to be"


Results: show train/val loss curves, final perplexity.

Design: diagram of data flow + config.

Ablations: context length, heads, layers, dropout.

Roadmap: BPE, AMP, LoRA finetune, small web demo.

Cite & license.