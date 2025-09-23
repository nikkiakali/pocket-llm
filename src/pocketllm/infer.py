"""
Inference for both char and BPE-based GPT models.
- Prefers model config saved in the checkpoint ("model_cfg").
- Otherwise, you can pass --d_model/--n_layer/--n_head/--ctx_len/--dropout.
- Works with checkpoints saved by train.py (char) and train_bpe_gpt.py (BPE).
"""
import torch
import torch.nn.functional as F
import argparse
from pocketllm.model import GPT
from pocketllm.bpe_tokenizer import BPETokenizer  # alias of SimpleBPE in your repo

__all__ = ["top_k_sampling", "main"]

def top_k_sampling(logits: torch.Tensor, k: int) -> int:
    """Sample a token id from the top-k logits (batch size = 1)."""
    k = max(1, min(int(k), logits.shape[-1]))
    top_vals, top_idx = torch.topk(logits, k, dim=-1)          # [1, k]
    probs = F.softmax(top_vals, dim=-1)                         # [1, k]
    sample = torch.multinomial(probs, num_samples=1)            # [1, 1]
    next_id = top_idx.gather(-1, sample).item()                 # int
    return next_id

def sample_next(logits, temperature: float = 1.0, top_k: int = 0) -> int:
    """Return next token id sampled from logits[1, V]."""
    logits = logits / max(temperature, 1e-8)
    if top_k > 0 and top_k < logits.size(-1):
        topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)   # [1, k], [1, k]
        probs = F.softmax(topk_vals, dim=-1)                      # [1, k]
        choice = torch.multinomial(probs, 1)                      # [1, 1]
        return topk_idx[0, choice.item()].item()
    probs = F.softmax(logits, dim=-1)                             # [1, V]
    return torch.multinomial(probs, 1).item()

def main(args):
    torch.manual_seed(args.seed)
    device = "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model_state = ckpt["model"]

    # --- Tokenizer setup ---
    # Char checkpoint: ckpt["vocab"] is a list[str] of chars
    # BPE checkpoint:  ckpt["vocab"] is a dict[int->str], and ckpt["merges"] is a list of pairs
    is_bpe = isinstance(ckpt.get("vocab"), dict) or ("merges" in ckpt)

    if is_bpe:
        print("BPE model detected.")
        id2token = ckpt["vocab"]
        # keys might be ints already, but normalize for safety
        id2token = {int(k): v for k, v in id2token.items()}
        merges = ckpt.get("merges", [])
        merges = [tuple(m) for m in merges]  # list[list[str,str]] -> list[tuple[str,str]]

        tok = BPETokenizer()
        tok.id2token = id2token
        tok.token2id = {t: i for i, t in id2token.items()}
        tok.merges = merges

        encode = tok.encode
        decode = tok.decode
        vocab_size = len(tok.id2token)

    else:
        print("Char-level model detected.")
        chars = ckpt["vocab"]
        vocab_size = len(chars)
        ctoi = {c: i for i, c in enumerate(chars)}
        itoc = {i: c for i, c in enumerate(chars)}
        encode = lambda s: [ctoi[c] for c in s if c in ctoi]
        decode = lambda l: "".join(itoc[int(i)] for i in l)

    # --- Model setup ---
    # Prefer model_cfg from checkpoint, else fall back to CLI flags.
    model_cfg = ckpt.get("model_cfg", {})
    d_model  = int(model_cfg.get("d_model",  args.d_model))
    n_layer  = int(model_cfg.get("n_layer",  args.n_layer))
    n_head   = int(model_cfg.get("n_head",   args.n_head))
    ctx_len  = int(model_cfg.get("ctx_len",  args.ctx_len))
    dropout  = float(model_cfg.get("dropout", args.dropout))

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        ctx_len=ctx_len,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # --- Generation ---
    print(f'Prompt: "{args.prompt}"')
    ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

    for _ in range(args.max_new_tokens):
        with torch.no_grad():
            logits, _ = model(ids[:, -model.ctx_len:])  # [1, T, V]
        next_id = sample_next(logits[:, -1, :], args.temperature, args.top_k)
        next_id_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
        ids = torch.cat([ids, next_id_t], dim=1)

    print("\n--- Output ---")
    print(decode(ids[0].tolist()))
    print("------------")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inference from a GPT model checkpoint.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint file.")
    p.add_argument("--prompt", default="ROMEO:", help="Starting prompt for generation.")
    p.add_argument("--max_new_tokens", type=int, default=300, help="Max new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    p.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    p.add_argument("--seed", type=int, default=1337, help="Random seed.")
    # Fallback model-shape flags (used if checkpoint lacks model_cfg)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--n_head",  type=int, default=4)
    p.add_argument("--ctx_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    a = p.parse_args()
    main(a)