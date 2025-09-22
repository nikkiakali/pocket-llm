"""
Inference for both char and BPE-based GPT models.
"""
import torch
import torch.nn.functional as F
from pocketllm.model import GPT
from pocketllm.bpe_tokenizer import BPETokenizer

def top_k_sampling(logits, k):
    """Sample from the top k logits."""
    top_k_logits, top_k_indices = torch.topk(logits, k)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(top_k_probs, num_samples=1)
    return top_k_indices[0, sampled_index.item()]

def main(args):
    torch.manual_seed(args.seed)
    device = "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model_state = ckpt["model"]
    
    # --- Tokenizer setup ---
    is_bpe = "merges" in ckpt
    if is_bpe:
        print("BPE model detected.")
        vocab = ckpt["vocab"]
        merges = ckpt["merges"]
        tok = BPETokenizer()
        tok.id_to_token = {int(k): v for k, v in vocab.items()}
        tok.token_to_id = {v: int(k) for k, v in vocab.items()}
        tok.merges = {tuple(m): "".join(m) for m in merges}
        encode = tok.encode
        decode = tok.decode
        vocab_size = len(vocab)
    else:
        print("Char-level model detected.")
        chars = ckpt["vocab"]
        vocab_size = len(chars)
        ctoi = {c: i for i, c in enumerate(chars)}
        itoc = {i: c for i, c in enumerate(chars)}
        encode = lambda s: [ctoi[c] for c in s]
        decode = lambda l: "".join([itoc[i] for i in l])

    # --- Model setup ---
    model_args = {
        'vocab_size': vocab_size,
        'd_model': model_state['tok_emb.weight'].shape[1],
        'n_layer': len([k for k in model_state if k.endswith('.attn.proj.weight')]),
        'n_head': model_state['blocks.0.attn.qkv.weight'].shape[0] // (model_state['tok_emb.weight'].shape[1] // len([k for k in model_state if k.endswith('.attn.proj.weight')])),
        'ctx_len': model_state['pos_emb.weight'].shape[0],
    }
    model = GPT(**model_args).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # --- Generation ---
    print(f"Prompt: \"{args.prompt}\"")
    ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    
    print("--- Output ---")
    for _ in range(args.max_new_tokens):
        with torch.no_grad():
            logits, _ = model(ids[:, -model.ctx_len:])
        
        # Apply temperature
        logits = logits[:, -1, :] / args.temperature

        # Apply top-k sampling
        if args.top_k > 0:
            next_id = top_k_sampling(logits, args.top_k)
            next_id = torch.tensor([[next_id]], device=device)
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)
        print(decode([next_id.item()]), end="", flush=True)
    print("\n------------")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inference from a GPT model checkpoint.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint file.")
    p.add_argument("--prompt", default="ROMEO:", help="Starting prompt for generation.")
    p.add_argument("--max_new_tokens", type=int, default=300, help="Max new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    p.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    p.add_argument("--seed", type=int, default=1337, help="Random seed.")
    a = p.parse_args()
    main(a)
