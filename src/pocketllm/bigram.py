# src/pocketllm/bigram.py
import argparse, numpy as np, math, os, time
from typing import List
from pocketllm.tokenizer import CharTokenizer

class BigramLM:
    def __init__(self, vocab_size: int, alpha: float = 1.0):
        self.vocab_size = vocab_size
        self.alpha = float(alpha)
        self.probs = np.full((vocab_size, vocab_size), 1.0 / vocab_size, dtype=np.float64)

    def fit(self, ids: List[int]):
        V = self.vocab_size
        counts = np.full((V, V), self.alpha, dtype=np.float64)  # Laplace smoothing
        for a, b in zip(ids[:-1], ids[1:]):
            counts[a, b] += 1.0
        self.probs = counts / counts.sum(axis=1, keepdims=True)

    def nll(self, ids: List[int]) -> float:
        rows, cols = np.array(ids[:-1]), np.array(ids[1:])
        p = np.clip(self.probs[rows, cols], 1e-12, 1.0)
        return float(-np.mean(np.log(p)))

    def perplexity(self, ids: List[int]) -> float:
        return math.exp(self.nll(ids))

    def _temp(self, p: np.ndarray, temp: float) -> np.ndarray:
        if temp == 1.0: return p
        q = np.power(p, 1.0 / temp); q /= q.sum()
        return q

    def generate(self, start_id: int, max_new_tokens: int, temperature: float = 1.0) -> List[int]:
        out = [int(start_id)]
        for _ in range(max_new_tokens):
            p = self._temp(self.probs[out[-1]], temperature)
            next_id = np.random.choice(self.vocab_size, p=p)
            out.append(int(next_id))
        return out

def main():
    ap = argparse.ArgumentParser(description="NumPy Bigram LM (train/eval/sample)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    ap.add_argument("--val_frac", type=float, default=0.1, help="fraction for validation split")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save_dir", default="runs/bigram")
    args = ap.parse_args()

    np.random.seed(args.seed)
    text = open(args.data, "r", encoding="utf8").read()
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    # train/val split
    split = int(len(ids) * (1.0 - args.val_frac))
    train_ids, val_ids = ids[:split], ids[split:]

    model = BigramLM(tok.vocab_size, alpha=args.alpha)
    model.fit(train_ids)

    train_ppl = model.perplexity(train_ids)
    val_ppl = model.perplexity(val_ids) if len(val_ids) > 10 else float("nan")
    print(f"Bigram perplexity  train={train_ppl:.2f}  val={val_ppl:.2f}  (alpha={args.alpha})")

    # sample
    start = np.random.randint(0, tok.vocab_size)
    gen_ids = model.generate(start, max_new_tokens=args.samples, temperature=args.temperature)
    print("Sample:\n" + tok.decode(gen_ids))

    # save artifacts
    ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.save_dir, exist_ok=True)
    np.savez(os.path.join(args.save_dir, f"bigram_{ts}.npz"),
             probs=model.probs, alpha=args.alpha, vocab=np.array(tok.chars))
    with open(os.path.join(args.save_dir, f"metrics_{ts}.txt"), "w") as f:
        f.write(f"train_ppl={train_ppl:.4f}\nval_ppl={val_ppl:.4f}\nalpha={args.alpha}\n")

if __name__ == "__main__":
    main()