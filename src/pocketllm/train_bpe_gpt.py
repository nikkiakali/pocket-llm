"""
Train a BPE-based GPT model (CPU-friendly, robust to YAML types).
"""
import os, yaml, torch, random, numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pocketllm.bpe_tokenizer import SimpleBPE as BPETokenizer
from pocketllm.model import GPT

class TextChunks(Dataset):
    def __init__(self, ids, ctx_len: int):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.ctx_len = int(ctx_len)
    def __len__(self):
        return max(0, len(self.ids) - self.ctx_len - 1)
    def __getitem__(self, i):
        x = self.ids[i:i+self.ctx_len]
        y = self.ids[i+1:i+self.ctx_len+1]
        return x, y

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def evaluate(model, dl, device: str):
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        _, vloss = model(xb, yb)
        tot += float(vloss.item())
        n += 1
    model.train()
    return tot / max(n, 1)

def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    # ---- robust config parsing (cast to correct types) ----
    seed      = int(cfg.get("seed", 1337))
    out_dir   = cfg.get("out_dir", "runs/tiny_bpe")
    vocab_sz  = int(cfg.get("vocab_size", 1000))

    data_file = cfg["data"]["train_file"]

    mcfg      = cfg["model"]
    d_model   = int(mcfg["d_model"])
    n_layer   = int(mcfg["n_layer"])
    n_head    = int(mcfg["n_head"])
    ctx_len   = int(mcfg["ctx_len"])
    dropout   = float(mcfg.get("dropout", 0.1))

    tcfg      = cfg["train"]
    batch_sz  = int(tcfg["batch_size"])
    lr        = float(tcfg["lr"])                 # <- was string sometimes
    epochs    = int(tcfg["epochs"])
    log_every = int(tcfg.get("log_every", 50))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    set_seed(seed)
    device = "cpu"  # keep CPU for stability on your iMac
    print(f"Using device: {device}")

    # ---- data & tokenizer ----
    text = open(data_file, "r", encoding="utf8", errors="ignore").read()
    tok = BPETokenizer()
    tok.train(text, vocab_sz)
    ids = tok.encode(text)
    print(f"BPE tokenizer trained with vocab size: {tok.vocab_size}")

    split = int(len(ids) * 0.9)
    train_ds = TextChunks(ids[:split], ctx_len)
    val_ds   = TextChunks(ids[split:], ctx_len)
    train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_sz, drop_last=True)

    # ---- model ----
    model = GPT(
        vocab_size=tok.vocab_size,
        d_model=d_model, n_layer=n_layer, n_head=n_head,
        ctx_len=ctx_len, dropout=dropout
    ).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # ---- training ----
    os.makedirs(out_dir, exist_ok=True)
    best = float("inf"); step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            opt.zero_grad(); loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1
            if step % log_every == 0:
                print(f"[train] epoch {epoch:02d} | step {step:05d} | loss {loss.item():.4f}")

        vloss = evaluate(model, val_dl, device)
        ppl = float(np.exp(vloss))
        print(f"[val] epoch {epoch:02d} | loss {vloss:.4f} | ppl {ppl:.2f}")

        if vloss < best:
            best = vloss
            ckpt = {
                "model": model.state_dict(),
                "merges": tok.merges,      # list[tuple[str,str]]
                "vocab": tok.id2token,     # dict[int,str]  <-- fixed name
            }
            torch.save(ckpt, f"{out_dir}/ckpt_best.pt")
            print("  -> new best checkpoint saved.")

    print(f"Done. Best val loss: {best:.4f}, perplexity: {np.exp(best):.2f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train a BPE-based GPT model.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    a = p.parse_args()
    main(a.config)