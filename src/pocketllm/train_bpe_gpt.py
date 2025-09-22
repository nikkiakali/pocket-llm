"""
Train a BPE-based GPT model.
"""
import os, yaml, torch, random, numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pocketllm.bpe_tokenizer import BPETokenizer
from pocketllm.model import GPT

class TextChunks(Dataset):
    def __init__(self, ids, ctx_len):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.ctx_len = ctx_len
    def __len__(self): return len(self.ids) - self.ctx_len
    def __getitem__(self, i):
        x = self.ids[i:i+self.ctx_len]
        y = self.ids[i+1:i+self.ctx_len+1]
        return x, y

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    tot_loss, tot_count = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        _, vloss = model(xb, yb)
        tot_loss += vloss.item() * xb.size(0)
        tot_count += xb.size(0)
    model.train()
    return tot_loss / tot_count

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg.get("seed", 1337))
    device = "cpu" # Force CPU as requested
    print(f"Using device: {device}")

    # 1. Load data and train tokenizer
    text = open(cfg["data"]["train_file"], "r", encoding="utf8").read()
    tok = BPETokenizer()
    tok.train(text, cfg["vocab_size"])
    ids = tok.encode(text)
    print(f"BPE tokenizer trained with vocab size: {tok.vocab_size}")

    # 2. Create datasets and dataloaders
    n_train = int(len(ids) * 0.9)
    train_ds = TextChunks(ids[:n_train], cfg["model"]["ctx_len"])
    val_ds = TextChunks(ids[n_train:], cfg["model"]["ctx_len"])
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    # 3. Initialize model and optimizer
    model_args = cfg["model"]
    model_args['vocab_size'] = tok.vocab_size
    model = GPT(**model_args).to(device)
    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"])

    # 4. Training loop
    model.train()
    best_loss, step = 1e9, 0
    os.makedirs(cfg["out_dir"], exist_ok=True)
    for epoch in range(cfg["train"]["epochs"]):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            step += 1
            if step % cfg["train"]["log_every"] == 0:
                print(f"[train] epoch {epoch+1:02d} | step {step:05d} | loss {loss.item():.4f}")

        val_loss = evaluate(model, val_dl, device)
        perplexity = np.exp(val_loss)
        print(f"[val] epoch {epoch+1:02d} | loss {val_loss:.4f} | ppl {perplexity:.2f}")

        if val_loss < best_loss:
            best_loss = val_loss
            # Save checkpoint with model, merges, and vocab
            merges_list = [list(p) for p in tok.merges.keys()]
            ckpt = {
                "model": model.state_dict(),
                "merges": merges_list,
                "vocab": {i: tok.id_to_token[i] for i in range(tok.vocab_size)},
            }
            torch.save(ckpt, f'{cfg["out_dir"]}/ckpt_best.pt')
            print(f"  -> new best checkpoint saved.")

    print(f"Done. Best val loss: {best_loss:.4f}, perplexity: {np.exp(best_loss):.2f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train a BPE-based GPT model.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    a = p.parse_args()
    main(a.config)
