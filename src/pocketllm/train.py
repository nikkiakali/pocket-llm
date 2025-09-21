import os, yaml, torch, random, numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pocketllm.tokenizer import CharTokenizer
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        _, vloss = model(xb, yb)
        tot += vloss.item() * xb.size(0)
        n += xb.size(0)
    model.train()
    return tot / n

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg.get("seed", 1337))

    text = open(cfg["data"]["train_file"], "r", encoding="utf8").read()
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    n_train = int(len(ids) * 0.9)
    train_ds = TextChunks(ids[:n_train], cfg["model"]["ctx_len"])
    val_ds = TextChunks(ids[n_train:], cfg["model"]["ctx_len"])
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = GPT(tok.vocab_size, **cfg["model"]).to(device)
    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("wd", 0.01))

    model.train()
    best_loss, step = 1e9, 0
    os.makedirs(cfg["out_dir"], exist_ok=True)
    for epoch in range(cfg["train"]["epochs"]):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
            opt.step()
            step += 1
            if step % cfg["train"].get("log_every", 100) == 0:
                print(f"step {step} | loss {loss.item():.4f}")

        val_loss = evaluate(model, val_dl, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict(), "vocab": tok.chars}, f'{cfg["out_dir"]}/ckpt_best.pt')

    print(f"Done. Best val loss: {best_loss:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.config)