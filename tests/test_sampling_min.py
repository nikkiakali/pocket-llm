
import torch
from pocketllm.infer import top_k_sampling

def test_top_k_sampling_returns_index():
    logits = torch.zeros(1, 10)
    logits[0, 3] = 5.0
    logits[0, 7] = 4.0
    idx = top_k_sampling(logits, k=2)
    assert int(idx) in (3, 7)
