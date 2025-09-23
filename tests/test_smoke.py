def test_import_and_forward_shape():
    import torch
    from pocketllm.model import GPT
    m = GPT(vocab_size=32, d_model=32, n_layer=1, n_head=4, ctx_len=8, dropout=0.0)
    x = torch.zeros(2, 8, dtype=torch.long)
    logits, loss = m(x, x)
    assert logits.shape == (2, 8, 32)
    assert loss.ndim == 0
