
| Run           | Data Size | Config                 | Device | Best Val Loss | Val PPL | Notes |
|---------------|-----------|------------------------|--------|---------------|---------|-------|
| bigram        | full      | alpha=1.0              | CPU    | ~2.48         | ~12.0   | baseline
| micro_char    | ~200k     | d_model=96, nL=2, ctx=128, 1 epoch | CPU    | 1.9020        | 6.70    | beats bigram
| tiny_bpe      | ~200k     | vsz=1000, d_model=128, nL=3, ctx=128, 2 epochs | CPU | <NUMBER> | <NUMBER> | BPE beats char micro.

