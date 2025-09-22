| Run        | Data Size | Config                                                | Device | Best Val Loss | Val PPL | Notes |
|------------|-----------|--------------------------------------------------------|--------|---------------|---------|-------|
| bigram     | full      | alpha=1.0                                             | CPU    | 2.48          | 11.96   | baseline |
| micro_char | ~200k     | d=96, nL=2, H=4, ctx=128, 1 epoch                     | CPU    | 1.9020        | 6.70    | beats bigram |
| tiny_bpe   | ~200k     | vsz=1000, d=128, nL=3, H=4, ctx=128, 2 epochs         | CPU    | 2.7025        | 14.92   | tokenizer differs from char; not directly comparable; sample: [ROMEO](samples/tiny_bpe_ROMEO.txt) |
| small_bpe  | ~200k     | vsz=512, d=96, nL=2, H=4, ctx=96, epochs=3 (cap 800)  | CPU    | 2.7693        | 15.95   | sample: [ROMEO](samples/small_bpe_ROMEO.txt) |
