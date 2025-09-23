
def test_char_tokenizer_roundtrip():
    from pocketllm.tokenizer import CharTokenizer
    t = CharTokenizer("hello world")
    ids = t.encode("hello")
    assert t.decode(ids) == "hello"
    assert t.vocab_size >= 8
