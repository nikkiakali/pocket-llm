"""
Minimal, educational Byte-Pair-Encoding tokenizer.
"""
import regex as re
from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def train(self, text, vocab_size, min_count=2):
        """
        Train BPE tokenizer on text.
        - text: raw text to train on.
        - vocab_size: number of tokens in the vocabulary.
        - min_count: min frequency for a pair to be merged.
        """
        # 1. Pre-tokenize the text into words. We use regex to split on whitespace
        # and punctuation, but keep them as separate tokens.
        word_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        words = [list(word) for word in word_pattern.findall(text)]

        # 2. Build initial vocabulary from single characters
        char_counts = defaultdict(int)
        for word in words:
            for char in word:
                char_counts[char] += 1
        
        vocab = sorted(char_counts.keys())
        self.token_to_id = {ch: i for i, ch in enumerate(vocab)}
        self.id_to_token = {i: ch for i, ch in enumerate(vocab)}

        # 3. Iteratively merge the most frequent pair of tokens
        num_merges = vocab_size - len(vocab)
        for i in range(num_merges):
            pair_counts = defaultdict(int)
            for word in words:
                for j in range(len(word) - 1):
                    pair_counts[(word[j], word[j+1])] += 1
            
            if not pair_counts: break

            # Find the most frequent pair, respecting min_count
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_count: break

            # Merge the best pair
            new_token_str = "".join(best_pair)
            new_token_id = len(self.token_to_id)
            self.token_to_id[new_token_str] = new_token_id
            self.id_to_token[new_token_id] = new_token_str
            self.merges[best_pair] = new_token_str

            # Update the words with the new merged token
            new_words = []
            for word in words:
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j+1]) == best_pair:
                        new_word.append(new_token_str)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_words.append(new_word)
            words = new_words
        
        # Store merges as a list of tuples for JSON serialization
        self.merges = {k: v for k, v in sorted(self.merges.items(), key=lambda item: self.token_to_id[item[1]])}

    def encode(self, text):
        """Encode text into a list of token IDs."""
        word_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        words = [list(word) for word in word_pattern.findall(text)]
        
        for pair, new_token in self.merges.items():
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

        ids = [self.token_to_id[token] for word in words for token in word]
        return ids

    def decode(self, ids):
        """Decode a list of token IDs into text."""
        return "".join([self.id_to_token[i] for i in ids])

    def save(self, file_prefix):
        """Save tokenizer to files."""
        with open(f"{file_prefix}.json", "w", encoding="utf-8") as f:
            import json
            json.dump({
                "token_to_id": self.token_to_id,
                "id_to_token": {int(k):v for k,v in self.id_to_token.items()},
                "merges": {' '.join(k):v for k,v in self.merges.items()},
            }, f, ensure_ascii=False, indent=2)

    def load(self, file_prefix):
        """Load tokenizer from files."""
        with open(f"{file_prefix}.json", "r", encoding="utf-8") as f:
            import json
            data = json.load(f)
            self.token_to_id = data["token_to_id"]
            self.id_to_token = {int(k):v for k,v in data["id_to_token"].items()}
            self.merges = {tuple(k.split(' ')):v for k,v in data["merges"].items()}
