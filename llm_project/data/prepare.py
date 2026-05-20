"""
data/prepare.py
===============
Tokenizer and PyTorch Dataset for the LLM project.

Two tokenisers are provided so you can swap them without touching the model:
  - CharTokenizer  : character-level, zero dependencies, great for learning
  - BPETokenizer   : byte-pair encoding via the `tokenizers` library (optional)
                     produces a smaller vocabulary and better sub-word representations

Both expose the same interface:  .encode(text) → list[int]
                                  .decode(ids)  → str
                                  .vocab_size   → int

TextDataset wraps any encoded token list into (x, y) training pairs
where y is x shifted one position to the right (next-token prediction).

Usage:
    from data.prepare import CharTokenizer, TextDataset, load_corpus
    tokenizer = CharTokenizer().fit(text)
    dataset   = TextDataset(tokenizer.encode(text), context_len=128)
"""

import os
import pickle
import json
import re
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# Built-in demo corpus (used when no file is given)

DEMO_CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
""" * 40

# 1. CORPUS LOADER
def load_corpus(path: Optional[str] = None) -> str:
    """
    Load a text corpus from a .txt file.
    Falls back to the built-in Shakespeare demo if path is None or missing.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"[Data] Loaded corpus: {len(text):,} chars from {path}")
    else:
        text = DEMO_CORPUS
        print(f"[Data] Using demo corpus: {len(text):,} chars (Shakespeare excerpt)")
    return text



# 2. CHARACTER-LEVEL TOKENIZER
class CharTokenizer:
    """
    Maps every unique character to an integer index.

    Advantages:
      - Zero external dependencies
      - Tiny vocab (typically 50–100 for English text)
      - Transparent — you can read the vocabulary by eye

    Disadvantages:
      - Long sequences (1 char = 1 token → slow for large text)
      - No sub-word sharing between similar words

    Typical use: Phase 1 learning, fast iteration on small corpora.
    """

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> "CharTokenizer":
        """Build vocab from a string. Must be called before encode/decode."""
        chars = sorted(set(text))
        self.char2idx  = {c: i for i, c in enumerate(chars)}
        self.idx2char  = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(chars)
        print(f"[CharTokenizer] vocab_size={self.vocab_size}")
        return self

    def encode(self, text: str) -> list[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2char.get(i, "?") for i in ids)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"char2idx": self.char2idx, "idx2char": self.idx2char}, f)
        print(f"[CharTokenizer] saved → {path}")

    def load(self, path: str) -> "CharTokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.char2idx   = data["char2idx"]
        self.idx2char   = data["idx2char"]
        self.vocab_size = len(self.char2idx)
        print(f"[CharTokenizer] loaded ← {path}  vocab_size={self.vocab_size}")
        return self

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"



# 3. SIMPLE BPE TOKENIZER
class SimpleBPETokenizer:
    """
    Minimal Byte-Pair Encoding tokenizer — no external dependencies.

    BPE works by iteratively merging the most frequent adjacent byte/char
    pair in the corpus until the vocabulary reaches the target size.

    This implementation is educational (not optimised for large corpora).
    For production use, swap in HuggingFace tokenizers.Tokenizer with BPE.

    Parameters
    ----------
    vocab_size : target vocabulary size (must be >= number of unique chars)
    """

    def __init__(self, vocab_size: int = 512):
        self.target_vocab = vocab_size
        self.merges: list[tuple[str, str]] = []   # ordered merge rules
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.vocab_size = 0

    # ── Training ─────────────────────────────────────────────────────────────
    def fit(self, text: str) -> "SimpleBPETokenizer":
        """Learn BPE merge rules from a corpus string."""
        # Start with character-level tokens (add end-of-word marker)
        words = text.split()
        vocab_freq: Counter = Counter()
        for word in words:
            # Represent each word as tuple of chars + end marker </w>
            token = tuple(list(word) + ["</w>"])
            vocab_freq[token] += 1

        # Base vocabulary: all unique characters + </w>
        base_chars = set(c for word in vocab_freq for c in word)
        token2id = {c: i for i, c in enumerate(sorted(base_chars))}

        num_merges = self.target_vocab - len(token2id)
        print(f"[BPE] base vocab={len(token2id)}, learning {num_merges} merges...")

        for merge_idx in range(num_merges):
            # Count all adjacent pairs
            pairs: Counter = Counter()
            for word, freq in vocab_freq.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq

            if not pairs:
                break

            # Find best pair
            best = max(pairs, key=pairs.get)
            self.merges.append(best)

            # Merge best pair everywhere
            merged = best[0] + best[1]
            new_vocab_freq: Counter = Counter()
            for word, freq in vocab_freq.items():
                new_word = _merge_pair(word, best, merged)
                new_vocab_freq[new_word] += freq
            vocab_freq = new_vocab_freq

            # Add merged token to vocab
            token2id[merged] = len(token2id)

            if (merge_idx + 1) % 100 == 0:
                print(f"[BPE] merge {merge_idx + 1}/{num_merges}  vocab={len(token2id)}")

        self.vocab     = token2id
        self.inv_vocab = {v: k for k, v in token2id.items()}
        self.vocab_size = len(token2id)
        print(f"[BPE] done. vocab_size={self.vocab_size}")
        return self

    def encode(self, text: str) -> list[int]:
        """Encode a string to token IDs using the learned merge rules."""
        ids = []
        for word in text.split():
            tokens = list(word) + ["</w>"]
            # Apply merge rules in order
            for (a, b) in self.merges:
                tokens = _merge_pair(tuple(tokens), (a, b), a + b)
                tokens = list(tokens)
            for tok in tokens:
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inv_vocab.get(i, "?") for i in ids]
        text   = "".join(tokens).replace("</w>", " ")
        return text.strip()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)
        print(f"[BPE] saved → {path}")

    def load(self, path: str) -> "SimpleBPETokenizer":
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab      = data["vocab"]
        self.merges     = [tuple(m) for m in data["merges"]]
        self.inv_vocab  = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"[BPE] loaded ← {path}  vocab_size={self.vocab_size}")
        return self

    def __repr__(self):
        return f"SimpleBPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


def _merge_pair(word: tuple, pair: tuple, merged: str) -> tuple:
    """Replace all occurrences of pair in word tuple with merged string."""
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)

# 4. PYTORCH DATASET
class TextDataset(Dataset):
    """
    Sliding-window next-token-prediction dataset.

    Each sample:
      x : token_ids[i   : i + context_len]   — input sequence
      y : token_ids[i+1 : i + context_len+1] — target (shifted right by 1)

    The model learns: given x[0..t], predict x[t+1] at every position t.
    This is the core objective of causal language modelling (CLM).

    Parameters
    ----------
    token_ids   : flat list of integer token IDs for the entire split
    context_len : number of tokens per training sample (sequence length)
    """

    def __init__(self, token_ids: list[int], context_len: int):
        self.data        = torch.tensor(token_ids, dtype=torch.long)
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len - 1)

    def __getitem__(self, idx: int):
        x = self.data[idx           : idx + self.context_len]
        y = self.data[idx + 1       : idx + self.context_len + 1]
        return x, y



# 5. CONVENIENCE: BUILD DATALOADERS
def build_dataloaders(
    token_ids: list[int],
    context_len: int,
    batch_size: int,
    train_split: float = 0.9,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Split token_ids into train/val and return two DataLoaders.

    Parameters
    ----------
    token_ids   : full encoded corpus
    context_len : sequence length per sample
    batch_size  : mini-batch size
    train_split : fraction of tokens used for training (rest → validation)
    num_workers : DataLoader worker processes (0 = main process, safe on Windows/Mac)

    Returns
    -------
    train_loader, val_loader
    """
    n          = int(len(token_ids) * train_split)
    train_ids  = token_ids[:n]
    val_ids    = token_ids[n:]

    train_ds   = TextDataset(train_ids, context_len)
    val_ds     = TextDataset(val_ids,   context_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers, pin_memory=True,
    )

    print(f"[Data] train samples={len(train_ds):,}  val samples={len(val_ds):,}")
    print(f"[Data] train batches={len(train_loader):,}  val batches={len(val_loader):,}")
    return train_loader, val_loader

