"""Data loading for the XOR-cipher decryption task.

The dataset pairs each plaintext line (Brown corpus sentences) with its
ciphertext: every plaintext byte is XOR-ed with the repeating 8-byte ASCII
key ``"ANLP2026"`` and expanded to an 8-character bit string.

Two data paths (matching the ablation table):

- C1-C4 ("standard subword"): both sides are tokenized with *learned BPE*
  tokenizers (trained on the train split).  The ciphertext is a bit string,
  so its BPE tokens are learned bit patterns (NOT fixed-width 8-bit chunks).
- C5 (BLT, token-free): raw cipher bytes (0-255) and raw plaintext bytes are
  used directly; no linguistic vocabulary.
"""

import os
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, models, trainers

# --- token ids ----------------------------------------------------------------
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"
SPECIALS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
BYTE_PAD = 256  # padding id for raw-byte (BLT) sequences
CIPHER_KEY = b"ANLP2026"


# --- raw data ------------------------------------------------------------------
def xor_decrypt(plain_bytes: bytes, key: bytes = CIPHER_KEY) -> bytes:
    """C[i] = P[i] XOR K[i mod len(key)]."""
    return bytes(p ^ key[i % len(key)] for i, p in enumerate(plain_bytes))


def load_pairs(cipher_path: str, plain_path: str, verify: bool = True
               ) -> List[Tuple[str, str]]:
    """Load (cipher_bit_string, plain_text) line pairs; verify the XOR mapping."""
    plain_lines = open(plain_path, "rb").read().decode("latin1").splitlines()
    cipher_lines = open(cipher_path, "rb").read().decode("latin1").splitlines()
    assert len(plain_lines) == len(cipher_lines), "line count mismatch"
    pairs: List[Tuple[str, str]] = []
    for i, (p, c) in enumerate(zip(plain_lines, cipher_lines)):
        if verify:
            assert len(c) == 8 * len(p), f"line {i}: cipher length mismatch"
            cb = bytes(int(c[j * 8 : j * 8 + 8], 2) for j in range(len(p)))
            expected = xor_decrypt(p.encode("latin1"))
            assert cb == expected, f"line {i}: XOR mapping does not hold"
        pairs.append((c, p))
    return pairs


def cipher_bytes_of(cipher_bits: str) -> bytes:
    """Cipher bit string -> cipher bytes."""
    return bytes(int(cipher_bits[i * 8 : i * 8 + 8], 2)
                 for i in range(len(cipher_bits) // 8))


# --- BPE tokenizer ---------------------------------------------------------------
class BPETextTokenizer:
    """Learned BPE subword tokenizer (HuggingFace `tokenizers`)."""

    def __init__(self, path: str = None):
        self.path = path
        if path is not None:
            self.tok = Tokenizer.from_file(path)
        else:
            self.tok = Tokenizer(models.BPE())

    @classmethod
    def train(cls, texts, vocab_size: int, save_dir: str, name: str
              ) -> "BPETextTokenizer":
        tok = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIALS,
        )
        tok.train_from_iterator(iter(texts), trainer=trainer)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.json")
        tok.save(path)
        return cls(path)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids) -> str:
        ids = [i for i in ids if i not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID)]
        return self.tok.decode(list(ids))


# --- datasets -----------------------------------------------------------------------
class TokenizedCipherDataset(Dataset):
    """C1-C4: BPE token ids.  src = cipher BPE ids, tgt = BOS + plain BPE + EOS.

    Each item stores (src_ids, tgt_ids, ref_text) so references stay aligned
    with length filtering.
    """

    def __init__(self, pairs, cipher_tok: BPETextTokenizer,
                 plain_tok: BPETextTokenizer, max_src_len: int, max_tgt_len: int):
        self.items: List[Tuple[List[int], List[int], str]] = []
        for c, p in pairs:
            s = cipher_tok.encode(c)
            t = plain_tok.encode(p)
            if len(s) <= max_src_len and len(t) + 2 <= max_tgt_len:
                self.items.append((s, [BOS_ID] + t + [EOS_ID], p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, t, _ = self.items[i]
        return {"src": torch.tensor(s, dtype=torch.long),
                "tgt": torch.tensor(t, dtype=torch.long)}

    def refs(self) -> List[str]:
        return [item[2] for item in self.items]


class ByteCipherDataset(Dataset):
    """C5 (BLT): raw bytes.  src = cipher bytes, tgt = plaintext bytes (1:1)."""

    def __init__(self, pairs, max_len: int):
        self.items: List[Tuple[List[int], List[int], str]] = []
        for c, p in pairs:
            cb = cipher_bytes_of(c)
            if len(cb) <= max_len:
                self.items.append((list(cb), [ord(ch) for ch in p], p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, t, _ = self.items[i]
        return {"src": torch.tensor(s, dtype=torch.long),
                "tgt": torch.tensor(t, dtype=torch.long)}

    def refs(self) -> List[str]:
        return [item[2] for item in self.items]


# --- collate / loaders ------------------------------------------------------------------
def collate_tokenized(batch):
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True, padding_value=PAD_ID)
    return {
        "src": src,
        "src_mask": src != PAD_ID,
        "tgt": tgt,
        "tgt_mask": tgt != PAD_ID,
    }


def collate_bytes(batch):
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=BYTE_PAD)
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True, padding_value=BYTE_PAD)
    return {
        "src": src,
        "src_mask": src != BYTE_PAD,
        "tgt": tgt,
        "lengths": torch.tensor([len(b["src"]) for b in batch], dtype=torch.long),
    }


def make_dataloaders(split_pairs: Dict[str, list], args, tokenizers=None
                     ) -> Dict[str, DataLoader]:
    """split_pairs: {"train": [...], "val": [...], "test": [...]}.

    tokenizers=None -> raw-byte (BLT) datasets for C5.
    """
    cipher_tok, plain_tok = (None, None) if tokenizers is None else tokenizers
    loaders = {}
    collate = collate_bytes if tokenizers is None else collate_tokenized
    for name in ("train", "val", "test"):
        pairs = split_pairs[name]
        if tokenizers is None:
            d = ByteCipherDataset(pairs, max_len=args.max_src_len)
        else:
            d = TokenizedCipherDataset(
                pairs, cipher_tok, plain_tok,
                max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
            )
        loaders[name] = DataLoader(
            d,
            batch_size=args.batch_size if name == "train" else args.eval_batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            collate_fn=collate,
            drop_last=(name == "train" and args.drop_last),
            persistent_workers=args.num_workers > 0,
        )
    return loaders
