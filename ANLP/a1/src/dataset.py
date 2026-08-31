"""Data loading for the XOR-cipher decryption task.

The dataset pairs each plaintext line (Brown corpus sentences) with its
ciphertext: every plaintext byte is XOR-ed with the repeating 8-byte ASCII
key ``"ANLP2026"`` and expanded to an 8-character bit string.

Two data paths (matching the ablation table):

- C1-C4 ("standard subword"): both sides are tokenized with *learned BPE*
  tokenizers (trained on the train split).  The ciphertext bit string is first
  converted to its raw bytes (one char per byte, latin-1 codepoints 0-255) and
  BPE is learned over that byte alphabet, so cipher tokens are learned
  variable-length byte sequences (NOT fixed-width 8-bit chunks).
- C5 (BLT, token-free): raw cipher bytes (0-255) and raw plaintext bytes are
  used directly; no linguistic vocabulary.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, models, trainers

# --- token ids ----------------------------------------------------------------
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"
SPECIALS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
BYTE_PAD = 256   # padding id for raw-byte (BLT) sequences
BYTE_BOS, BYTE_EOS, BYTE_VOCAB = 257, 258, 259  # byte-target specials (C1-C4)
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


def cipher_bitstring_to_byte_str(cipher_bits: str) -> str:
    """Cipher bit string -> str with one char per cipher byte (latin-1 0-255).

    This is the input space of the cipher-side BPE tokenizer: BPE over the
    raw cipher bytes, i.e. learned subword sequences of byte values.
    """
    n = len(cipher_bits) // 8
    v = int(cipher_bits, 2) if n else 0
    return v.to_bytes(n, "big").decode("latin-1")


# --- BPE tokenizer ---------------------------------------------------------------
class BPETextTokenizer:
    """Learned BPE subword tokenizer (HuggingFace `tokenizers`)."""

    def __init__(self, path: str = None):
        self.path = path
        if path is not None:
            self.tok = Tokenizer.from_file(path)
        else:
            self.tok = Tokenizer(models.BPE())
        # id -> token lookup.  We decode by table lookup + join instead of
        # ``tok.decode(ids)`` because the library's batch decode re-inserts a
        # space between tokens (it assumes whitespace pre-tokenization), which
        # corrupts the text.  Per-id lookup is exact and fast.
        self._id2tok = {tid: t for t, tid in self.tok.get_vocab().items()}

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
        """Exact decode: skip special ids, concatenate token strings."""
        specials = (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
        return "".join(self._id2tok[i] for i in ids if i not in specials)


# --- datasets -----------------------------------------------------------------------
class PhaseByteBPE:
    """BPE over phase-annotated ciphertext bytes.

    Each ciphertext byte i is represented as the single symbol (b, i mod 8),
    i.e. its byte value together with its position in the repeating 8-byte
    key.  The key position is a deterministic function of the byte index (no
    extra information), and with the key known, decrypting a symbol is the
    per-symbol lookup ``b XOR K[i mod 8]``.  BPE is then *learned* over this
    2048-symbol alphabet (8000 = 2048 base symbols + merges), producing
    variable-length tokens -- a learned subword scheme, not fixed-width
    chunking.
    """

    N_PHASES = 8
    BASE_VOCAB = 256 * 8

    @staticmethod
    def cipher_symbols(cipher_bits: str) -> str:
        """cipher bit string -> (byte, phase) symbol string.

        Symbol for byte i is the codepoint b*8 + (i mod 8) in [0, 2047].
        """
        cb = cipher_bytes_of(cipher_bits)
        return "".join(chr(b * 8 + (i % 8)) for i, b in enumerate(cb))

    def __init__(self, path: str = None):
        self.path = path
        if path is not None:
            self.tok = Tokenizer.from_file(path)
        else:
            self.tok = None

    @classmethod
    def train(cls, cipher_bits_list, vocab_size: int, save_dir: str, name: str,
              min_frequency: int = 2) -> "PhaseByteBPE":
        docs = [cls.cipher_symbols(c) for c in cipher_bits_list]
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency,
            special_tokens=SPECIALS)
        tok.train_from_iterator(iter(docs), trainer=trainer)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, name)
        tok.save(path)
        self = cls(path)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.tok.get_vocab())

    def encode(self, cipher_bits: str) -> List[int]:
        return self.tok.encode(self.cipher_symbols(cipher_bits),
                               add_special_tokens=False).ids

    def decode_bytes(self, ids) -> bytes:
        """Token ids -> cipher bytes (diagnostics only)."""
        s = self.tok.decode(ids)
        return bytes(v // 8 for v in (ord(ch) for ch in s))


class TokenizedCipherDataset(Dataset):
    """C1-C4: BPE token ids.  src = cipher BPE ids, tgt = BOS + plain BPE + EOS.

    Each item stores (src_ids, tgt_ids, ref_text) so references stay aligned
    with length filtering.
    """

    def __init__(self, pairs, cipher_tok: BPETextTokenizer,
                 plain_tok: BPETextTokenizer, max_src_len: int, max_tgt_len: int):
        self.items: List[Tuple[List[int], List[int], str]] = []
        for c, p in pairs:
            s = cipher_tok.encode(cipher_bitstring_to_byte_str(c))
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


class ByteTargetCipherDataset(Dataset):
    """C1-C4 byte-target variant: src = learned subword cipher ids, tgt = BYTES.

    Target position i (after BOS) corresponds exactly to plaintext byte i, so
    there is no target-side resegmentation to learn.  The ciphertext side is
    tokenized with a learned subword scheme (BPE): either over raw cipher
    bytes, or over phase-annotated cipher bytes (PhaseByteBPE).
    """

    def __init__(self, pairs, cipher_tok,
                 max_src_len: int, max_tgt_len: int):
        self.items: List[Tuple[List[int], List[int], str]] = []
        for c, p in pairs:
            s = cipher_tok.encode(c)
            pb = p.encode("latin1")
            if len(s) <= max_src_len and len(pb) + 2 <= max_tgt_len:
                self.items.append(
                    (s, [BYTE_BOS] + list(pb) + [BYTE_EOS], p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, t, _ = self.items[i]
        return {"src": torch.tensor(s, dtype=torch.long),
                "tgt": torch.tensor(t, dtype=torch.long),
                "idx": i}

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
class LengthBatchSampler(torch.utils.data.BatchSampler):
    """Batches of similar target length (much less padding waste than random
    batches, which pad to the batch maximum).

    Sequences are sorted by length, cut into length chunks, and the chunk
    order is re-shuffled every epoch (set_epoch) so consecutive epochs see a
    different global order while each batch stays length-homogeneous.
    """

    def __init__(self, items, batch_size: int, seed: int = 42, epoch: int = 0,
                 n_chunks: int = 50):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.n_chunks = n_chunks
        self._build(items)

    def _build(self, items):
        lens = np.array([len(t) for _, t, _ in items])
        order = np.argsort(lens, kind="stable")
        n = len(order)
        n_chunks = max(1, min(self.n_chunks, n // self.batch_size))
        bounds = np.linspace(0, n, n_chunks + 1).astype(int)
        chunks = [order[bounds[i]:bounds[i + 1]] for i in range(n_chunks)]
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        rng.shuffle(chunks)
        seq = np.concatenate(chunks)
        self.batches = [seq[i:i + self.batch_size].tolist()
                        for i in range(0, len(seq), self.batch_size)]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_tokenized(batch):
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True, padding_value=PAD_ID)
    return {
        "src": src,
        "src_mask": src != PAD_ID,
        "tgt": tgt,
        "tgt_mask": tgt != PAD_ID,
    }


def collate_byte_target(batch):
    """src padded with BPE PAD (0), tgt padded with BYTE_PAD (256).

    ``indices``: dataset row ids, used by scheduled sampling to fetch each
    sample's cached self-predicted prefix.
    """
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True, padding_value=BYTE_PAD)
    out = {
        "src": src,
        "src_mask": src != PAD_ID,
        "tgt": tgt,
        "tgt_mask": tgt != BYTE_PAD,
    }
    if "idx" in batch[0]:
        out["indices"] = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    return out


def collate_bytes(batch):
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=BYTE_PAD)
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True, padding_value=BYTE_PAD)
    return {
        "src": src,
        "src_mask": src != BYTE_PAD,
        "tgt": tgt,
        "lengths": torch.tensor([len(b["src"]) for b in batch], dtype=torch.long),
    }


def make_dataloaders(split_pairs: Dict[str, list], args, tokenizers=None,
                     target_bytes: bool = False
                     ) -> Dict[str, DataLoader]:
    """split_pairs: {"train": [...], "val": [...], "test": [...]}.

    tokenizers=None -> raw-byte (BLT) datasets for C5.
    target_bytes=True -> C1-C4 byte-target datasets (cipher BPE -> plain bytes).
    """
    cipher_tok, plain_tok = (None, None) if tokenizers is None else tokenizers
    loaders = {}
    if tokenizers is None:
        collate = collate_bytes
    elif target_bytes:
        collate = collate_byte_target
    else:
        collate = collate_tokenized
    for name in ("train", "val", "test"):
        pairs = split_pairs[name]
        if tokenizers is None:
            d = ByteCipherDataset(pairs, max_len=args.max_src_len)
        elif target_bytes:
            d = ByteTargetCipherDataset(
                pairs, cipher_tok,
                max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
            )
        else:
            d = TokenizedCipherDataset(
                pairs, cipher_tok, plain_tok,
                max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
            )
        if target_bytes and getattr(args, "length_bucketing", False):
            # length-homogeneous batches (byte targets have heavy length
            # variance: ~600 bytes median, ~2600 max)
            bs = args.batch_size if name == "train" else args.eval_batch_size
            sampler = LengthBatchSampler(d.items, bs, seed=getattr(args, "seed", 42),
                                         epoch=0)
            loaders[name] = DataLoader(
                d, batch_sampler=sampler, num_workers=args.num_workers,
                collate_fn=collate, persistent_workers=args.num_workers > 0)
        else:
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
