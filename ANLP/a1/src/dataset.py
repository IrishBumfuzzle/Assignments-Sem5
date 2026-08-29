"""Dataset and DataLoader pipelines for tokenized and token-free (BLT) experiments."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def pack_bitstring(text: str) -> str:
    """If text is a binary string composed solely of '0' and '1' and divisible by 8, pack to bytes/chars."""
    if len(text) >= 8 and len(text) % 8 == 0 and all(c in "01" for c in text):
        return "".join(chr(int(text[i : i + 8], 2)) for i in range(0, len(text), 8))
    return text


from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

class SubwordTokenizer:
    """Subword-level tokenizer using ByteLevel BPE with special tokens."""

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    def __init__(self, texts: Optional[List[str]] = None, vocab_size: int = 8000):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        if texts is not None:
            packed_texts = [pack_bitstring(t) for t in texts]
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )
            self.tokenizer.train_from_iterator(packed_texts, trainer)

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save(self, path: Union[str, Path]) -> None:
        """Save the tokenizer model to a JSON file."""
        self.tokenizer.save(str(path))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SubwordTokenizer":
        """Load tokenizer from a saved JSON file."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(str(path))
        return instance

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        packed = pack_bitstring(text)
        ids = self.tokenizer.encode(packed).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Union[torch.Tensor, List[int]], remove_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=remove_special_tokens)


class ByteTokenizer:
    """Token-free representation: raw UTF-8 bytes with special tokens."""

    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    unk_id: int = 259
    vocab_size: int = 260

    def __init__(self, auto_pack_bits: bool = True):
        self.auto_pack_bits = auto_pack_bits

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        if self.auto_pack_bits and len(text) >= 8 and len(text) % 8 == 0 and all(c in "01" for c in text):
            raw_bytes = [int(text[i : i + 8], 2) for i in range(0, len(text), 8)]
        else:
            raw_bytes = list(text.encode("utf-8"))

        if add_special_tokens:
            raw_bytes = [self.bos_id] + raw_bytes + [self.eos_id]
        return torch.tensor(raw_bytes, dtype=torch.long)

    def decode(self, ids: Union[torch.Tensor, List[int]], remove_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id} if remove_special_tokens else set()
        byte_list = [int(i) for i in ids if int(i) not in special and 0 <= int(i) <= 255]
        return bytes(byte_list).decode("utf-8", errors="replace")


class ParallelTextDataset(Dataset):
    """Aligned Cipher -> Plaintext parallel dataset."""

    def __init__(
        self,
        cipher_path: str,
        plain_path: str,
        tokenizer: object,
        max_src_len: int = 4096,
        max_tgt_len: int = 1024,
    ):
        if tokenizer is None:
            raise ValueError("A fitted tokenizer must be passed to ParallelTextDataset to prevent data leakage. "
                             "Use make_dataloaders() which handles dataset splitting before tokenizer training.")
        self.cipher = Path(cipher_path).read_text(encoding="utf-8").splitlines()
        self.plain = Path(plain_path).read_text(encoding="utf-8").splitlines()
        if len(self.cipher) != len(self.plain):
            raise ValueError(f"Length mismatch: {len(self.cipher)} cipher vs {len(self.plain)} plain")

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.cipher)

    def __getitem__(self, index: int) -> Dict[str, object]:
        src_text = self.cipher[index]
        tgt_text = self.plain[index]

        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text)

        if len(src_ids) > self.max_src_len:
            src_ids = torch.cat([src_ids[: self.max_src_len - 1], src_ids[-1:]])
        if len(tgt_ids) > self.max_tgt_len:
            tgt_ids = torch.cat([tgt_ids[: self.max_tgt_len - 1], tgt_ids[-1:]])

        return {
            "source": src_ids,
            "target": tgt_ids,
            "source_text": src_text,
            "target_text": tgt_text,
        }


def collate_batch(batch: List[Dict[str, object]], pad_id: int = 0) -> Dict[str, object]:
    source = pad_sequence([x["source"] for x in batch], batch_first=True, padding_value=pad_id)
    target = pad_sequence([x["target"] for x in batch], batch_first=True, padding_value=pad_id)

    return {
        "source": source,
        "target": target,
        # True = valid token, False = padding token (matches SDPA boolean mask convention)
        "source_mask": source.ne(pad_id),
        "target_mask": target.ne(pad_id),
        "source_text": [x["source_text"] for x in batch],
        "target_text": [x["target_text"] for x in batch],
    }


def make_dataloaders(
    cipher_path: str,
    plain_path: str,
    tokenizer: Optional[object] = None,
    batch_size: int = 16,
    max_src_len: int = 4096,
    max_tgt_len: int = 1024,
    vocab_size: int = 8000,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    raw_cipher = Path(cipher_path).read_text(encoding="utf-8").splitlines()
    raw_plain = Path(plain_path).read_text(encoding="utf-8").splitlines()
    if len(raw_cipher) != len(raw_plain):
        raise ValueError(f"Length mismatch: {len(raw_cipher)} cipher vs {len(raw_plain)} plain")

    n = len(raw_cipher)
    train_n = int(split_ratios[0] * n)
    val_n = int(split_ratios[1] * n)
    test_n = n - train_n - val_n

    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=gen).tolist()
    train_indices = indices[:train_n]
    val_indices = indices[train_n : train_n + val_n]
    test_indices = indices[train_n + val_n :]

    if tokenizer is None:
        # Fit tokenizer ONLY on the train split to prevent data leakage
        train_cipher = [raw_cipher[i] for i in train_indices]
        train_plain = [raw_plain[i] for i in train_indices]
        tokenizer = SubwordTokenizer(train_cipher + train_plain, vocab_size=vocab_size)

    dataset = ParallelTextDataset(
        cipher_path,
        plain_path,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    pad_id = getattr(tokenizer, "pad_id", 0)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=pad_id),
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, tokenizer
