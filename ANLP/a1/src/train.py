"""Main training/evaluation loop for the ANLP M26 Assignment 1 ablation study.

Configs (Table 1 of the assignment):
  C1  Base          : Sinusoidal PE + MHA    + LayerNorm + BPE subword
  C2  RoPE          : RoPE           + MHA    + LayerNorm + BPE subword
  C3  GQA           : Sinusoidal PE + GQA    + LayerNorm + BPE subword
  C4  RMSNorm       : Sinusoidal PE + MHA    + RMSNorm   + BPE subword
  C5  BLT           : Sinusoidal PE + MHA    + LayerNorm + Token-Free (raw bytes)

Usage (C1 example):
  python src/train.py --config C1 --epochs 15 --batch-size 8 --grad-accum-steps 2 \
      --lr 5e-4 --dim 256 --heads 8 --layers 4 --max-src-len 1024 --max-tgt-len 512 \
      --vocab-size 8000 --wandb
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from dataset import (
    BOS_ID,
    BYTE_BOS,
    BYTE_EOS,
    BYTE_PAD,
    BYTE_VOCAB,
    EOS_ID,
    PAD_ID,
    BPETextTokenizer,
    PhaseByteBPE,
    cipher_bitstring_to_byte_str,
    collate_byte_target,
    load_pairs,
    make_dataloaders,
)
from models.transformer import BLTModel, Seq2SeqTransformer, TransformerConfig
from utils import compute_metrics, plot_training_curves

# --------------------------------------------------------------------------- #
# Configs                                                                      #
# --------------------------------------------------------------------------- #
CONFIGS = {
    "C1": dict(pos_encoding="sinusoidal", attention="mha",
               norm="layernorm", tokenization="subword"),
    "C2": dict(pos_encoding="rope", attention="mha",
               norm="layernorm", tokenization="subword"),
    "C3": dict(pos_encoding="sinusoidal", attention="gqa",
               norm="layernorm", tokenization="subword"),
    "C4": dict(pos_encoding="sinusoidal", attention="mha",
               norm="rmsnorm", tokenization="subword"),
    "C5": dict(pos_encoding="sinusoidal", attention="mha",
               norm="layernorm", tokenization="blt"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, choices=list(CONFIGS))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="outputs")
    # training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2, 
                   help="effective batch = batch-size * grad-accum-steps (default 16)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    # architecture
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=4,
                   help="KV heads for GQA (C3); ignored otherwise")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=0, help="0 -> 4*dim")
    p.add_argument("--dropout", type=float, default=0.1)
    # data
    p.add_argument("--max-src-len", type=int, default=1024)
    p.add_argument("--length-bucketing", action="store_true",
                   help="use length-homogeneous batches for byte-target "
                        "training (default off: random batches; bucketed "
                        "batches were found to stall alignment learning)")
    p.add_argument("--max-line-bytes", type=int, default=0,
                   help="truncate lines longer than this (cipher and plain "
                        "consistently, key resets per line so prefixes stay "
                        "valid); 0 = no truncation")
    p.add_argument("--max-tgt-len", type=int, default=512,
                   help="auto-set to longest plain line + 2 for byte targets")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--target-bpe", action="store_true",
                   help="C1-C4: use BPE subword targets (default: byte targets)")
    p.add_argument("--no-phase-alphabet", action="store_true",
                   help="C1-C4 byte targets: BPE over raw cipher bytes instead "
                        "of phase-annotated bytes (ablation)")
    p.add_argument("--eval-greedy-every", type=int, default=10,
                   help="run full greedy val eval every N epochs (byte targets)")
    p.add_argument("--scheduled-sampling", action="store_true",
                   help="train C1-C4 byte-target models against the model's own "
                        "greedy prefixes (mixing probability ramps 0->ss-max-p over "
                        "ss-ramp epochs); addresses the teacher-forced/greedy gap "
                        "caused by prefix-content alignment (exposure bias)")
    p.add_argument("--ss-max-p", type=float, default=0.5,
                   help="max scheduled-sampling mixing probability")
    p.add_argument("--ss-ramp", type=int, default=10,
                   help="epochs over which the mixing probability ramps 0->ss-max-p")
    p.add_argument("--ss-every", type=int, default=2,
                   help="regenerate the self prefixes every N epochs (stale reuse "
                        "keeps the greedy pass amortised)")
    p.add_argument("--ss-max-prefix-len", type=int, default=1024,
                   help="cap on the cached self-prefix length in bytes (0 = "
                        "full length); positions beyond the cap keep the true "
                        "prefix, keeping the greedy pass affordable")
    p.add_argument("--prefix-dropout", type=float, default=0.0,
                   help="prob of replacing a decoder-input prefix token with a "
                        "random id during training; forces position-based "
                        "alignment and fixes greedy-decode collapse")
    # BLT
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--byte-dim", type=int, default=64)
    p.add_argument("--local-layers", type=int, default=2)
    p.add_argument("--local-heads", type=int, default=4)
    # logging / checkpoints
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", type=str, default=None,
                   help="WandB run name (default: the config name)")
    p.add_argument("--wandb-project", type=str, default="anlp-assignment1")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--hf-repo", type=str, default=None,
                   help="HuggingFace repo id, e.g. user/anlp-a1 (uploaded per config)")
    p.add_argument("--max-test-log-samples", type=int, default=10)
    # smoke testing
    p.add_argument("--quick", action="store_true",
                   help="tiny subset + 2 epochs for a fast sanity check")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_settings(device: torch.device):
    """Pick the right mixed-precision mode for the target GPU.

    - Ampere or newer (compute capability >= 8.0): bfloat16, no GradScaler.
    - Older GPUs (Turing/V100/2080-Ti, cc < 8.0): float16 + GradScaler.
    - CPU: no autocast.

    Returns (autocast_dtype or None, use_grad_scaler).
    """
    if device.type != "cuda":
        return None, False
    major, minor = torch.cuda.get_device_capability(device.index)
    cc = major * 10 + minor
    if cc >= 80:
        return torch.bfloat16, False
    return torch.float16, True


def make_scheduler(opt, total_steps: int, warmup_steps: int, base_lr: float,
                   min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + cosine * (1.0 - min_lr / base_lr)
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class _ByteBPEAdapter:
    """Give BPETextTokenizer (byte-string API) the cipher-bitstring interface."""

    def __init__(self, bpe):
        self.bpe = bpe

    def encode(self, cipher_bits: str):
        return self.bpe.encode(cipher_bitstring_to_byte_str(cipher_bits))

    @property
    def vocab_size(self):
        return self.bpe.vocab_size


def get_tokenizers(args, train_pairs, output_dir, use_bytes_target: bool,
                   phase_alphabet: bool):
    """Train (or load cached) BPE tokenizers for C1-C4.

    Cipher side:
      - phase_alphabet: BPE learned over phase-annotated cipher bytes,
        symbols (byte, i mod 8) (PhaseByteBPE, 2048 base symbols).
      - otherwise: BPE over raw cipher bytes (variable-length byte tokens).
    Plain side: BPE over the English text (used for --target-bpe ablation).
    """
    tok_dir = os.path.join(output_dir, "tokenizers")
    tag = f"v{args.vocab_size}_s{args.seed}" + ("_quick" if args.quick else "")
    plain_path = os.path.join(tok_dir, f"plain_bpe_{tag}.json")
    plain_tok = BPETextTokenizer(plain_path) if os.path.exists(plain_path) \
        else BPETextTokenizer.train([p for _, p in train_pairs], args.vocab_size,
                                    tok_dir, f"plain_bpe_{tag}")
    if use_bytes_target and phase_alphabet:
        ciph_name = f"cipher_phase_bpe_{tag}"
        ciph_path = os.path.join(tok_dir, ciph_name)
        cipher_tok = PhaseByteBPE(ciph_path) if os.path.exists(ciph_path) \
            else PhaseByteBPE.train([c for c, _ in train_pairs], args.vocab_size,
                                    tok_dir, ciph_name)
    else:
        ciph_name = f"cipher_byte_bpe_{tag}"
        ciph_path = os.path.join(tok_dir, ciph_name)
        cipher_tok = BPETextTokenizer(ciph_path) if os.path.exists(ciph_path) \
            else BPETextTokenizer.train(
                [cipher_bitstring_to_byte_str(c) for c, _ in train_pairs],
                args.vocab_size, tok_dir, ciph_name)
    return cipher_tok, plain_tok


def build_model(args, spec: dict, device: torch.device, src_vocab: int = None,
                tgt_vocab: int = None, use_bytes_target: bool = False):
    max_len = max(args.max_src_len, args.max_tgt_len, 1024) + 32
    tcfg = TransformerConfig(
        d_model=args.dim,
        n_heads=args.heads,
        n_kv_heads=(args.kv_heads if spec["attention"] == "gqa" else args.heads),
        n_layers=args.layers,
        d_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=max_len,
        pos_encoding=spec["pos_encoding"],
        norm=spec["norm"],
        prefix_dropout=getattr(args, "prefix_dropout", 0.0),
    )
    if spec["tokenization"] == "blt":
        model = BLTModel(tcfg, patch_size=args.patch_size, byte_dim=args.byte_dim,
                         n_local_layers=args.local_layers, n_local_heads=args.local_heads)
    elif use_bytes_target:
        model = Seq2SeqTransformer(src_vocab, tgt_vocab, tcfg,
                                   pad_idx=PAD_ID, eos_idx=BYTE_EOS,
                                   tgt_pad_idx=BYTE_PAD)
    else:
        model = Seq2SeqTransformer(src_vocab, tgt_vocab, tcfg,
                                   pad_idx=PAD_ID, eos_idx=EOS_ID)
    return model.to(device), tcfg


def train_step(model, batch, criterion, device, is_blt: bool,
               dec_input: torch.Tensor = None):
    if is_blt:
        src, tgt = batch["src"].to(device), batch["tgt"].to(device)
        mask = batch["src_mask"].to(device)
        logits = model(src, mask)
        logits = logits[:, : tgt.size(1)]  # align to padded byte length
        return criterion(logits.reshape(-1, 256), tgt.reshape(-1))
    src, tgt = batch["src"].to(device), batch["tgt"].to(device)
    src_mask = batch["src_mask"].to(device)
    if dec_input is not None:
        dec_input = dec_input.to(device)
    logits = model(src, src_mask, tgt, dec_input=dec_input)
    return criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))


@torch.no_grad()
def generate_self_prefixes(model, loader, device, args) -> list:
    """Greedy-decode every sample in ``loader`` (byte-target C1-C4) and return
    a list of predicted byte lists (values 0-255, EOS removed) in dataset
    order.  Used by scheduled sampling: the model is later trained against its
    own (plausibly wrong) prefixes so that autoregressive decoding does not
    collapse when its prefix drifts from the ground truth.

    Decoding is capped at ``ss_max_prefix_len`` bytes: the alignment drift
    starts within the first ~50 bytes, so a capped prefix still corrupts the
    regime that matters, while cutting the (quadratic) decode cost for the
    ~2600-byte sequences."""
    model.eval()
    cap = int(getattr(args, "ss_max_prefix_len", 0) or args.max_tgt_len)
    out = []
    for batch in loader:
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)
        ids = model.generate(src, src_mask,
                             max_len=min(args.max_tgt_len - 1, cap),
                             bos_idx=BYTE_BOS)  # (B, T) ids, EOS-truncated
        for i in range(ids.size(0)):
            row = ids[i].tolist()
            if BYTE_EOS in row:
                row = row[: row.index(BYTE_EOS)]
            out.append([min(max(v, 0), 255) for v in row])
    return out


def build_scheduled_dec_input(batch, self_prefixes, indices, p, device):
    """Mixed decoder input for scheduled sampling.

    For each sample in the batch, with probability ``p`` use that sample's own
    greedy prediction (``self_prefixes``) as the decoder-side prefix; otherwise
    keep the true teacher-forced prefix.  If the cached prefix is shorter than
    the sequence (it is generated with a length cap, or ended at EOS), the
    remaining positions keep the TRUE prefix (clean teacher forcing after the
    corrupted span).  The result has shape (B, T-1) and is aligned with
    ``batch['tgt'][:, 1:]`` (the loss target).
    """
    tgt = batch["tgt"]            # (B, T) padded, incl. BOS/EOS
    B, T = tgt.shape
    true_in = tgt[:, :-1]         # (B, T-1) true decoder input
    dec_in = true_in.clone()
    for j, idx in enumerate(indices):
        if random.random() >= p:
            continue
        sp = self_prefixes[idx]   # list of ints, 0-255
        n = T - 2                 # content positions after BOS
        k = min(len(sp), n)
        row = [BYTE_BOS] + sp[:k] + list(true_in[j, 1:1 + (n - k)])
        dec_in[j, :] = torch.tensor(row, dtype=torch.long)
    return dec_in.to(device)


@torch.no_grad()
def evaluate(model, loader, device, args, is_blt: bool, plain_tok=None, amp=None,
             target_bytes: bool = False, mode: str = "greedy") -> dict:
    """Evaluate the model.

    mode="greedy":  autoregressive decoding (single forward pass for C5/BLT).
    mode="teacher": one teacher-forced forward pass + argmax.  Fast; used for
                    per-epoch tracking of byte-target models (T up to ~2.7k).
    target_bytes: C1-C4 byte-target variant (256-way output, one byte per
                    target position; ids 0-255, truncated at BYTE_EOS).
    """
    model.eval()
    preds = []
    if amp is None:
        amp = get_amp_settings(device)
    amp_dtype, amp_enabled = amp

    def decode_byte_row(row: list) -> str:
        if BYTE_EOS in row:
            row = row[: row.index(BYTE_EOS)]
        return bytes([min(max(v, 0), 255) for v in row]).decode("latin1")

    for batch in loader:
        src = batch["src"].to(device)
        if is_blt:
            lengths = batch["lengths"].to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                byte_lists = model.predict_bytes(src, lengths)
            preds.extend([bytes(b).decode("latin1") for b in byte_lists])
        else:
            src_mask = batch["src_mask"].to(device)
            tgt = batch["tgt"].to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                if mode == "teacher":
                    ids = model(src, src_mask, tgt).argmax(-1)
                elif target_bytes:
                    max_len = min(args.max_tgt_len, int(tgt.size(1)))
                    ids = model.generate(src, src_mask, max_len, BYTE_BOS)
                else:
                    max_len = min(args.max_tgt_len, int(tgt.size(1)))
                    ids = model.generate(src, src_mask, max_len, BOS_ID)
            for row in ids:
                row = row.tolist()
                if target_bytes:
                    preds.append(decode_byte_row(row))
                else:
                    if EOS_ID in row:
                        row = row[: row.index(EOS_ID) + 1]
                    preds.append(plain_tok.decode(row))
    refs = _ordered_refs(loader)  # aligned with the loader's iteration order
    metrics = compute_metrics(refs, preds)
    metrics["_preds"] = preds[: args.max_test_log_samples]
    metrics["_refs"] = refs[: args.max_test_log_samples]
    return metrics


def _ordered_refs(loader) -> list:
    """References in the loader's batch iteration order (aligned with preds).

    With a custom batch_sampler (length bucketing) or a shuffled loader the
    batch order differs from dataset order, so refs must be permuted to match.
    LengthBatchSampler is deterministic for a fixed epoch, so re-iterating it
    reproduces the exact order the DataLoader used.
    """
    ds = loader.dataset
    if not hasattr(ds, "items"):
        return ds.refs()
    import itertools

    flat = list(itertools.chain.from_iterable(loader.batch_sampler))
    return [ds.items[i][2] for i in flat]


def log_samples(wandb_run, metrics: dict, step_name: str):
    if wandb_run is None:
        return
    import wandb

    rows = [[r, p] for r, p in zip(metrics["_refs"], metrics["_preds"])]
    wandb_run.log({
        f"samples_{step_name}": wandb.Table(
            data=rows, columns=["reference", "prediction"]
        )
    })


def save_checkpoint(path, model, tcfg, args, epoch, metrics):
    torch.save({
        "model": model.state_dict(),
        "tcfg": asdict(tcfg),
        "args": vars(args),
        "epoch": epoch,
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
    }, path)


def upload_to_hf(args, output_dir: str) -> str:
    """Upload the best checkpoint, config, results and tokenizers to HuggingFace."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hf] huggingface_hub not installed - skipping upload "
              "(pip install huggingface_hub)")
        return ""
    api = HfApi()
    repo = args.hf_repo
    try:
        api.create_repo(repo, exist_ok=True)
    except Exception as e:  # token missing / offline
        print(f"[hf] could not create repo {repo}: {e}")
        return ""
    files = []
    for fn in ("model_best.pt", "model_last.pt", "config.json", "results.json"):
        p = os.path.join(output_dir, fn)
        if os.path.exists(p):
            files.append(p)
    tok_dir = os.path.join(output_dir, "tokenizers")
    if os.path.isdir(tok_dir):
        files += [os.path.join(tok_dir, f) for f in os.listdir(tok_dir)]
    for f in files:
        url = api.upload_file(
            path_or_fileobj=f,
            path_in_repo=os.path.basename(f),
            repo_id=repo,
        )
        print(f"[hf] uploaded {f} -> {url}")
    return api.repo_url(repo)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    spec = CONFIGS[args.config]
    set_seed(args.seed)
    device = pick_device(args.device)
    is_blt = spec["tokenization"] == "blt"
    # C1-C4 target scheme: byte targets by default (position i <-> plain byte i);
    # --target-bpe restores the BPE subword target for ablation comparison.
    use_bytes_target = (not is_blt) and not args.target_bpe

    output_dir = os.path.join(args.output_dir, args.config)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Config {args.config}: {spec} ===")
    print(f"device: {device}")

    # ---------------- data ----------------
    t0 = time.time()
    pairs = load_pairs(
        os.path.join(args.data_dir, "brown_cipher.txt"),
        os.path.join(args.data_dir, "brown_plain.txt"),
    )
    print(f"[data] {len(pairs)} pairs loaded & verified in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    n_train = int(0.8 * len(pairs))
    n_val = int(0.1 * len(pairs))
    split = {
        "train": [pairs[i] for i in idx[:n_train]],
        "val": [pairs[i] for i in idx[n_train : n_train + n_val]],
        "test": [pairs[i] for i in idx[n_train + n_val :]],
    }

    if args.max_line_bytes > 0:
        n_trunc = 0
        for name in split:
            out = []
            for c, p in split[name]:
                if len(p) > args.max_line_bytes:
                    p = p[:args.max_line_bytes]
                    c = c[:args.max_line_bytes * 8]
                    n_trunc += 1
                out.append((c, p))
            split[name] = out
        print(f"[data] {n_trunc} lines truncated to {args.max_line_bytes} bytes "
              f"(all splits, cipher and plain consistently)")

    if args.quick:
        split = {k: v[: 256 if k == "train" else 64] for k, v in split.items()}
        args.epochs = 2
        args.eval_batch_size = 8
        print("[quick] smoke-test mode: subset + 2 epochs")

    if use_bytes_target:
        # target position i <-> plaintext byte i  =>  cap from the longest line
        args.max_tgt_len = max(len(p) for pl in split.values() for _, p in pl) + 2
        if args.quick:
            args.max_tgt_len = min(args.max_tgt_len, 800)
        print(f"[data] byte targets: max_tgt_len set to {args.max_tgt_len}")

    tokenizers = None
    plain_tok = None
    phase_alphabet = use_bytes_target and not args.no_phase_alphabet
    if not is_blt:
        t0 = time.time()
        cipher_tok, plain_tok = get_tokenizers(args, split["train"], output_dir,
                                               use_bytes_target, phase_alphabet)
        if use_bytes_target and not phase_alphabet:
            cipher_tok = _ByteBPEAdapter(cipher_tok)
        tokenizers = (cipher_tok, plain_tok)
        print(f"[bpe] trained/loaded tokenizers in {time.time() - t0:.1f}s "
              f"(cipher vocab={cipher_tok.vocab_size}, plain vocab={plain_tok.vocab_size})")

    loaders = make_dataloaders(split, args, tokenizers, target_bytes=use_bytes_target)
    for name in ("train", "val", "test"):
        print(f"[data] {name}: {len(loaders[name].dataset)} samples")

    # ---------------- model / optimizer ----------------
    src_vocab = cipher_tok.vocab_size if not is_blt else None
    tgt_vocab = (BYTE_VOCAB if use_bytes_target else plain_tok.vocab_size) \
        if not is_blt else None
    model, tcfg = build_model(args, spec, device, src_vocab, tgt_vocab,
                              use_bytes_target=use_bytes_target)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M params; {tcfg.describe()}")
    if not is_blt:
        if use_bytes_target:
            ciph_desc = ("phase-annotated bytes (byte, i mod 8)"
                         if phase_alphabet else "raw cipher bytes")
            print(f"[target] byte-level (256-way); cipher BPE over {ciph_desc}")
        else:
            print("[target] BPE subword; cipher BPE over raw cipher bytes")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"config": args.config, "spec": spec, "tcfg": asdict(tcfg),
                   "n_params": n_params, "args": vars(args)}, f, indent=2, default=str)

    criterion = nn.CrossEntropyLoss(
        ignore_index=BYTE_PAD if (is_blt or use_bytes_target) else PAD_ID)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(len(loaders["train"]) // args.grad_accum_steps, 1)
    total_steps = steps_per_epoch * args.epochs
    sched = make_scheduler(opt, total_steps, args.warmup_steps, args.lr, args.min_lr)
    amp_dtype, use_scaler = get_amp_settings(device)
    amp_enabled = amp_dtype is not None
    amp = (amp_dtype, amp_enabled)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if device.type == "cuda":
        print(f"[amp] autocast dtype={amp_dtype}, grad_scaler={use_scaler}")

    # ---------------- wandb ----------------
    wandb_run = None
    if args.wandb and not args.quick:
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed - running without wandb")
        else:
            if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline"):
                print("[wandb] WANDB_API_KEY not set - running without wandb")
            else:
                mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
                # retry transient network errors (a CommError must not kill a
                # multi-hour run); fall back to offline so wandb sync can be
                # done later with `wandb sync`.
                for attempt in range(1, 4):
                    try:
                        wandb_run = wandb.init(
                            project=args.wandb_project,
                            entity=args.wandb_entity,
                            name=args.run_name or args.config,
                            config=vars(args),
                            mode=mode,
                        )
                        break
                    except Exception as e:
                        print(f"[wandb] init failed (attempt {attempt}/3): {e}")
                        if attempt < 3:
                            import time as _t
                            _t.sleep(20)
                if wandb_run is None and mode == "online":
                    print("[wandb] online init failed - falling back to offline mode")
                    try:
                        wandb_run = wandb.init(
                            project=args.wandb_project,
                            entity=args.wandb_entity,
                            name=args.run_name or args.config,
                            config=vars(args),
                            mode="offline",
                        )
                    except Exception as e:
                        print(f"[wandb] offline init failed too: {e} - continuing without wandb")

    # ---------------- training loop ----------------
    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_bit_accuracy",
                               "val_sequence_accuracy", "val_levenshtein", "val_bleu",
                               "val_rouge_l", "train_samples_per_sec", "peak_mem_mb")}
    best_val = (float("inf"), 0.0)  # (loss, seq_acc)
    global_step = 0
    t_start = time.time()

    # ---------------- scheduled sampling setup (C1-C4 byte targets) --------
    use_ss = bool(getattr(args, "scheduled_sampling", False)) and use_bytes_target
    self_prefixes = None
    ss_gen_loader = None
    if use_ss:
        # Unshuffled, larger-batch loader used only to regenerate the cached
        # self prefixes (dataset order 0..N-1).  Kept separate from the
        # shuffled training loader so the index -> prefix mapping is stable.
        ss_gen_loader = torch.utils.data.DataLoader(
            loaders["train"].dataset,
            batch_size=32,  # the KV-cache greedy pass is Python-loop bound:
            # a big batch amortises the per-step overhead (fits in ~1.5 GB)
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_byte_target,
            persistent_workers=args.num_workers > 0,
        )

    for epoch in range(1, args.epochs + 1):
        # rotate the length-bucket order each epoch (byte targets)
        train_sampler = getattr(loaders["train"], "batch_sampler", None)
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_loss, n_batches, t_ep = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)

        # Scheduled sampling: mixing probability ramps 0 -> ss_max_p over
        # ss_ramp epochs; self prefixes are regenerated every ss_every epochs.
        ss_p = 0.0
        if use_ss:
            ss_p = args.ss_max_p * min(1.0, (epoch - 1) / max(1, args.ss_ramp))
            if ss_p > 0.0 and (self_prefixes is None or epoch % args.ss_every == 0):
                t_ss = time.time()
                was_training = model.training
                model.eval()  # generate() decodes with dropout off
                self_prefixes = generate_self_prefixes(
                    model, ss_gen_loader, device, args)
                if was_training:
                    model.train()
                print(f"[{args.config}] epoch {epoch:02d} | self prefixes "
                      f"regenerated ({len(self_prefixes)} samples) in "
                      f"{time.time() - t_ss:.0f}s | ss_p={ss_p:.2f}")

        for i, batch in enumerate(loaders["train"]):
            dec_input = None
            if use_ss and ss_p > 0.0 and self_prefixes is not None \
                    and "indices" in batch:
                dec_input = build_scheduled_dec_input(
                    batch, self_prefixes, batch["indices"].tolist(), ss_p, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = train_step(model, batch, criterion, device, is_blt,
                                  dec_input=dec_input)
            if use_scaler:
                scaler.scale(loss / args.grad_accum_steps).backward()
            else:
                (loss / args.grad_accum_steps).backward()
            epoch_loss += loss.item()
            n_batches += 1
            if (i + 1) % args.grad_accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if wandb_run is not None and global_step % 25 == 0:
                    wandb_run.log({"train/loss": epoch_loss / n_batches,
                                   "train/lr": opt.param_groups[0]["lr"],
                                   "train/step": global_step})

        # Flush a trailing partial accumulation group, if any.
        if n_batches % args.grad_accum_steps != 0:
            if use_scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

        epoch_time = time.time() - t_ep
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
        train_loss = epoch_loss / max(n_batches, 1)
        n_train_samples = len(loaders["train"].dataset)
        sps = n_train_samples / epoch_time

        # ---------------- validation ----------------
        model.eval()
        val_logits_loss = 0.0
        val_n = 0
        for batch in loaders["val"]:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                vloss = train_step(model, batch, criterion, device, is_blt)
            val_logits_loss += vloss.item() * batch["src"].size(0)
            val_n += batch["src"].size(0)
        val_loss = val_logits_loss / max(val_n, 1)
        if use_bytes_target:
            val_metrics = evaluate(model, loaders["val"], device, args, is_blt,
                                   plain_tok, amp, target_bytes=True, mode="teacher")
        else:
            val_metrics = evaluate(model, loaders["val"], device, args, is_blt, plain_tok, amp)
        greedy_metrics = None
        if use_bytes_target and epoch % args.eval_greedy_every == 0:
            greedy_metrics = evaluate(model, loaders["val"], device, args, is_blt,
                                      plain_tok, amp, target_bytes=True, mode="greedy")
            print(f"[{args.config}] epoch {epoch:02d} GREEDY val: bit_acc "
                  f"{greedy_metrics['bit_accuracy']:.4f} | seq_acc "
                  f"{greedy_metrics['sequence_accuracy']:.4f} | lev "
                  f"{greedy_metrics['levenshtein']:.1f} | bleu "
                  f"{greedy_metrics.get('bleu', float('nan')):.4f} | rougeL "
                  f"{greedy_metrics.get('rouge_l', float('nan')):.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bit_accuracy"].append(val_metrics["bit_accuracy"])
        history["val_sequence_accuracy"].append(val_metrics["sequence_accuracy"])
        history["val_levenshtein"].append(val_metrics["levenshtein"])
        history["val_bleu"].append(val_metrics.get("bleu"))
        history["val_rouge_l"].append(val_metrics.get("rouge_l"))
        history["train_samples_per_sec"].append(sps)
        history["peak_mem_mb"].append(peak_mb)

        print(f"[{args.config}] epoch {epoch:02d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f}{' (tf)' if use_bytes_target else ''} | bit_acc {val_metrics['bit_accuracy']:.4f} | "
              f"seq_acc {val_metrics['sequence_accuracy']:.4f} | "
              f"lev {val_metrics['levenshtein']:.1f} | "
              f"bleu {val_metrics.get('bleu', float('nan')):.4f} | "
              f"rougeL {val_metrics.get('rouge_l', float('nan')):.4f} | "
              f"{sps:.0f} samples/s | peak {peak_mb:.0f} MB | {epoch_time:.0f}s")

        # ---------------- checkpoints / logging ----------------
        improved = epoch == 1 or (math.isfinite(val_loss) and val_loss < best_val[0])
        if improved:
            best_val = (val_loss, val_metrics["sequence_accuracy"])
            save_checkpoint(os.path.join(output_dir, "model_best.pt"), model, tcfg,
                            args, epoch, val_metrics)
        save_checkpoint(os.path.join(output_dir, "model_last.pt"), model, tcfg,
                        args, epoch, val_metrics)

        if wandb_run is not None:
            wandb_run.log({
                "train/epoch_loss": train_loss,
                "val/loss": val_loss,
                "val/bit_accuracy": val_metrics["bit_accuracy"],
                "val/sequence_accuracy": val_metrics["sequence_accuracy"],
                "val/levenshtein": val_metrics["levenshtein"],
                "val/bleu": val_metrics.get("bleu"),
                "val/rouge_l": val_metrics.get("rouge_l"),
                "train/samples_per_sec": sps,
                "train/peak_mem_mb": peak_mb,
                "train/epoch_time_s": epoch_time,
                "epoch": epoch,
            })
            log_samples(wandb_run, val_metrics, f"epoch_{epoch}")
            if greedy_metrics is not None:
                wandb_run.log({f"val/greedy_{k}": v for k, v in greedy_metrics.items()
                               if not k.startswith("_")})
        if epoch == 1:
            with open(os.path.join(output_dir, "val_samples_epoch1.txt"), "w") as f:
                for r, p in zip(val_metrics["_refs"], val_metrics["_preds"]):
                    f.write(f"REF:  {r}\nPRED: {p}\n{'-' * 80}\n")

    total_time = time.time() - t_start
    print(f"[{args.config}] training done in {total_time / 60:.1f} min")

    # ---------------- final test with best checkpoint ----------------
    ckpt = torch.load(os.path.join(output_dir, "model_best.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, loaders["test"], device, args, is_blt, plain_tok, amp,
                            target_bytes=use_bytes_target, mode="greedy")
    results = {
        "config": args.config,
        "spec": spec,
        "n_params": n_params,
        "train_samples": len(loaders["train"].dataset),
        "val_samples": len(loaders["val"].dataset),
        "test_samples": len(loaders["test"].dataset),
        "epochs": args.epochs,
        "total_time_min": total_time / 60,
        "best_epoch": ckpt["epoch"],
        "best_val_loss": best_val[0],
        "test": {k: v for k, v in test_metrics.items() if not k.startswith("_")},
        "history": {k: v for k, v in history.items()},
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(output_dir, "test_samples.txt"), "w") as f:
        for r, p in zip(test_metrics["_refs"], test_metrics["_preds"]):
            f.write(f"REF:  {r}\nPRED: {p}\n{'-' * 80}\n")

    print(f"[{args.config}] TEST: bit_acc {test_metrics['bit_accuracy']:.4f} | "
          f"seq_acc {test_metrics['sequence_accuracy']:.4f} | "
          f"lev {test_metrics['levenshtein']:.1f} | "
          f"bleu {test_metrics.get('bleu', float('nan')):.4f} | "
          f"rougeL {test_metrics.get('rouge_l', float('nan')):.4f}")

    plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))

    if wandb_run is not None:
        wandb_run.log({"test/bit_accuracy": test_metrics["bit_accuracy"],
                       "test/sequence_accuracy": test_metrics["sequence_accuracy"],
                       "test/levenshtein": test_metrics["levenshtein"],
                       "test/bleu": test_metrics.get("bleu"),
                       "test/rouge_l": test_metrics.get("rouge_l"),
                       "train/avg_samples_per_sec":
                           float(np.mean(history["train_samples_per_sec"])),
                       "train/avg_peak_mem_mb":
                           float(np.mean(history["peak_mem_mb"])),
                       "train/total_time_min": total_time / 60})
        log_samples(wandb_run, test_metrics, "test")
        try:
            wandb_run.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")

    hf_url = ""
    if args.hf_repo:
        hf_url = upload_to_hf(args, output_dir)
        results["hf_url"] = hf_url
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
