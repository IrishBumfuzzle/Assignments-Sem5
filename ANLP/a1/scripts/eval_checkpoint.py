"""Evaluate a saved checkpoint on the train/val/test split.

Usage:
    python scripts/eval_checkpoint.py outputs/C1/model_best.pt
    python scripts/eval_checkpoint.py outputs/C1/model_best.pt --split val
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

import train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--target-bpe", action="store_true",
                    help="force BPE target interpretation (pre-flag checkpoints)")
    cli = ap.parse_args()

    ckpt = torch.load(cli.checkpoint, map_location="cpu")
    avars = dict(ckpt["args"])
    # pre-flag checkpoints were always BPE-target
    avars["target_bpe"] = cli.target_bpe or avars.get("target_bpe", True)
    avars["wandb"] = False
    avars["hf_repo"] = None
    avars["quick"] = False
    avars["eval_batch_size"] = cli.eval_batch_size
    avars["num_workers"] = cli.num_workers
    args = argparse.Namespace(**avars)

    spec = T.CONFIGS[args.config]
    is_blt = spec["tokenization"] == "blt"
    # getattr: checkpoints from runs before these flags existed
    use_bytes_target = (not is_blt) and not getattr(args, "target_bpe", False)
    phase_alphabet = use_bytes_target and not getattr(args, "no_phase_alphabet", False)

    T.set_seed(args.seed)
    pairs = T.load_pairs(
        os.path.join(args.data_dir, "brown_cipher.txt"),
        os.path.join(args.data_dir, "brown_plain.txt"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    n_train, n_val = int(0.8 * len(pairs)), int(0.1 * len(pairs))
    split = {
        "train": [pairs[i] for i in idx[:n_train]],
        "val": [pairs[i] for i in idx[n_train:n_train + n_val]],
        "test": [pairs[i] for i in idx[n_train + n_val:]],
    }
    cap = avars.get("max_line_bytes", 0)
    if cap > 0:
        for name in split:
            split[name] = [(c[:cap * 8], p[:cap]) if len(p) > cap else (c, p)
                           for c, p in split[name]]

    out_dir = os.path.dirname(os.path.abspath(cli.checkpoint))
    device = T.pick_device(args.device)

    # C5: restore the (dynamic) patch structures saved with the checkpoint.
    patch_structures = None
    if is_blt:
        patching = ckpt.get("patching") or {}
        if patching.get("method") == "entropy" and "entropy_lm" in ckpt:
            from dataset import cipher_bytes_of
            from entropy_patching import (ByteEntropyLM, EntropyLMConfig,
                                         build_patch_structures,
                                         next_byte_entropies)
            pelm = ByteEntropyLM(EntropyLMConfig(**ckpt["entropy_lm"]["cfg"]))
            pelm.load_state_dict(ckpt["entropy_lm"]["state"])
            pelm.to(device).eval()
            kept = []
            for c, _ in split[cli.split]:
                cb = cipher_bytes_of(c)
                if len(cb) <= args.max_src_len:
                    kept.append(list(cb))
            H = next_byte_entropies(pelm, kept, device)
            patch_structures = {cli.split: build_patch_structures(
                kept, patching["theta_g"], patching.get("theta_r", 0.25),
                patching.get("max_patch", 12), H)}
            n_bytes = sum(len(x) for x in kept)
            n_patches = sum(len(l) for _, l in patch_structures[cli.split])
            print(f"[c5] entropy patching restored: theta_g={patching['theta_g']:.4f} "
                  f"| mean patch {n_bytes / max(n_patches, 1):.2f} bytes")
        elif not patching:
            print("[c5] note: checkpoint has no patching info (pre-entropy C5 "
                  "checkpoint uses the old fixed-patch architecture and cannot "
                  "be loaded by the current model code)")

    if use_bytes_target:
        args.max_tgt_len = max(len(p) for pl in split.values() for _, p in pl) + 2

    tokenizers = None
    plain_tok = None
    if not is_blt:
        cipher_tok, plain_tok = T.get_tokenizers(
            args, split["train"], out_dir, use_bytes_target, phase_alphabet)
        if use_bytes_target and not phase_alphabet:
            cipher_tok = T._ByteBPEAdapter(cipher_tok)
        tokenizers = (cipher_tok, plain_tok)

    loaders = T.make_dataloaders(split, args, tokenizers,
                                 target_bytes=use_bytes_target,
                                 patch_structures=patch_structures)
    src_vocab = cipher_tok.vocab_size if not is_blt else None
    tgt_vocab = (T.BYTE_VOCAB if use_bytes_target else plain_tok.vocab_size) \
        if not is_blt else None
    model, _ = T.build_model(args, spec, device, src_vocab, tgt_vocab,
                             use_bytes_target=use_bytes_target)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    metrics = T.evaluate(model, loaders[cli.split], device, args, is_blt,
                         plain_tok, T.get_amp_settings(device),
                         target_bytes=use_bytes_target, mode="greedy")
    print(f"\n{cli.split} split @ epoch {ckpt['epoch']} ({cli.checkpoint})")
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        print(f"  {k:15s} {v}")
    with open(os.path.join(out_dir, f"reval_{cli.split}.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")},
                  f, indent=2)


if __name__ == "__main__":
    main()
