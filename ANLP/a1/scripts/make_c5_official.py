"""Assemble the official outputs/C5 folder from the (stopped) fixed-stride run.

The C5-fixed SLURM job trained 116 of its 120 planned epochs (val loss
plateaued at ~0.0010 from epoch ~90; best checkpoint at epoch 112) before
being stopped. This script:
  1. parses the per-epoch training log into the standard history dict,
  2. re-runs the official greedy test evaluation on model_best.pt,
  3. writes results.json / test_samples.txt / training_curves.png in the
     exact format src/train.py would have produced.

Usage:
    python scripts/make_c5_official.py \
        --run-dir outputs/ablations/C5 \
        --log outputs/anlp_C5_fixed_2686327.log \
        --out-dir outputs/C5
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

import train as T

EPOCH_RE = re.compile(
    r"^\[C5\] epoch (\d+) \| train_loss ([\d.]+) \| val_loss ([\d.]+)"
    r"(?: \((\w+)\))? \| bit_acc ([\d.]+) \| seq_acc ([\d.]+) \| lev ([\d.]+)"
    r" \| bleu ([\d.]+) \| rougeL ([\d.]+) \| (\d+) samples/s \| peak (\d+) MB"
    r" \| (\d+)s")


def parse_log(path):
    history = {k: [] for k in
               ("epoch", "train_loss", "val_loss", "val_bit_accuracy",
                "val_sequence_accuracy", "val_levenshtein", "val_bleu",
                "val_rouge_l", "train_samples_per_sec", "peak_mem_mb")}
    total_s = 0
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if not m:
                continue
            (ep, tl, vl, _tag, ba, sa, lev, bleu, rl, sps, mem, sec) = m.groups()
            history["epoch"].append(int(ep))
            history["train_loss"].append(float(tl))
            history["val_loss"].append(float(vl))
            history["val_bit_accuracy"].append(float(ba))
            history["val_sequence_accuracy"].append(float(sa))
            history["val_levenshtein"].append(float(lev))
            history["val_bleu"].append(float(bleu))
            history["val_rouge_l"].append(float(rl))
            history["train_samples_per_sec"].append(int(sps))
            history["peak_mem_mb"].append(int(mem))
            total_s += int(sec)
    return history, total_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir with config.json + model_best.pt")
    ap.add_argument("--log", required=True, help="SLURM training log with per-epoch lines")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    cli = ap.parse_args()

    with open(os.path.join(cli.run_dir, "config.json")) as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg["args"])
    args.wandb, args.hf_repo, args.quick = False, None, False
    args.eval_batch_size, args.num_workers = cli.eval_batch_size, 0

    spec = T.CONFIGS[args.config]
    assert spec["tokenization"] == "blt"
    T.set_seed(args.seed)
    pairs = T.load_pairs(os.path.join(args.data_dir, "brown_cipher.txt"),
                         os.path.join(args.data_dir, "brown_plain.txt"))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    n_train, n_val = int(0.8 * len(pairs)), int(0.1 * len(pairs))
    split = {
        "train": [pairs[i] for i in idx[:n_train]],
        "val": [pairs[i] for i in idx[n_train:n_train + n_val]],
        "test": [pairs[i] for i in idx[n_train + n_val:]],
    }
    # C5 fixed: no tokenizers, no patch structures (fixed stride built in dataset)
    loaders = T.make_dataloaders(split, args, None, target_bytes=False,
                                 patch_structures=None)

    device = T.pick_device(args.device)
    model, _ = T.build_model(args, spec, device, None, None, use_bytes_target=False)
    ckpt = torch.load(os.path.join(cli.run_dir, "model_best.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    metrics = T.evaluate(model, loaders["test"], device, args, True, None,
                         T.get_amp_settings(device), target_bytes=False,
                         mode="greedy")

    history, total_s = parse_log(cli.log)
    n_epochs = len(history["epoch"])
    assert n_epochs > 0, "no epoch lines found in log"
    best_epoch = int(ckpt["epoch"])
    best_val = history["val_loss"][best_epoch - 1]

    results = {
        "config": args.config,
        "spec": spec,
        "patching": cfg["patching"],
        "n_params": cfg["n_params"],
        "train_samples": len(loaders["train"].dataset),
        "val_samples": len(loaders["val"].dataset),
        "test_samples": len(loaders["test"].dataset),
        "epochs": n_epochs,
        "total_time_min": total_s / 60.0,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test": {k: v for k, v in metrics.items() if not k.startswith("_")},
        "history": history,
    }

    os.makedirs(cli.out_dir, exist_ok=True)
    with open(os.path.join(cli.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(cli.out_dir, "test_samples.txt"), "w") as f:
        for r, p in zip(metrics["_refs"], metrics["_preds"]):
            f.write(f"REF:  {r}\nPRED: {p}\n{'-' * 80}\n")
    T.plot_training_curves(history, os.path.join(cli.out_dir, "training_curves.png"))

    t = results["test"]
    print(f"official C5: {n_epochs} epochs | best ep {best_epoch} "
          f"(val_loss {best_val:.4f}) | {total_s / 60:.1f} min")
    print(f"TEST: bit_acc {t['bit_accuracy']:.4f} | seq_acc "
          f"{t['sequence_accuracy']:.4f} | lev {t['levenshtein']:.1f} | "
          f"bleu {t['bleu']:.4f} | rougeL {t['rouge_l']:.4f}")


if __name__ == "__main__":
    main()
