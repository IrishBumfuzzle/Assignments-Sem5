"""Print the report tables (LaTeX + Markdown) from outputs/<C>/results.json.

Usage:
    python scripts/make_results_table.py [outputs_dir]
"""
import json
import os
import sys

NAMES = {
    "C1": "C1 Base", "C2": "C2 RoPE", "C3": "C3 GQA",
    "C4": "C4 RMSNorm", "C5": "C5 BLT",
}


def load(out_dir):
    out = {}
    for c in NAMES:
        p = os.path.join(out_dir, c, "results.json")
        if os.path.exists(p):
            out[c] = json.load(open(p))
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    data = load(out_dir)
    if not data:
        print(f"no results.json found under {out_dir}/C*/")
        return

    print("## Markdown (README)\n")
    print("| Config | Bit acc | Seq acc | Levenshtein | BLEU | ROUGE-L | Best epoch |")
    print("|--------|---------|---------|-------------|------|---------|------------|")
    for c, d in data.items():
        t = d["test"]
        print(f"| {NAMES[c]} | {t['bit_accuracy']:.4f} | {t['sequence_accuracy']:.4f} "
              f"| {t['levenshtein']:.1f} | {t['bleu']:.4f} | {t['rouge_l']:.4f} "
              f"| {d['best_epoch']} |")

    print("\n## LaTeX: Table 1 (main results)\n")
    for c, d in data.items():
        t = d["test"]
        print(f"{NAMES[c]} & {t['bit_accuracy']:.4f} & {t['sequence_accuracy']:.4f} "
              f"& {t['levenshtein']:.1f} & {t['bleu']:.4f} & {t['rouge_l']:.4f} "
              f"& {d['best_epoch']} \\\\")

    print("\n## LaTeX: Table 2 (training behaviour)\n")
    for c, d in data.items():
        h = d["history"]
        be = d["best_epoch"] - 1
        tl = h["train_loss"][be]
        vl_best = h["val_loss"][be]
        vl_final = h["val_loss"][-1]
        print(f"{c} & {tl:.3f} & {vl_best:.3f} & {vl_best - tl:+.3f} & {vl_final:.3f} \\\\")


if __name__ == "__main__":
    main()
