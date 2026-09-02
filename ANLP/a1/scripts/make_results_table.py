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
        # Check both outputs/C1/results.json and outputs/C1/C1/results.json
        p_nested = os.path.join(out_dir, c, c, "results.json")
        p_top = os.path.join(out_dir, c, "results.json")
        if os.path.exists(p_nested):
            out[c] = json.load(open(p_nested))
        elif os.path.exists(p_top):
            out[c] = json.load(open(p_top))
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
        t = d.get("test", {})
        bit_acc = t.get("bit_accuracy", 0.0)
        seq_acc = t.get("sequence_accuracy", 0.0)
        lev = t.get("levenshtein", 0.0)
        bleu = t.get("bleu", 0.0)
        rouge = t.get("rouge_l", 0.0)
        best_ep = d.get("best_epoch", "-")
        print(f"| {NAMES[c]} | {bit_acc:.4f} | {seq_acc:.4f} "
              f"| {lev:.1f} | {bleu:.4f} | {rouge:.4f} "
              f"| {best_ep} |")

    print("\n## LaTeX: Table 1 (main results)\n")
    for c, d in data.items():
        t = d.get("test", {})
        bit_acc = t.get("bit_accuracy", 0.0)
        seq_acc = t.get("sequence_accuracy", 0.0)
        lev = t.get("levenshtein", 0.0)
        bleu = t.get("bleu", 0.0)
        rouge = t.get("rouge_l", 0.0)
        best_ep = d.get("best_epoch", "-")
        print(f"{NAMES[c]} & {bit_acc:.4f} & {seq_acc:.4f} "
              f"& {lev:.1f} & {bleu:.4f} & {rouge:.4f} "
              f"& {best_ep} \\\\")

    print("\n## LaTeX: Table 2 (training behaviour)\n")
    for c, d in data.items():
        h = d.get("history", {})
        best_ep = d.get("best_epoch", 1)
        be = max(0, best_ep - 1)
        if "train_loss" in h and "val_loss" in h and len(h["train_loss"]) > be and len(h["val_loss"]) > be:
            tl = h["train_loss"][be]
            vl_best = h["val_loss"][be]
            vl_final = h["val_loss"][-1]
            print(f"{c} & {tl:.3f} & {vl_best:.3f} & {vl_best - tl:+.3f} & {vl_final:.3f} \\\\")


if __name__ == "__main__":
    main()
