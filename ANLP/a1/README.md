# ANLP M26 — Assignment 1: Transformer Ablation Study (from scratch)

Learned decryption of a repeating-key XOR cipher: a Seq2Seq Transformer built
**from scratch** (no `nn.Transformer` / `nn.MultiheadAttention`) maps the
bit-string ciphertext in `data/brown_cipher.txt` to the plaintext in
`data/brown_plain.txt`.

The cipher is `C[i] = P[i] XOR K[i mod 8]` with key `"ANLP2026"` (verified on
all 5000 lines). The theoretical optimum is 100% bit/sequence accuracy, so the
test set is a pure model-quality probe.

## Ablation configs

| Config | Positional encoding | Attention     | Normalization | Tokenization |
|--------|--------------------|---------------|---------------|--------------|
| C1     | Sinusoidal         | MHA           | LayerNorm     | BPE subword (512) + byte target |
| C2     | RoPE               | MHA           | LayerNorm     | BPE subword (512) + byte target |
| C3     | Sinusoidal         | GQA (kv=4)    | LayerNorm     | BPE subword (512) + byte target |
| C4     | Sinusoidal         | MHA           | RMSNorm       | BPE subword (512) + byte target |
| C5     | Sinusoidal         | MHA           | LayerNorm     | BLT token-free (raw bytes, fixed 4-byte patches) |

Shared: `dim=256, heads=8, layers=4, dim_ff=1024, dropout=0.1, lr=5e-4
(warmup 250 + cosine to 1e-5), AdamW (wd 0.01), grad clip 1.0, effective
batch 16 (8 x 2 accumulation on 11 GB GPUs; 4 x 4 or 2 x 8 on 6 GB cards),
40 epochs (C1–C4) / 120-epoch budget (C5), max_src 1024 tokens, seed 42`.

## Outputs (per config, in `outputs/<C>/`)

- `results.json` — final test metrics + full training history
- `model_best.pt` / `model_last.pt` — checkpoints (best by val loss)
- `training_curves.png` — loss / bit acc / seq acc / Levenshtein / BLEU / ROUGE-L
- `test_samples.txt`, `val_samples_epoch1.txt` — reference/prediction samples
- `config.json`, `tokenizers/` — exact config + learned BPE tokenizers

Metrics: bit accuracy (shorter predictions zero-padded to the reference
length), sequence accuracy (exact match), mean Levenshtein distance, corpus
BLEU with brevity penalty and mean sentence ROUGE-L F1 (char n-grams). All
reported metrics are greedy-decoded.


## Links

WandB: <https://wandb.ai/irishbumfuzzle-team/anlp-assignment1>

Hugging Face (best + last checkpoints, tokenizers, results):
<https://huggingface.co/IrishBumfuzzle/anlp-a1-C1>
<https://huggingface.co/IrishBumfuzzle/anlp-a1-C2>
<https://huggingface.co/IrishBumfuzzle/anlp-a1-C3>
<https://huggingface.co/IrishBumfuzzle/anlp-a1-C4>
<https://huggingface.co/IrishBumfuzzle/anlp-a1-C5>