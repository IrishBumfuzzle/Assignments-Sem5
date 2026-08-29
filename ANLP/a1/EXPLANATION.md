# Assignment 1 — Complete Technical Explanation

**What this is:** a sequence-to-sequence Transformer, built entirely from
scratch (no `nn.Transformer` / `nn.MultiheadAttention` anywhere), that learns to
*decrypt* a repeating-key XOR cipher. It is the implementation behind the
5-configuration ablation study required by `ANLP_M26_A1.md`.

**How to read this file:** §1–2 set up the task and the experiment; §3 is the
tokenization design (the single biggest lever in the study); §4 is every
architecture component with the math and the reason for each choice; §5 is the
training recipe; §6 is evaluation and metrics; §7 is the exposure-bias
diagnosis; §8 is the results and their interpretation; §9 is the full
hyperparameter reference; §10 is the code map and reproduction; §11 lists the
gotchas.

---

## Table of contents

1. [The task](#1-the-task)
2. [The five ablation configurations](#2-the-five-ablation-configurations)
3. [Tokenization design (the dominant lever)](#3-tokenization-design-the-dominant-lever)
4. [Architecture, component by component](#4-architecture-component-by-component)
5. [Training recipe](#5-training-recipe)
6. [Evaluation and metrics](#6-evaluation-and-metrics)
7. [The exposure-bias diagnosis (8000-vocab plateau)](#7-the-exposure-bias-diagnosis)
8. [Results and interpretation](#8-results-and-interpretation)
9. [Hyperparameter reference](#9-hyperparameter-reference)
10. [Code map and reproduction](#10-code-map-and-reproduction)
11. [Gotchas and known issues](#11-gotchas-and-known-issues)

---

## 1. The task

### 1.1 The data

`data/brown_plain.txt` and `data/brown_cipher.txt` each contain **5000 lines**:
sentences from the Brown corpus and their ciphertext. The ciphertext line is a
bit string — 8 characters per plaintext byte — produced by the **repeating-key
XOR cipher**

```
C[i] = P[i] XOR K[i mod 8],      K = b"ANLP2026"   (8-byte key)
```

`load_pairs()` in `src/dataset.py` re-derives `C[i] XOR K[i mod 8]` for all
5000 lines at load time and *asserts* the mapping holds, so the data is
verified before anything else runs.

### 1.2 Why this is a clean model-quality probe

- The cipher is **deterministic and invertible**, so the theoretical optimum is
  exactly **100% bit accuracy and 100% sequence accuracy**. Every point below
  100% is pure model failure — there is no irreducible noise to argue about.
- The plaintext is real English, so a model that has *given up on decryption*
  falls back to the English language prior. The "always predict space"
  baseline scores **0.662 bit accuracy** (space is the most frequent English
  character). Any real progress must visibly clear that floor; several failed
  variants in this project (BPE targets, 8000-vocab greedy decoding) sat right
  on it, which is how they were diagnosed as failures rather than slow
  learners.
- Because `i mod 8` is a *known* function of position, the task secretly
  contains a free per-position label (the key phase) — which is exactly what
  the tokenizer exploits in §3.1.

### 1.3 Data splits

Deterministic, seed 42: a `np.random.default_rng(42).permutation` of the 5000
pairs, then 80/10/10 → 4000/500/500. Lines longer than the cap are dropped, not
truncated (for C1–C4 the cap is 1024 *BPE tokens* of source → **468 test
lines**; for C5 the cap is 1024 *raw bytes* → **426 test lines**). Training
sizes: C1 3789 lines, C5 3506 lines. Every run logs its split sizes to
`results.json`.

---

## 2. The five ablation configurations

Per the assignment: start from a base model **C1** and change *exactly one*
component at a time. Everything else — depth, width, LR, batch size, schedule,
seed — is shared so the effect of each component is measured in isolation.

| Config | Positional encoding | Attention | Normalization | Tokenization |
|--------|--------------------|-----------|---------------|--------------|
| **C1** Base | Sinusoidal | Multi-Head (8 heads) | LayerNorm | BPE subword (512) + byte target |
| **C2** | **RoPE** | Multi-Head | LayerNorm | as C1 |
| **C3** | Sinusoidal | **Grouped-Query (kv=4)** | LayerNorm | as C1 |
| **C4** | Sinusoidal | Multi-Head | **RMSNorm** | as C1 |
| **C5** BLT | Sinusoidal | Multi-Head | LayerNorm | **Token-free** (raw bytes, 4-byte patches) |

Implementation: `CONFIGS` in `src/train.py` maps each name to a spec dict;
`build_model()` turns the spec into a `TransformerConfig` + model class. C1–C4
all instantiate `Seq2SeqTransformer` with only the changed field different; C5
instantiates `BLTModel`.

Shared skeleton (the "consistent hyperparameters" the assignment requires):

```
d_model = 256,  n_heads = 8 (C3: n_kv_heads = 4),  n_layers = 4 (encoder + 4 decoder),
d_ff = 1024 (4×d),  dropout = 0.1,  max_src = 1024
```

---

## 3. Tokenization design (the dominant lever)

The single biggest result of this project is that **tokenization dominates
architecture**: 8000→512 BPE vocab is worth +0.29–0.31 bit accuracy for every
autoregressive config, versus ≤0.33 for any architectural change. The design
below is the consequence of that finding, and each piece has a specific reason.

### 3.1 Source side (C1–C4): learned BPE over a *phase-annotated* byte alphabet

**Cipher structure.** Byte `i` decrypts as `C[i] XOR K[i mod 8]`. The decrypt
operation needs two things per position: the byte value `b` and the phase
`i mod 8`.

**Phase annotation (`PhaseByteBPE` in `src/dataset.py`).** Before BPE, each
cipher byte is mapped to a *unique* symbol `(b, i mod 8)` with codepoint
`b*256 + (i mod 8)` — a **2048-symbol base alphabet**. Then standard BPE is
learned *over this alphabet* (HuggingFace `tokenizers`), producing
variable-length merged tokens. This is genuinely learned subword
tokenization (0 UNK on all splits; merges are data-driven), **not** the
fixed-width 8-bit chunking the assignment explicitly disqualifies.

**Why annotate the phase?** The phase is a deterministic function of position,
so annotating adds *no new information* — but it converts decryption from a
two-part problem ("guess `i mod 8` from context, then XOR") into a
**per-symbol lookup** (`b XOR K[p]` is a constant-time table entry each token
can perform independently). Empirically, a probe of the *same* model with
BPE over raw (un-annotated) cipher bytes stalled at ~0.62 bit accuracy, while
the phase-annotated source reached 0.89–0.90 teacher-forced by epoch 28. The
phase is free information; the 25-epoch probes showed the model does not
regress it reliably from context alone.

**Why vocab 512 (and not 8000)?** Only **424 of the 2048** base symbols occur
in the data, so a 512-vocab BPE is 424 base symbols + **88 learned merges** —
within a few percent of a 1:1 token↔byte correspondence (mean 508 tokens/line
vs 213 at vocab 8000). The 1:1 correspondence matters for a deep reason
documented in §7: with multi-byte source tokens, the decoder's *own prefix*
carries the token-index→byte-index mapping, and a wrong prefix corrupts both
content **and** ordinal alignment, making mislocalization permanent. The
8000-vocab runs (retained in `outputs/ablations/vocab8000/`) hit exactly this:
all four autoregressive configs plateaued at ~0.69 greedy bit accuracy. At 512
the same configs reach 0.9663–0.9996. Side benefit: the smaller input
embedding cuts the parameter count from 9.54M to 7.63M.

### 3.2 Target side (C1–C4): one token per plaintext **byte**

The first attempt tokenized the plaintext with a *second* BPE vocabulary
(cipher-BPE → plain-BPE). It collapsed to the English prior: **0.668 bit
accuracy** (vs 0.662 always-space) after 40 epochs, with teacher-forced loss
still falling. The diagnosis: two *independent* BPE segmentations of the same
text disagree — the boundary overlap (IoU) between the two segmentations on
matched lines is **0.41**. Before the model could learn the cipher it had to
learn an *ordinal alignment between segmentations* (which plain token j
corresponds to which cipher token i), a harder problem than the cipher
itself, and the 7.6M parameters spent on it.

**Fix:** target = one token per plaintext **byte** (256-way alphabet +
specials `BYTE_BOS=257`, `BYTE_EOS=258`, `BYTE_PAD=256`;
`tgt_vocab = 259`). Target position `i` is then exactly plaintext byte `i`:
the alignment is the **identity**, matching the per-position, per-phase
structure of the cipher. This was the second-biggest single improvement in the
study (0.668 → 0.927 teacher-forced at 8000, and 0.9996 greedy at 512).

**Consequence:** sequences are long — ~500–2670 target tokens. Training works
because of activation checkpointing (§5.4) and the fp16 path (§5.5); greedy
decoding at inference is O(T²) in the target length per line, so the scoring
harness caps C1–C4 test decoding at 1024 bytes (§6.1).

### 3.3 C5: no tokenization at all (token-free)

Per the assignment, C5 consumes **raw cipher bytes 0–255** directly — the
token-free BLT design. No BPE, no vocabulary, no chunking-as-tokenization:
bytes are grouped into fixed 4-byte *patches*, which are an architectural
device of the BLT model, not a tokenizer (§4.6).

---

## 4. Architecture, component by component

All components are implemented from scratch in `src/models/`:

| file | from-scratch components |
|------|--------------------------|
| `positional.py` | `SinusoidalPE`, `RotaryPositionalEmbedding` (RoPE) |
| `attention.py` | `ScaledDotProductAttention`, `MultiHeadAttention` (MHA + GQA), `FeedForward`, encoder/decoder layers |
| `norm.py` | `LayerNorm`, `RMSNorm` |
| `transformer.py` | `TransformerEncoder`, `TransformerDecoder`, `Seq2SeqTransformer` (C1–C4), `BLTModel` (C5) |
| `blt.py` | `LocalByteEncoder`, `LocalByteDecoder` (C5) |
| `entropy_patching.py` | entropy LM + dynamic patch boundaries (C5 variant) |

### 4.1 Scaled dot-product attention (`attention.py`)

```
scores = Q Kᵀ / √d_k          (d_k = head_dim = d_model / n_heads = 32)
attn   = softmax(scores + mask)
out    = attn V
```

- **Why the √d_k scaling:** for unit-variance q, k the dot product has
  variance d_k; without the division, softmax saturates (one hot entry) and
  gradients vanish. Dividing keeps the logits in the softmax's sensitive
  region.
- **Masking convention:** boolean mask, `True` = *allowed to attend*;
  disallowed positions get `-inf` before softmax. **Every query row must keep
  at least one allowed key** or the softmax over all `-inf` produces NaN whose
  gradient poisons the whole batch. The BLT code paths are careful about this
  (padding queries are given a dummy key — see §4.6).
- **Softmax in fp32:** computed with `scores.float()` then cast back — fp16
  softmax over long rows is where mixed-precision training loses precision
  first.

### 4.2 Multi-Head Attention (MHA)

Q/K/V/O are **bias-free** `nn.Linear` projections; the head vector
`d_model × 1` is reshaped to `n_heads × head_dim` (8 × 32). No biases follow
the standard modern convention (GPT-2 onwards) and save 4·d² parameters per
layer.

**Why multiple heads:** each head can implement a different attention pattern
in its own 32-dim subspace; the concatenation + `W_o` recombines. On this task
heads also carry the *position-locating* computation (which source position the
current output position reads) — which is why degrading them (GQA) costs
something, §4.3.

### 4.3 Grouped-Query Attention (GQA, C3)

**Theory:** GQA (Ainslie et al. 2023, the intermediate point between MHA and
MQA) gives each *group* of query heads a **shared** key/value head pair. With
`n_heads=8`, `n_kv_heads=4`, each KV head serves 2 query heads (`group=2`).
Effect: K/V projections shrink to `d_model × (n_kv_heads·head_dim)`, cutting
parameters and (at inference) the KV-cache size, at a small quality cost.

**Implementation details:**

- `W_k`, `W_v` output `n_kv_heads·head_dim` instead of `d_model`;
- after projection, K/V are expanded to the full head count with
  `repeat_interleave(group, dim=1)`, so the rest of the attention is
  literally MHA on the expanded tensors;
- the returned **KV cache stores K/V after expansion**, so incremental
  decoding (§4.7) needs no bookkeeping about which query head maps to which KV
  head.

Parameter cost measured in this project: 7.63M → 7.10M (−0.53M), a quality cost
of 0.27 bit / 7.9 sequence points (§8).

### 4.4 Positional encodings (`positional.py`)

#### Sinusoidal (C1, C3, C4, C5)

```
pe(pos, 2i)   = sin(pos / 10000^(2i/d))
pe(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Fixed (unlearned) table, `register_buffer`, **added** to the embedding. The
geometric-frequency design gives each position a unique, smoothly varying code
whose dot-product kernel depends on position *differences* (which is what makes
relative offsets learnable), while still assigning every position an **absolute
identity**.

#### RoPE (C2)

NeoX-style half-rotation applied **only to the Q and K head vectors** inside
attention (never added to the embedding):

```
(x1, x2) = head vector split in halves
RoPE_p(x) = (x1 cos pθ − x2 sin pθ,  x2 cos pθ + x1 sin pθ),   θ_i = 10000^(−2i/d)
```

**Why RoPE is theoretically relative:** rotating q at position `i` and k at
position `j` makes the attention score depend on the vectors only through the
relative angle `(i − j)θ`. It therefore encodes *offsets* intrinsically and
extrapolates well in relative terms.

**Why RoPE loses on this task (C2 is the worst config):** decryption requires
*absolute* target↔source alignment — output position `i` must read source
position `i` — over up to ~2600 positions. With sinusoidal codes, position has
a stable identity **in both sequences simultaneously**, and even when the
decoder's own (wrong) prefix is corrupted, the PE still tells it where it is.
With RoPE, absolute position must be *integrated* from the prefix's relative
offsets — and if the prefix content is wrong, the integrated position is wrong
too. The observed failure mode (teacher-forced 0.9998 by epoch 30, yet greedy
stalled ~0.68 for 20 epochs, then late-line **repetition loops**) is exactly
"lost the position, started looping." For alignment-heavy long-sequence tasks
the absolute encoding is the safer choice — the reverse of the general-purpose
LLM default, where the mapping is semantic and relative offsets transfer.

### 4.5 Normalization (`norm.py`)

#### LayerNorm

```
LN(x) = (x − mean(x)) / √(var(x) + ε) · γ + β        (ε = 1e-5)
```

Centering **and** rescaling, with learned affine `γ, β`. Implemented with
explicit mean/variance ops (not a wrapper around `nn.LayerNorm`).

#### RMSNorm (C4)

```
RMSNorm(x) = x / √(mean(x²) + ε) · γ        (ε = 1e-6, no bias)
```

Drops the centering (and therefore the bias). Theory: Zhai et al. (2023)
showed mean-centering contributes little to the benefit — the rescaling of the
per-neuron RMS is what stabilizes layers — so removing it costs nothing in
quality at scale while being ~15% faster (no mean pass, no bias). It is the
normalization of the LLaMA family.

**Why both compute statistics in fp32:** under fp16 autocast, `mean`/`var`/
`rms` over a 256-dim vector accumulate rounding error that biases the
normalized values; computing in fp32 and casting the affine part back makes
mixed precision safe. Under pure fp32/bf16 this is a numerical no-op.

#### Pre-LN block layout (all configs)

```
x = x + Dropout(Attn(Norm(x)))
x = x + Dropout(FFN(Norm(x)))
...
out = FinalNorm(x)
```

- **Pre-LN vs Post-LN:** post-LN (normalize the *residual sum*) requires
  careful warmup and is unstable early in training because the first layers
  operate on raw (large-variance) embeddings; pre-LN keeps every residual
  stream at stable scale from step 0, which is why modern transformer
  training uses it with only a short warmup.
- **Final norm:** pre-LN models have the *output* of the last residual path
  unnormalized; the trailing `final_norm` (after the loop) restores comparable
  scale into the output head. It is the standard companion to pre-LN.
- The decoder layer is the three-sublayer variant: causal self-attention →
  cross-attention to encoder memory → FFN, each with its own norm.

### 4.6 Feed-Forward Network (`attention.py`)

```
FFN(x) = W2 · Dropout(GELU(W1 x))        (d_model → 4·d_model → d_model)
```

The position-wise MLP is where most transformer parameters live (2·d·4d per
layer vs 4·d·d for attention) and where much of the *per-position*
transformation happens — attention mixes information *between* positions, the
FFN transforms the representation *at* each position. 4× width is the
Vaswani-era standard; GELU is the smooth nonlinearity standard for
transformers.

### 4.7 Encoder/decoder stack (`transformer.py`)

#### Embeddings and the *missing* √d scale

`nn.Embedding` with `padding_idx` (rows stay zero and receive no gradient).
**Note the deliberate omission of the `x·√d_model` embedding scale** that the
original paper uses. With pre-LN blocks and *additive sinusoidal* PE, scaling
the token embedding by √256≈16 makes the positional signal ~16× weaker than
the token signal, which drastically slowed learning of position-sensitive
mappings (verified on a synthetic copy task). Modern pre-norm models (GPT-2
onwards) omit the scale entirely; we do the same.

#### Causal mask (decoder)

`causal_mask(Lq, Lk)` returns `idx_k <= Lk − Lq + idx_q` — the lower-left
triangular alignment that generalizes to both teacher-forced (Lq = Lk) and
incremental (Lq = 1) shapes. During a single-token cached step the mask is
`None`: the new query attends to the full cache, which is already causally
valid.

#### Padding

Source padding is masked on the *key* side (`src_mask` broadcast to
`(B,1,1,S)`) so no position ever attends into padding. Target padding positions
are excluded from the loss via `ignore_index`, not by masking (their logits
are computed but discarded).

#### KV cache and incremental decoding

`Seq2SeqTransformer.generate()` decodes greedily with a **per-layer KV cache**:
step `t` computes attention for the single new query against all cached keys,
so the full decode is O(T²) *FLOPs* instead of O(T³).

Two cache formats exist in `MultiHeadAttention`:

1. **2-tuple** `(k, v)` — `torch.cat` the new rows onto the cache each step.
   Simple, but each step copies O(t) data, making long greedy decodes O(T²) in
   *wall time* (memory traffic, not FLOPs).
2. **3-tuple** `(k_buf, v_buf, write_pos)` — a **pre-allocated** buffer of
   shape `(B, H, max_len, head_dim)`; each step writes the new rows *in place*
   with `copy_` and exposes the grown prefix as a **view**. Zero per-step
   allocation. This is the format `generate()` uses, and it is the difference
   between a 2670-step decode taking minutes or seconds.

Cache dtype follows autocast (fp16 under fp16 training, bf16 under bf16,
fp32 otherwise) so the cache never up-casts on every access.

#### Finished sequences

Rows that emit EOS early get EOS fed back (keeps the batch rectangular);
outputs are truncated at the first EOS after the loop, then
`pad_sequence` re-packs the ragged rows.

#### Activation checkpointing

Every encoder/decoder block is wrapped in
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` **on the
training path only** (detected via `torch.is_grad_enabled() and
h.requires_grad`). Trade: ~30% extra compute (blocks recompute in backward)
for not storing the intermediate activations — critically, the `T×T` attention
matrices of every layer — which is what makes 2600-token byte targets fit an
11 GB card at batch 8. Two subtleties handled in the code:

- loop variables (`layer`, `mask`, `pos`) are bound as **default arguments**
  of the checkpointed lambda; a bare closure would *late-bind* and recompute
  the *last* layer in every node during backward (a silent correctness bug);
- the incremental KV-cache decoding path is deliberately **not** checkpointed
  (no gradients there; checkpointing would only slow it down).

### 4.8 C5: the Byte Latent Transformer (`blt.py`, `entropy_patching.py`)

C5 follows the BLT paper (Pagnoni et al., arXiv:2412.09871, "Patches Scale
Better Than Tokens") in simplified form, adapted from a language model to a
seq2seq decoder. Core idea: **token-free** processing — raw bytes are grouped
into *patches*; a local encoder compresses each patch to one latent vector; a
global transformer operates over latents (short sequences → cheap attention);
a local decoder expands each latent back to per-byte distributions.

```
raw bytes (B, L)
  → LocalByteEncoder ──────────────→ latents (B, Np, 256)     Np ≈ L/4
      (byte + n-gram embeddings,
       banded local attention,
       cross-attention pooling)
  → global TransformerEncoder over latents (same 4/8/256 budget as C1–C4,
    repr_input=True → no embedding layer, sinusoidal PE over patch index)
  → LocalByteDecoder ──────────────→ byte logits (B, L, 256)
      (byte slots query their own patch latent,
       banded local mixing, byte head)
```

**Decoding is non-autoregressive.** Because the target length equals the
cipher length byte-for-byte, the model emits all 256-way distributions in a
**single forward pass** — no prefix is fed back, so the train/decode mismatch
(exposure bias, §7) does not exist *by construction*. This is the structural
reason C5 is ~5× faster than C1 (§8).

#### LocalByteEncoder (one latent per patch)

- **Byte embedding:** `nn.Embedding(257, byte_dim=64)` — 256 real byte values +
  one learned `BYTE_PAD=256` row.
- **Rolling-hash byte n-gram embeddings** (the BLT paper's substitute for
  subword context, with *no vocabulary*): for each n ∈ {3,4,5,6,7,8} the code
  `rolling_ngram_indices` computes, per position `i`, the rolling polynomial
  hash of the n-gram *ending* at `i`:
  `h = b[i]·BASE^(n−1) + b[i−1]·BASE^(n−2) + … + b[i−n+1]  (mod 2³¹−1)`
  with `BASE=131` and Mersenne-prime modulus (fast modular arithmetic, low
  collision rate for n ≤ 8). Each hash indexes its own `nn.Embedding(4096,
  byte_dim)` table; the six n-gram vectors plus the byte embedding are summed
  and divided by `len(ns)+1 = 7` (the paper's normalization). This gives every
  byte a *contextual* representation (its surrounding 1–7 bytes) with no
  learned vocabulary at all — the token-free analogue of BPE giving each token
  multi-byte context. Incomplete n-grams at line start are masked to 0.
- **Banded causal local attention:** the `n_local_layers=2` byte-level
  transformer layers (width 64, 4 heads, FFN 256) use a **banded causal mask**:
  byte `i` attends to at most the 16 preceding bytes (window ≥ max patch size,
  so a patch never needs information from outside its neighborhood). Banded
  attention makes the local layers O(L·window) instead of O(L²) and encodes the
  inductive bias that byte-level structure is local. The window may cross patch
  boundaries (useful context) but never line boundaries (right padding keeps
  rows isolated; a diagonal escape-hatch keeps padded queries' rows finite —
  the "at least one allowed key" rule of §4.1).
- **Cross-attention pooling (Perceiver-style):** each patch is a *query*
  initialized by the mean of its own bytes' masked embeddings, projected to
  `d_model=256`; at each local layer the query cross-attends **only to the
  bytes of its own patch** (`enc_x_mask`) and adds the result residually.
  Padded patches get a dummy key (byte 0) so their rows stay finite, and are
  zeroed out by `patch_mask` afterward. Output: one 256-dim latent per patch,
  plus the final byte-level representation `h_last` (handed to the decoder).

#### Global encoder

The same `TransformerEncoder` as C1–C4 (4 layers, 8 heads, 256 dim, sinusoidal
PE, pre-LayerNorm, activation checkpointing), called with `repr_input=True` so
it skips the token embedding and operates directly on patch latents. Sequence
length is ~L/4 — e.g. a 1024-byte line is ~256 patch queries instead of ~1000
BPE tokens — which is where the compute and memory win comes from.

#### LocalByteDecoder (latent → byte logits)

Roles reversed from the encoder (paper §3.3):

- **Byte slots** are the queries: each byte's *final encoder representation*
  (`h_last`) plus an **intra-patch slot-offset embedding** (position of the
  byte inside its patch, 0..max_patch−1 — the local analogue of a positional
  encoding, disambiguating the bytes *within* a patch).
- Each slot cross-attends to **its own patch's latent only** (single-key
  cross-attention; `dec_x_mask`), then a banded 16-byte causal local layer
  mixes neighboring slots. One decoder local layer.
- A linear `byte_head` (64 → 256) outputs the per-byte distributions, aligned
  **1:1** with source positions. Padding slots are masked to 0 and dropped
  (`predict_bytes` slices each row to its true length).

#### Patching: fixed (official) vs entropy-dynamic (explored variant)

- **Fixed stride 4 (official C5)** — `fixed_stride(L, 4)` in
  `entropy_patching.py`: consecutive 4-byte groups, final patch holds the
  remaining 1–4 bytes. This is the BLT paper's base scheme and the official
  configuration (`--patching fixed`, the default; checkpoint saves
  `patching = {method: "fixed", patch_size: 4}`).
- **Entropy-based dynamic patching (C5 variant, negative result)** — the BLT
  paper §2.3/§4.2–4.4 scheme, implemented faithfully in
  `entropy_patching.py`:
  1. A small **entropy LM** (`ByteEntropyLM`: 2 layers, 4 heads, 128 dim,
     256-byte sliding-window causal attention, 256-way head, START token 256)
     is trained on the train split first. The patch structure is a function of
     the *source* alone, so it is identically available at training and
     inference with no target leakage.
  2. **Next-byte entropies** `H(x_i)` (bits) are computed for every position,
     with the LM context **reset at each line start** (paper §4.4: avoids
     entropy drift across document boundaries).
  3. A new patch starts at `t>0` iff `H(x_t) > θ_g` (global constraint) or
     `H(x_t) − H(x_{t−1}) > θ_r` (approx-monotonicity constraint, θ_r = 1.0),
     or the current patch reached `max_patch = 12` bytes.
  4. **`θ_g` is calibrated by bisection** on the train lines so the *mean*
     patch size hits the target 4.0 bytes — this makes the dynamic scheme
     compute-matched to fixed stride-4 (paper §4.3). Calibrated value on this
     data: **θ_g ≈ 4.6332** (mean patch 4.0, max 12, ~436k patches on train).
  5. The scheme satisfies **incremental patching** (`f_p(x_<i) = f_p(x)_<i`):
     boundary decisions at `t` use only bytes `< t`, so segmenting a prefix
     and segmenting the whole line agree. `verify_incremental()` property-tests
     this on 200 random entropy sequences before any entropy run is allowed
     to start; `patch_stats` logs the per-split patch-length distributions
     (exact float histogram edges — a previous lossy "rounded centers" format
     once crashed `wandb.Histogram` server-side).
- **Result:** the dynamic variant (120-epoch run, `outputs/ablations/c5-entropy/`)
  scored **0.9931 bit / 0.0305 seq** vs fixed's **0.9999 / 0.9343**. The
  entropy-triggered boundaries scatter single-byte errors across a line —
  bit accuracy survives, *exact-sequence* accuracy collapses. Fixed stride is
  therefore official. The variant remains fully supported via
  `--patching entropy`.

---

## 5. Training recipe

Single entry point: `src/train.py`. All decisions below are reflected in CLI
flags; §9 lists the official values.

### 5.1 Loss

`nn.CrossEntropyLoss(ignore_index=pad)`, applied to flattened
`(B·T, V)` logits vs `tgt[:, 1:]` (teacher forcing: the decoder input is the
true prefix `tgt[:, :-1]`). `ignore_index` is `BYTE_PAD` (256) for byte-target
and C5 runs, `PAD_ID` (0) for BPE-target runs. No label smoothing — with a
deterministic target, smoothing would only add a constant floor to a problem
whose optimum is 0.

### 5.2 Optimizer and schedule

- **AdamW**, weight decay `1e-2`, PyTorch default β = (0.9, 0.999) — the code
  calls `AdamW(params, lr, weight_decay)` with no `betas` override.
  (The report states β₂ = 0.98; the executed runs used the default 0.999 —
  noted here so the doc matches the artifacts.)
- **Schedule (`make_scheduler`):** 250-step **linear warmup** (from ~0 to the
  peak), then **cosine decay** from `5e-4` to a floor of `1e-5` over the whole
  epoch budget. Warmup is needed because Adam's second-moment estimates are
  unreliable in the first steps (large updates early in pre-LN transformer
  training destabilize layer 1); cosine-to-floor (rather than to zero) keeps a
  non-negligible effective LR near the end, which is what lets C1–C4 keep
  climbing to epoch 40.
- **Gradient clipping** at norm 1.0 after unscaling — standard for fp16
  stability.
- **Total steps** = `steps_per_epoch × epochs`, so the cosine shape is
  identical in step count for a given budget — which is why C5's run must keep
  its 120-epoch budget even though it was stopped at 116 (changing the budget
  reshapes the schedule, not just the endpoint).

### 5.3 Batching: effective batch 16, and why *random* batches

- C1–C4: micro-batch **8 × 2 gradient-accumulation steps** = effective 16.
  The micro-batch is small because byte targets are up to ~2670 tokens long
  (the KV/activation memory of the decoder scales with target length).
- C5: micro-batch **16 × 1** = effective 16 — no accumulation needed because a
  1024-byte line is only ~256 patch latents through the global encoder.
- **Random (shuffled) batches, *not* length bucketing.** This was a genuine
  experimental finding, not a convenience: with length-homogeneous buckets
  (`LengthBatchSampler` is implemented and available behind
  `--length-bucketing`), alignment learning **stalled** (~0.67 bit accuracy by
  epoch 6); the identical data and schedule with shuffled random batches
  reached ~0.83 by epoch 6. Theory: a bucketed update sees one length band per
  step, so the model never has to consolidate a *single* alignment rule across
  all lengths; the cipher mapping must hold at every length simultaneously,
  and mixing lengths in every batch forces that generalization. The cost
  (padding to the batch max) is real but small next to the benefit.
- `drop_last` is off by default (every line is trained on).

### 5.4 Mixed precision (auto-selected by compute capability)

`get_amp_settings()`:

| GPU class | dtype | GradScaler | why |
|-----------|-------|------------|-----|
| Ampere+ (cc ≥ 8.0) | **bf16** | no | bf16 has fp32's exponent range — no overflow, no scaler needed |
| Turing / V100 / 2080 Ti (cc < 8.0) | **fp16** | **yes** | fp16's 65504 max overflows on long-attention activations; the scaler scales losses down and unscales gradients, skipping steps with inf/norm (these cards lack bf16 tensor cores) |
| CPU | none | no | — |

Complementary fp32 islands: softmax in attention, mean/var/rms in the norm
modules, entropy computation in the patching phase.

### 5.5 Checkpointing and selection

- `model_best.pt` saved whenever **val loss** improves (val loss = the
  teacher-forced cross-entropy on the val set; for C1–C4 the per-epoch val
  *metrics* are also teacher-forced — a cheap, monotone proxy; greedy val eval
  is expensive at ~2600 steps/line and is done every 10 epochs instead).
- `model_last.pt` every epoch. Checkpoints embed the full `args`, the
  `TransformerConfig`, and (C5) the patching info — including the entropy LM
  state for dynamic-patch runs, so `scripts/eval_checkpoint.py` can
  reconstruct the *exact* patch structure and re-score any saved model.
- **C5 horizon:** the official fixed-patch run used a 120-epoch budget and was
  stopped at epoch 116 after val loss was flat at ≈0.0010 for 25 epochs; best
  checkpoint is epoch 112. `scripts/make_c5_official.py` assembles the final
  `outputs/C5` (history + re-run test eval + artifacts) from the stopped run's
  log and `model_best.pt`.

### 5.6 Implemented-but-officially-unused exposure-bias mitigations

Two standard fixes for the teacher-forced/greedy gap were implemented and
available via CLI; both were **negative** at the 8000-vocab regime (§7) and
are not used by the official study:

- **Random prefix dropout** (`--prefix-dropout q`): during training, each
  decoder-input position (except BOS) is replaced by a random token id with
  probability `q`, forcing the model to align from positional signals rather
  than prefix content. Failed because random noise is not the distribution of
  greedy errors, which are *plausible* wrong bytes (q=0.5: greedy 0.679, TF
  0.774; q=1.0: collapse to the prior).
- **Scheduled sampling** (`--scheduled-sampling` etc.): each sample is trained
  against its *own cached greedy prefix* with probability ramping 0→0.5 over
  10 epochs; self prefixes are regenerated every 2 epochs on a dedicated
  unshuffled loader (index→prefix mapping must be stable), capped at 1024
  bytes. Failed because the model learns to distrust even correct prefixes (TF
  drops) without learning re-localization from wrong ones.

---

## 6. Evaluation and metrics

All official numbers are **greedy-decoded** on the test set, per the
assignment's consistency requirement. Implementation: `src/utils.py` (metrics,
all from scratch) and `evaluate()` in `src/train.py`.

### 6.1 Two evaluation modes (C1–C4)

- **Teacher-forced ("teacher")**: one forward pass, `logits.argmax(-1)`.
  Fast (no loop), used every epoch for best-checkpoint selection. Measures
  what the model *knows* given a correct prefix.
- **Greedy**: the real autoregressive loop with the KV cache and the model's
  own prefix — the *same* code path as test. Run every 10 epochs on val to
  track the actual decoding quality; run once at the end on test for the
  reported numbers. The gap between the two is the exposure-bias indicator
  (§7).
- C5 has no such distinction: it is a single non-autoregressive pass in both
  modes.

**Test cap:** C1–C4 greedy decoding is O(T²) per line (2670 steps × growing
cache), so scoring is capped at 1024 bytes of target (the reported C1–C4 test
set is the 468 lines that fit the 1024-token source cap; predictions longer
than 1024 bytes are truncated for scoring). C5 needs no cap (single pass;
426 lines at the 1024-byte source cap).

**Oracle verification:** the incremental KV-cache decode path was verified
bit-for-bit against teacher forcing by feeding the oracle prefix — confirming
the cache math is exact and any TF/greedy gap is genuine model behavior, not
an implementation artifact.

### 6.2 Metrics (all implemented from scratch in `src/utils.py`)

| Metric | Definition | Notes |
|--------|-----------|-------|
| **Bit accuracy** | fraction of matching bits | predictions shorter than the reference are **zero-padded**, longer ones truncated, so each reference bit contributes exactly one comparison |
| **Sequence accuracy** | fraction of *perfectly* reconstructed lines | exact string match; the strictest metric and the one that exposed the C5-dynamic error scattering |
| **Levenshtein** | mean edit distance (ins/del/sub) over the test set | vectorized O(mn) DP: the inner row recurrence is folded into a `np.minimum.accumulate` prefix-minimum, so there is no Python inner loop (a naive O(mn) Python reference is included for validation) |
| **BLEU** | corpus BLEU, **character** n-grams 1–4, with brevity penalty | character n-grams (not word) so the metric is defined identically for tokenized (C1–C4) and token-free (C5) outputs; standard clipped counting + BP |
| **ROUGE-L** | mean sentence-level F1 over the corpus | LCS-based (longest common subsequence), itself vectorized with a running `np.maximum.accumulate`; the assignment calls for these "for tokenized models only" but the char n-gram formulation works for C5 too, so all five are reported for all five configs |

The always-space baseline (0.662 bit accuracy) is the null hypothesis every
run must beat; it is the English prior's score on the test set.

### 6.3 Artifacts per config (`outputs/<C>/`)

- `results.json` — test metrics + full per-epoch history (loss, val metrics,
  throughput, peak memory)
- `model_best.pt` / `model_last.pt` — checkpoints (also on HuggingFace)
- `training_curves.png` — loss / bit acc / seq acc / Levenshtein / BLEU /
  ROUGE-L / peak memory (2×3 grid)
- `test_samples.txt` (10 REF/PRED pairs), `val_samples_epoch1.txt`
- `config.json` — exact spec + args + `n_params` + patching info
- `tokenizers/` — the learned BPE files (C1–C4)

`scripts/eval_checkpoint.py <ckpt>` re-runs the full test evaluation for any
saved checkpoint; `scripts/make_results_table.py` regenerates the report
tables from the `results.json` files; `scripts/make_report_figures.py` builds
the report figures.

---

## 7. The exposure-bias diagnosis

The 8000-vocab C1 runs exhibited a large **teacher-forced/greedy gap**:
0.927 TF vs 0.690 greedy bit accuracy at epoch 40; greedy validation nearly
flat (0.679 at epoch 10 → 0.691 at epoch 40) while TF climbed steadily.
Greedy outputs started correctly, then **degenerated into fluent but unrelated
English**. The diagnosis (full version in Report.tex §3) ran a **prefix sweep**
on a 30-line val set with the epoch-25 checkpoint, controlling exactly what
the decoder is fed as its own prefix:

| Decoder prefix | Bit acc. |
|----------------|---------|
| oracle (ground truth) | 0.893 (≈ TF 0.898 → cache path is exact) |
| 10% of prefix positions randomised | 0.791 |
| 30% / 50% / 80% randomised | 0.722 / 0.688 / 0.664 |
| 100% randomised | 0.654 (below the always-space baseline!) |
| resync to oracle every 2 / 5 / 10 bytes | 0.767 / 0.708 / 0.691 |
| pure greedy (own prefix) | 0.676 |
| always-space baseline | 0.662 |

**Reading:** the decoder uses the *content* of its own prefix to locate the
source position. One wrong byte and it never re-locates, then emits the
English prior. Randomizing 10% of the prefix costs 0.10; at 100% the model is
*worse* than always-space (it commits to fluent but wrong text). Re-anchoring
to the oracle only every 10 bytes barely helps (0.691) — the drift is
continuous, not stepwise.

**Why the two standard fixes failed** (implemented, measured, both negative —
see §5.6): random prefix dropout teaches robustness to *obviously random*
bytes, which is not the greedy error distribution; scheduled sampling corrupts
with *plausible* bytes but the model then distrusts even correct prefixes (TF
also drops). Neither teaches **re-localization from a wrong prefix**.

**The actual resolution was the tokenizer.** The 8000 plateau was exposure
failure *compounded* by multi-byte BPE resegmentation: with tokens spanning
2–5 bytes, a wrong prefix corrupts the content *and* the token-index→byte-index
mapping, so mislocalization is permanent. At the 512 vocab (1:1 tokens, §3.1)
the same architecture's TF–greedy gap shrinks from 0.184 at epoch 10 to 0.016,
and ~0 by epoch 40; greedy no longer plateaus (C1: 0.9506 → 0.9937 → 1.0000
over epochs 10/20/40). Residual exposure bias remains, most visibly in C2
(RoPE: TF 0.9998 vs greedy 0.9592, late-line repetition loops) — where the
missing *absolute* position signal (§4.4) makes re-localization from a wrong
prefix impossible in principle.

---

## 8. Results and interpretation

### 8.1 Main table (greedy test, official 512-vocab study)

| Config | Bit acc | Seq acc | Lev. ↓ | BLEU | ROUGE-L | Best epoch | Params |
|--------|---------|---------|--------|------|---------|-----------|--------|
| C1 Base | 0.9996 | **0.9722** | 0.12 | **0.9998** | **0.9999** | 40 | 7.63M |
| C2 RoPE | 0.9663 | 0.7372 | 8.26 | 0.9909 | 0.9880 | 39 | 7.63M |
| C3 GQA(4) | 0.9969 | 0.8932 | 0.34 | 0.9991 | 0.9997 | 40 | 7.10M |
| C4 RMSNorm | 0.9981 | 0.9615 | 0.22 | 0.9997 | 0.9999 | 40 | 7.62M |
| **C5 BLT (fixed 4)** | **0.9999** | 0.9343 | **0.08** | 0.9996 | 0.9999 | 112/116 | 5.55M |

Ablations (same architecture/schedule, greedy test):

| Ablation | Bit acc | Seq acc | Note |
|----------|---------|---------|------|
| C1–C4 @ BPE 8000 (40 ep, 500 lines) | 0.6773–0.6905 | 0.00–0.60% | the tokenization plateau |
| C1 @ 512, 15-ep probe | 0.9499 | 0.0833 | what prompted the full retrain |
| C5 entropy-dynamic patching (120 ep) | 0.9931 | 0.0305 | scattered single-byte errors |

Always-space baseline: **0.662** bit acc.

### 8.2 What each single-component change did (vs C1)

- **Sinusoidal → RoPE (C2): clearly negative.** −3.3 bit / −23.5 seq points;
  the worst on every axis except param count. Teacher-forced accuracy was
  near-perfect by epoch 30 (0.9993), yet greedy stalled ≈0.68 for 20 epochs
  before escaping to 0.9249/0.9592, and residual errors are **late-line
  repetition loops** ("…before transferring to No Squadron before
  transferring to No Squadron…"). Interpretation: decryption is an
  *absolute* alignment problem over hundreds of positions; relative-only
  encoding forces absolute position to be integrated from a possibly-wrong
  prefix, which is the step that fails at length. For alignment-heavy tasks,
  the absolute encoding is the safer inductive bias — the reverse of the
  general-purpose LLM default.
- **MHA → GQA(4) (C3): small, expected cost.** −0.27 bit / −7.9 seq points for
  halved KV heads and −0.53M parameters. Consistent with a softer
  cross-attention position index: with 4 shared KV heads, less
  position-locating information per head on a task that *is* position lookup.
  At larger scale GQA's inference-time KV savings usually dominate; at this
  budget MHA keeps more alignment precision.
- **LayerNorm → RMSNorm (C4): indistinguishable.** Within 1.1 seq points on
  every axis; the sensible production default (fewer ops, no centering) —
  confirms the Zhai et al. result at this scale.
- **Subword → token-free BLT (C5): the interesting trade.** Best bit accuracy
  (0.9999), best Levenshtein (0.08 on ~554-byte-median lines), 5.55M params
  (27% fewer than C1). The price is 3.8 seq points (0.9343 vs 0.9722): C5's
  residual errors are **scattered single bytes** (≈0.006% of positions), which
  barely register in bit accuracy or edit distance but each one disqualifies
  an exact match; C1's errors are rarer but come in prefix-drift *bursts*. A
  64-line overfit probe drove C5's train loss to 5e-4 at 100% bit accuracy, so
  capacity — not architecture — bounds the residual. Bonus: C5's test pass is
  its training mode (non-autoregressive), so the exposure-bias axis is removed
  entirely.

### 8.3 C5 vs C1: the memory–speed–accuracy trade (assignment's benchmark focus)

| Config | Peak mem (GB) | Per line (MB) | Epoch (s) | Throughput | Total |
|--------|---------------|---------------|-----------|------------|-------|
| C1 | 2.55 | 319 | 118 | 32 lines/s | 129 min (40 ep) |
| C5 | **4.27** | **267** | **24** | **147 lines/s** | **46 min (120 ep)** |

- **Memory:** C5's *peak* looks higher, but C5 runs 2× the micro-batch (16
  vs 8) and skips activation checkpointing (which saves C1 memory at a ~30%
  compute penalty). **Per line, C5 is lighter** (267 vs 319 MB); at batch 8
  C5 would sit at ~2.5 GB — matching C1 *without* the checkpointing penalty.
  Fewer parameters (5.55M) and patch-level global attention (sequence length
  ÷4) are the structural sources.
- **Speed:** ~5× (24 s vs 118 s per epoch) because C5 is one parallel pass per
  line vs C1's up-to-2670 sequential decode steps — the same advantage C5 has
  at inference (1 forward pass vs 2670 steps).
- **Accuracy:** best on bit acc + Levenshtein, −3.8 seq points (error-type
  effect above). Net: the token-free approach buys ~5× speed, lower per-line
  memory, and fewer parameters, at a minor exact-match tax.

---

## 9. Hyperparameter reference

Official values (from the executed runs' `config.json`; CLI flags in
parentheses).

| Parameter | C1–C4 | C5 | Why |
|-----------|-------|----|-----|
| `dim` (d_model) | 256 | 256 | shared width; fits 11 GB cards with long byte targets |
| `heads` | 8 | 8 | head_dim 32; 8 heads for the position-locating computation |
| `kv_heads` | 8 (=heads) | 8 | C3: 4 (GQA, group 2) |
| `layers` | 4+4 | 4 (global) + 2 local enc + 1 local dec | shared depth budget |
| `dim_ff` | 1024 (4×d) | 1024 global; 256 local (4× byte_dim) | Vaswani-standard ratio |
| `dropout` | 0.1 | 0.1 | |
| `pos_encoding` | sinusoidal (C2: rope) | sinusoidal over patch index | §4.4 |
| `norm` | layernorm (C4: rmsnorm) | layernorm | §4.5 |
| `vocab` (C1–C4 src) | **512** (phase-annotated BPE) | — (token-free) | §3.1 — the dominant lever |
| `tgt` (C1–C4) | **byte** (256-way + 3 specials = 259) | byte (256-way head) | §3.2 |
| `max_src_len` | 1024 (tokens) | 1024 (bytes) | line cap |
| `max_tgt_len` | longest plain line + 2 (≈2672) | = source length (1:1) | byte targets auto-sized |
| `epochs` | 40 | 120 budget (ran 116, best 112) | C5 converges slower (more steps per byte of signal) but needs the 120-ep cosine shape |
| `batch-size` | 8 | 16 | §5.3 (memory: target length vs patch latents) |
| `grad-accum` | 2 | 1 | effective batch 16 for all |
| `lr` | 5e-4 | 5e-4 | AdamW peak |
| `min-lr` | 1e-5 | 1e-5 | cosine floor |
| `warmup-steps` | 250 | 250 | §5.2 |
| `weight-decay` | 0.01 | 0.01 | AdamW (code default β=(0.9,0.999)) |
| `grad-clip` | 1.0 | 1.0 | fp16 stability |
| `seed` | 42 | 42 | deterministic splits & init |
| `patch-size` | — | **4** (fixed) | BLT patch; final patch 1–4 bytes |
| `byte-dim` | — | 64 | local enc/dec width |
| `local-layers` / `local-heads` | — | 2 / 4 (enc), 1 (dec) | |
| `local-window` | — | 16 | banded causal window ≥ max patch |
| `ngram-table` | — | 4096 | hash-embedding table (n = 3..8, base 131, mod 2³¹−1) |
| `max-patch` | — | 12 | entropy-variant hard cap (fixed: 4) |
| `target-patch-size` / `theta-r` | — | 4.0 / 1.0 | entropy-variant calibration targets |
| calibrated `theta_g` | — | ≈4.6332 | bisection on train (mean patch 4.0) |
| entropy LM | — | 2 layers, 4 heads, 128 dim, FFN 256, 256-byte window, 40 epochs, trained on train split only | §4.8 |

---

## 10. Code map and reproduction

```
src/
  dataset.py            PhaseByteBPE (+ 2048 phase-annotated alphabet), BPETextTokenizer,
                        ByteTargetCipherDataset / ByteCipherDataset / TokenizedCipherDataset,
                        LengthBatchSampler (off by default), collates, load_pairs (XOR verify)
  models/
    positional.py       SinusoidalPE, RoPE (NeoX half-rotation, applied to Q/K)
    attention.py        ScaledDotProductAttention, MultiHeadAttention (MHA+GQA, KV cache,
                        RoPE hook), FeedForward, TransformerEncoderLayer/DecoderLayer (pre-LN)
    norm.py             LayerNorm, RMSNorm (fp32 statistics), make_norm factory
    transformer.py      TransformerConfig, TransformerEncoder/Decoder (checkpointed blocks,
                        causal mask, incremental decode), Seq2SeqTransformer (C1–C4), BLTModel (C5)
    blt.py              LocalByteEncoder / LocalByteDecoder (byte + rolling-hash n-gram
                        embeddings, banded attention, cross-attention pooling, byte head)
    entropy_patching.py ByteEntropyLM, next-byte entropies, patch_line / fixed_stride,
                        calibrate_theta_g (bisection), patch_stats, verify_incremental
  train.py              single entry point: configs, data pipeline, tokenizer training,
                        model build, AdamW + warmup/cosine, AMP (bf16/fp16+scaler),
                        activation checkpointing, scheduled sampling / prefix dropout
                        (off by default), per-epoch TF val + every-10 greedy val,
                        checkpoints, WandB (retry/fallback logic), HF upload, final test
  utils.py              all metrics from scratch (bit acc, seq acc, vectorized Levenshtein,
                        char-n-gram corpus BLEU, LCS ROUGE-L) + plot_training_curves
scripts/
  setup_cluster.sh      venv for the SLURM node (torch cu124, tokenizers, wandb)
  run_experiment.sh     positional-arg wrapper for one config
  submit_c14_vocab512.sh   C1–C4 official: 40 ep, vocab 512, 4 parallel SLURM jobs
  submit_c5.sh          C5 official (fixed 4-byte patches, 120-ep budget);
                        SKIP_FIXED=1 → only the explored entropy-dynamic variant
  run_all.sh / submit_all.sh   5-way SLURM job array
  eval_checkpoint.py    re-run full test eval for any saved checkpoint
  make_c5_official.py   assemble outputs/C5 from the (stopped) fixed run's log + model_best.pt
  make_results_table.py regenerate report tables from results.json files
  make_report_figures.py  report figures
  make_submission.sh    build <roll>_assignment1.zip (no .pt files — checkpoints on HF)
  upload_to_hf.py       batch HF upload
data/                   brown_plain.txt / brown_cipher.txt (5000 lines each)
outputs/                per-config artifacts; ablations/: vocab8000/, probe-c1-v512-ep15/,
                        c5-entropy/
report_code/Report.tex  the 6-page report (→ Report.pdf)
```

**Reproduce on one GPU:**

```bash
uv sync                      # or bash scripts/setup_cluster.sh on the cluster
# C1 (official): 40 epochs, vocab 512, batch 8 x accum 2
bash scripts/run_experiment.sh C1 40 8 5e-4 256 8 4 1024 512 4 64 512 2 4
# C5 (official): fixed 4-byte patches, 120-epoch budget, batch 16
bash scripts/run_experiment.sh C5 120 16 5e-4 256 8 4 1024 512 4 64 8000 1 4 fixed
# smoke test the whole pipeline (10 lines x 3 epochs, no WandB/HF)
python src/train.py --config C1 --quick
```

Artifacts: WandB project `irishbumfuzzle-team/anlp-assignment1`
(official runs C1 `d1kf0gih`, C2 `a00tbq4j`, C3 `ei6jhlnk`, C4 `4n9j9a0p`,
C5 `ufvkwo3r`); HuggingFace repos `IrishBumfuzzle/anlp-a1-C1..C5` (best + last
checkpoints, tokenizers, results).

---

## 11. Gotchas and known issues

1. **Long targets + fp16:** norms compute in fp32 and the GradScaler skips
   inf-norm steps — do not switch to fp32 training without cutting batch size
   (memory doubles), and do not disable the fp32 softmax in attention.
2. **Never enable length bucketing** (`--length-bucketing`): it stalls
   alignment learning (§5.3). Random batches are a correctness requirement for
   this task, not a preference.
3. **`--patching` default is `fixed`** (official C5). `entropy` requires the
   auxiliary-LM phase (auto-run at startup) and is the explored negative
   variant.
4. **C5 horizon:** if you rerun C5, keep the 120-epoch budget — the cosine
   schedule is defined over the budget, so a shorter run changes the LR
   trajectory, not just the number of epochs.
5. **WandB `patch_stats` histogram:** saved with exact float `hist_edges`
   (n+1 edges for n bins). Old checkpoints saved rounded `hist_centers`;
   `_log_patch_stats` has a fallback for them (the lossy format once crashed
   `wandb.Histogram` server-side).
6. **HF upload needs `HF_TOKEN`** (server runs set `SKIP_HF=1` and upload
   manually).
7. **Python 3.14:** iterating `bytes.encode("latin1")` yields ints — build
   byte lists explicitly (see `PhaseByteBPE.cipher_symbols`).
8. **β₂ documentation drift:** the report/IMPLEMENTATION state AdamW β₂ =
   0.98, but the executed code uses PyTorch's default 0.999 (no `betas`
   override in `train.py`). The official numbers were produced with 0.999.
9. **`eval.py` naming:** the file map in IMPLEMENTATION.md mentions
   `src/eval.py`; the evaluation logic actually lives in `train.py:evaluate()`
   plus `scripts/eval_checkpoint.py` (checkpoint re-scoring).
10. **Report scope:** the 6-page report covers the same ground as this file in
    compressed form; where they disagree, this document and the executed
    artifacts (`outputs/*/config.json`, `results.json`) are authoritative.
