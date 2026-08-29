"""Main training and evaluation loop for C1-C5 ablation experiments."""

import argparse
import math
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import torch
from torch import nn

from dataset import ByteTokenizer, SubwordTokenizer, make_dataloaders
from models.attention import FeedForward, GroupedQueryAttention, MultiHeadAttention
from models.blt import LocalDecoder, LocalEncoder
from models.norm import LayerNorm, RMSNorm
from models.positional import RoPE, SinusoidalPositionalEncoding
from utils import causal_mask, compute_metrics, greedy_decode, set_seed


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer Encoder Layer."""

    def __init__(
        self,
        dim: int,
        heads: int,
        norm_type: str = "layer",
        gqa: bool = False,
        kv_heads: int = 4,
        dropout: float = 0.1,
        rope: Optional[RoPE] = None,
    ):
        super().__init__()
        Norm = RMSNorm if norm_type == "rms" else LayerNorm
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)

        if gqa:
            self.self_attn = GroupedQueryAttention(
                dim=dim, heads=heads, kv_heads=kv_heads, dropout=dropout, rope=rope
            )
        else:
            self.self_attn = MultiHeadAttention(
                dim=dim, heads=heads, dropout=dropout, rope=rope
            )

        self.ffn = FeedForward(dim=dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self Attention with residual connection
        x = x + self.self_attn(self.norm1(x), mask=mask)
        # Pre-LN FFN with residual connection
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    """Pre-LayerNorm Transformer Decoder Layer with Self-Attention and Cross-Attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        norm_type: str = "layer",
        gqa: bool = False,
        kv_heads: int = 4,
        dropout: float = 0.1,
        rope: Optional[RoPE] = None,
    ):
        super().__init__()
        Norm = RMSNorm if norm_type == "rms" else LayerNorm
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)
        self.norm3 = Norm(dim)

        if gqa:
            self.self_attn = GroupedQueryAttention(
                dim=dim, heads=heads, kv_heads=kv_heads, dropout=dropout, rope=rope
            )
            self.cross_attn = GroupedQueryAttention(
                dim=dim, heads=heads, kv_heads=kv_heads, dropout=dropout, rope=rope
            )
        else:
            self.self_attn = MultiHeadAttention(
                dim=dim, heads=heads, dropout=dropout, rope=rope
            )
            self.cross_attn = MultiHeadAttention(
                dim=dim, heads=heads, dropout=dropout, rope=rope
            )

        self.ffn = FeedForward(dim=dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Masked Self-Attention
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
        # Pre-LN Cross-Attention to Encoder Memory
        x = x + self.cross_attn(self.norm2(x), context=memory, mask=src_mask)
        # Pre-LN FFN
        x = x + self.ffn(self.norm3(x))
        return x


class TransformerModel(nn.Module):
    """Full Encoder-Decoder Transformer supporting C1-C4 Ablations."""

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        norm_type: str = "layer",
        gqa: bool = False,
        kv_heads: int = 4,
        use_rope: bool = False,
        dropout: float = 0.1,
        max_len: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.use_rope = use_rope

        self.src_emb = nn.Embedding(src_vocab, dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, dim)

        self.rope = RoPE(dim=dim // heads, max_len=max_len) if use_rope else None
        self.pos_enc = SinusoidalPositionalEncoding(dim=dim, max_len=max_len) if not use_rope else None

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=dim,
                heads=heads,
                norm_type=norm_type,
                gqa=gqa,
                kv_heads=kv_heads,
                dropout=dropout,
                rope=self.rope,
            )
            for _ in range(layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                dim=dim,
                heads=heads,
                norm_type=norm_type,
                gqa=gqa,
                kv_heads=kv_heads,
                dropout=dropout,
                rope=self.rope,
            )
            for _ in range(layers)
        ])

        Norm = RMSNorm if norm_type == "rms" else LayerNorm
        self.final_enc_norm = Norm(dim)
        self.final_dec_norm = Norm(dim)
        self.out_proj = nn.Linear(dim, tgt_vocab)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, source: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pos_enc is not None:
            pe = self.pos_enc.pe[:, :source.size(1), :].to(device=source.device, dtype=self.src_emb.weight.dtype)
            x = (self.src_emb(source) + pe) * math.sqrt(self.dim)
        else:
            x = self.src_emb(source) * math.sqrt(self.dim)
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        return self.final_enc_norm(x)

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pos_enc is not None:
            pe = self.pos_enc.pe[:, :target.size(1), :].to(device=target.device, dtype=self.tgt_emb.weight.dtype)
            x = (self.tgt_emb(target) + pe) * math.sqrt(self.dim)
        else:
            x = self.tgt_emb(target) * math.sqrt(self.dim)
        for layer in self.decoder_layers:
            x = layer(x, memory=memory, tgt_mask=tgt_mask, src_mask=src_mask)
        x = self.final_dec_norm(x)
        return self.out_proj(x)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(source, src_mask=src_mask)
        return self.decode(target, memory, tgt_mask=tgt_mask, src_mask=src_mask)


class BLTSeq2Seq(nn.Module):
    """Token-free Byte Latent Transformer (C5)."""

    def __init__(
        self,
        vocab_size: int = 260,
        dim: int = 256,
        byte_dim: int = 64,
        patch_size: int = 4,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        self.src_local_enc = LocalEncoder(
            byte_dim=byte_dim, model_dim=dim, patch_size=patch_size, vocab_size=vocab_size
        )
        self.tgt_local_enc = LocalEncoder(
            byte_dim=byte_dim, model_dim=dim, patch_size=patch_size, vocab_size=vocab_size
        )

        self.pos_enc = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=dim, heads=heads, norm_type="layer", gqa=False, dropout=dropout, rope=None
            )
            for _ in range(layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                dim=dim, heads=heads, norm_type="layer", gqa=False, dropout=dropout, rope=None
            )
            for _ in range(layers)
        ])

        self.final_enc_norm = LayerNorm(dim)
        self.final_dec_norm = LayerNorm(dim)
        self.local_decoder = LocalDecoder(
            model_dim=dim, byte_dim=byte_dim, patch_size=patch_size, vocab_size=vocab_size
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _downsample_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        b, t = mask.shape
        pad_len = (-t) % self.patch_size
        if pad_len > 0:
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=False)
        return mask.view(b, -1, self.patch_size).any(dim=-1)

    def encode(self, source: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # source: [B, T_bytes] -> patch_states: [B, T_patches, dim]
        patches = self.src_local_enc(source)
        pe = self.pos_enc.pe[:, :patches.size(1), :].to(device=patches.device, dtype=patches.dtype)
        x = (patches + pe) * math.sqrt(self.dim)
        patch_src_mask = self._downsample_mask(src_mask)
        for layer in self.encoder_layers:
            x = layer(x, mask=patch_src_mask)
        return self.final_enc_norm(x)

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_len = target.size(1)
        # target: [B, T_bytes] -> patch_states: [B, T_patches, dim]
        patches = self.tgt_local_enc(target)
        pe = self.pos_enc.pe[:, :patches.size(1), :].to(device=patches.device, dtype=patches.dtype)
        x = (patches + pe) * math.sqrt(self.dim)

        patch_src_mask = self._downsample_mask(src_mask)

        patch_causal = causal_mask(x.size(1), device=x.device).unsqueeze(0).unsqueeze(0)
        target_valid = target.ne(256)
        patch_tgt_valid = self._downsample_mask(target_valid)
        if patch_tgt_valid is not None:
            patch_tgt_mask = patch_causal & patch_tgt_valid.unsqueeze(1).unsqueeze(2)
        else:
            patch_tgt_mask = patch_causal

        for layer in self.decoder_layers:
            x = layer(x, memory=memory, tgt_mask=patch_tgt_mask, src_mask=patch_src_mask)

        x = self.final_dec_norm(x)
        logits = self.local_decoder(x, target_len=target_len)
        return logits

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(source, src_mask=src_mask)
        return self.decode(target, memory, tgt_mask=tgt_mask, src_mask=src_mask)


def build_config(name: str) -> Dict[str, Any]:
    """Return configuration dictionary for C1-C5."""
    configs = {
        "C1": {
            "name": "C1_Base",
            "pos": "sinusoidal",
            "attn": "mha",
            "norm": "layer",
            "tokenization": "subword",
            "use_rope": False,
            "gqa": False,
            "blt": False,
        },
        "C2": {
            "name": "C2_RoPE",
            "pos": "rope",
            "attn": "mha",
            "norm": "layer",
            "tokenization": "subword",
            "use_rope": True,
            "gqa": False,
            "blt": False,
        },
        "C3": {
            "name": "C3_GQA",
            "pos": "sinusoidal",
            "attn": "gqa",
            "norm": "layer",
            "tokenization": "subword",
            "use_rope": False,
            "gqa": True,
            "kv_heads": 4,
            "blt": False,
        },
        "C4": {
            "name": "C4_RMSNorm",
            "pos": "sinusoidal",
            "attn": "mha",
            "norm": "rms",
            "tokenization": "subword",
            "use_rope": False,
            "gqa": False,
            "blt": False,
        },
        "C5": {
            "name": "C5_BLT",
            "pos": "sinusoidal",
            "attn": "mha",
            "norm": "layer",
            "tokenization": "blt",
            "use_rope": False,
            "gqa": False,
            "blt": True,
        },
    }
    if name not in configs:
        raise ValueError(f"Unknown config '{name}'. Available: {list(configs.keys())}")
    return configs[name]


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum_steps: int = 1,
) -> Tuple[float, float]:
    """Train for one epoch with teacher forcing, AMP, and gradient accumulation."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    use_amp = scaler is not None and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    for batch_idx, batch in enumerate(loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        src_mask = batch["source_mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        # Teacher forcing inputs: target[:, :-1] predicts target[:, 1:]
        decoder_input = target[:, :-1]
        target_labels = target[:, 1:]
        decoder_input_mask = target_mask[:, :-1]

        # Combine causal mask [1, 1, T_dec, T_dec] and key padding mask [B, 1, 1, T_dec]
        c_mask = causal_mask(decoder_input.size(1), device=device).unsqueeze(0).unsqueeze(0)
        tgt_mask = c_mask & decoder_input_mask.unsqueeze(1).unsqueeze(2)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(source, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), target_labels.reshape(-1))
            loss = loss / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * grad_accum_steps

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    tokenizer: Any,
    device: torch.device,
    is_tokenized: bool = True,
    max_decode_samples: int = 50,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluate loss and greedy-decoding metrics on validation/test set."""
    model.eval()
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    total_loss = 0.0

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = use_amp and device.type == "cuda"

    all_preds: list[str] = []
    all_targets: list[str] = []

    for idx, batch in enumerate(loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        src_mask = batch["source_mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        decoder_input = target[:, :-1]
        target_labels = target[:, 1:]
        decoder_input_mask = target_mask[:, :-1]

        c_mask = causal_mask(decoder_input.size(1), device=device).unsqueeze(0).unsqueeze(0)
        tgt_mask = c_mask & decoder_input_mask.unsqueeze(1).unsqueeze(2)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits = model(source, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), target_labels.reshape(-1))
        total_loss += loss.item()

        # Run greedy decoding for metric calculation on a subset of samples
        if len(all_preds) < max_decode_samples:
            num_needed = min(source.size(0), max_decode_samples - len(all_preds))
            sub_source = source[:num_needed]
            sub_src_mask = src_mask[:num_needed] if src_mask is not None else None
            sub_targets = batch["target_text"][:num_needed]

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                decoded_tokens = greedy_decode(
                    model=raw_model,
                    source=sub_source,
                    tokenizer=tokenizer,
                    max_len=min(target.size(1) + 10, 512),
                    source_mask=sub_src_mask,
                    device=device,
                )

            for pred_ids, tgt_text in zip(decoded_tokens, sub_targets):
                pred_str = tokenizer.decode(pred_ids, remove_special_tokens=True)
                all_preds.append(pred_str)
                all_targets.append(tgt_text)

    metrics = compute_metrics(all_preds, all_targets, tokenized=is_tokenized)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def configure_optimizers(model: nn.Module, lr: float, weight_decay: float = 1e-2) -> torch.optim.Optimizer:
    """Separate model parameters into decay (2D+ weight matrices) and no_decay (biases, norms, embeddings)."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases, 1D normalization weights, scales, and embeddings are excluded from weight decay
        if (
            param.dim() < 2
            or "norm" in name.lower()
            or "emb" in name.lower()
            or "bias" in name.lower()
            or "scale" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train custom transformer architectures C1-C5.")
    parser.add_argument(
        "--config",
        choices=["C1", "C2", "C3", "C4", "C5"],
        default="C1",
        help="Ablation configuration",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing dataset files")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio for cosine scheduler")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Subword BPE vocabulary size")
    parser.add_argument("--dim", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--max-src-len", type=int, default=1024, help="Max source sequence length")
    parser.add_argument("--max-tgt-len", type=int, default=512, help="Max target sequence length")
    parser.add_argument("--patch-size", type=int, default=4, help="BLT patch size (e.g. 4 or 8)")
    parser.add_argument("--byte-dim", type=int, default=64, help="BLT byte dimension")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", default="anlp-assignment1", help="WandB project name")
    parser.add_argument("--wandb-entity", default="irishbumfuzzle-team", help="WandB team/entity name")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for checkpoints")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        print(f"Using device: {device} ({num_gpus} GPU{'s' if num_gpus > 1 else ''} detected)")
    else:
        print(f"Using device: {device}")

    cfg = build_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Tokenizer & DataLoaders
    cipher_file = f"{args.data_dir}/brown_cipher.txt"
    plain_file = f"{args.data_dir}/brown_plain.txt"

    cipher_path = Path(cipher_file)
    plain_path = Path(plain_file)
    if not cipher_path.exists() or not plain_path.exists():
        raise FileNotFoundError(
            f"Dataset files '{cipher_path}' or '{plain_path}' not found.\n"
            f"Please download the dataset files into the '{args.data_dir}' folder or specify --data-dir."
        )

    if cfg["blt"]:
        tokenizer = ByteTokenizer()
    else:
        # None enables make_dataloaders to train SubwordTokenizer strictly on the training split
        tokenizer = None

    train_loader, val_loader, test_loader, tokenizer = make_dataloaders(
        cipher_path=cipher_file,
        plain_path=plain_file,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        vocab_size=args.vocab_size,
    )

    # Build Model
    vocab_size = len(tokenizer)
    if cfg["blt"]:
        model = BLTSeq2Seq(
            vocab_size=vocab_size,
            dim=args.dim,
            byte_dim=args.byte_dim,
            patch_size=args.patch_size,
            heads=args.heads,
            layers=args.layers,
        ).to(device)
    else:
        model = TransformerModel(
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            dim=args.dim,
            heads=args.heads,
            layers=args.layers,
            norm_type=cfg["norm"],
            gqa=cfg["gqa"],
            kv_heads=cfg.get("kv_heads", 4),
            use_rope=cfg["use_rope"],
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration: {cfg['name']} | Trainable parameters: {total_params:,}")

    # Multi-GPU DataParallel Support
    if num_gpus > 1:
        print(f"Enabling DataParallel across {num_gpus} GPUs (effective batch size: {args.batch_size})")
        model = nn.DataParallel(model)

    # WandB Setup
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=cfg["name"],
                config={**cfg, **vars(args), "num_gpus": num_gpus, "trainable_params": total_params},
            )
        except Exception as e:
            print(f"WandB initialization failed or offline: {e}")

    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Automatic Mixed Precision (AMP) enabled.")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = configure_optimizers(model, lr=args.lr, weight_decay=1e-2)

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    best_val_loss = float("inf")
    checkpoint_path = output_dir / f"{cfg['name']}_checkpoint.pt"

    print("\nStarting Training...")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            for i in range(num_gpus):
                torch.cuda.reset_peak_memory_stats(i)

        train_loss, train_time = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
        )

        # Measure peak training GPU memory usage across all GPUs
        if num_gpus > 1:
            peak_train_gpu_mb = sum(torch.cuda.max_memory_allocated(i) for i in range(num_gpus)) / (1024 * 1024)
        elif num_gpus == 1:
            peak_train_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_train_gpu_mb = 0.0

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device,
            is_tokenized=not cfg["blt"],
            use_amp=use_amp,
        )

        gpu_info_str = f"Peak Train GPU: {peak_train_gpu_mb:.1f} MB"
        if num_gpus > 1:
            gpu_breakdown = ", ".join(f"GPU{i}: {torch.cuda.max_memory_allocated(i)/(1024*1024):.0f}MB" for i in range(num_gpus))
            gpu_info_str += f" ({gpu_breakdown})"

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Bit Acc: {val_metrics['bit_accuracy']:.2%} | "
            f"Seq Acc: {val_metrics['sequence_accuracy']:.2%} | "
            f"Time: {train_time:.1f}s | "
            f"{gpu_info_str}"
        )

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "bit_accuracy": val_metrics["bit_accuracy"],
                "sequence_accuracy": val_metrics["sequence_accuracy"],
                "levenshtein": val_metrics["levenshtein"],
                "bleu": val_metrics.get("bleu"),
                "rouge": val_metrics.get("rouge"),
                "epoch_time_sec": train_time,
                "peak_gpu_mb": peak_train_gpu_mb,
            })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            tokenizer_saved_path = None
            if hasattr(tokenizer, "save"):
                tokenizer_file = output_dir / f"{cfg['name']}_tokenizer.json"
                tokenizer.save(tokenizer_file)
                tokenizer_saved_path = str(tokenizer_file)

            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "config": cfg,
                    "val_metrics": val_metrics,
                    "tokenizer_type": "byte" if cfg["blt"] else "subword",
                    "tokenizer_path": tokenizer_saved_path,
                },
                checkpoint_path,
            )

    print("-" * 75)
    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}. Saved checkpoint: {checkpoint_path}")

    # Evaluate on Test Set using best checkpoint
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    if checkpoint_path.exists():
        print(f"\nLoading best checkpoint from '{checkpoint_path}' for final evaluation on Test Set...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state"])
    else:
        print("\nRunning final evaluation on Test Set...")

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
        is_tokenized=not cfg["blt"],
        use_amp=use_amp,
    )
    print(f"Test Results for {cfg['name']}:")
    for k, v in test_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if wandb_run:
        wandb.summary.update({f"test_{k}": v for k, v in test_metrics.items() if v is not None})
        wandb.finish()


if __name__ == "__main__":
    main()
