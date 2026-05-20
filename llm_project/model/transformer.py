"""
model/transformer.py
====================
GPT-style decoder-only transformer — pure PyTorch, no HuggingFace.

Components (bottom-up):
  GPTConfig           — all hyperparameters in one dataclass
  CausalSelfAttention — multi-head self-attention with causal mask
  FeedForward         — position-wise FFN (Linear → GELU → Linear)
  TransformerBlock    — one layer: pre-norm + attention + pre-norm + FFN
  GPT                 — full model: embeddings + N blocks + LM head

The model returns (logits, loss, attention_weights) so the dashboard
can visualise attention without an extra forward pass.

Usage:
    from model.transformer import GPT, GPTConfig
    cfg   = GPTConfig(vocab_size=65, context_len=128)
    model = GPT(cfg)
    logits, loss, attn = model(x, targets=y)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



# CONFIG
@dataclass
class GPTConfig:
    """
    All model hyperparameters in one place.

    Defaults produce a tiny model suitable for CPU training:
      ~100k parameters, trains on Shakespeare in a few minutes.

    Scaling guide:
      Small  (GPT-2 small)  : n_layers=12, n_heads=12, d_model=768
      Medium (GPT-2 medium) : n_layers=24, n_heads=16, d_model=1024
      For home training keep n_layers<=4, d_model<=256.
    """
    vocab_size:  int   = 65       # set automatically by the tokenizer
    context_len: int   = 128      # max sequence length the model can handle
    n_layers:    int   = 4        # number of stacked TransformerBlocks
    n_heads:     int   = 4        # attention heads (d_model must be divisible by n_heads)
    d_model:     int   = 128      # embedding / hidden dimension
    d_ff:        int   = 512      # feed-forward inner dimension (typically 4 × d_model)
    dropout:     float = 0.1      # dropout applied to attention weights and residuals
    bias:        bool  = False    # whether Linear layers use a bias term


# MULTI-HEAD CAUSAL SELF-ATTENTION
class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (autoregressive) mask.

    Why "causal"?
      Each position can only attend to itself and earlier positions.
      This prevents the model from "cheating" by looking at future tokens
      during training, which would make next-token prediction trivial.

    Implementation note:
      We compute Q, K, V with a single matrix multiplication (3 × d_model output)
      then split — this is faster than three separate projections on GPU.

    Shape walkthrough:
      Input  x          : (B, T, C)     B=batch, T=seq_len, C=d_model
      After qkv_proj    : (B, T, 3C)
      q, k, v           : (B, T, C)  each
      After reshape     : (B, H, T, D) H=n_heads, D=C//H
      attn_scores       : (B, H, T, T)
      attn_weights      : (B, H, T, T) — softmax, masked
      output            : (B, T, C)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )

        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.d_model = config.d_model

        # Single projection for Q, K, V
        self.qkv_proj  = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj   = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Upper-triangular causal mask (True = masked out → −inf before softmax)
        # Registered as a buffer: moves with .to(device) but is not a trainable parameter
        mask = torch.triu(torch.ones(config.context_len, config.context_len), diagonal=1)
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        # ── Q, K, V projection ───────────────────────────────────────────────
        qkv     = self.qkv_proj(x)                        # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=-1)         # each: (B, T, C)

        # ── Reshape to (B, H, T, D) ──────────────────────────────────────────
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # ── Scaled dot-product attention ─────────────────────────────────────
        scale       = 1.0 / math.sqrt(self.d_head)
        attn_scores = (q @ k.transpose(-2, -1)) * scale   # (B, H, T, T)

        # Mask future positions
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:T, :T], float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)     # (B, H, T, T)
        attn_weights = self.attn_drop(attn_weights)

        # ── Weighted sum of values ───────────────────────────────────────────
        out = attn_weights @ v                             # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_drop(self.out_proj(out))

        # Return detached weights for dashboard visualisation (no grad needed)
        return out, attn_weights.detach()



# FEED-FORWARD NETWORK
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently to every token position:
      x → Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model) → Dropout

    GELU (Gaussian Error Linear Unit) is smoother than ReLU and is used in
    GPT-2, BERT, and virtually all modern transformers.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# TRANSFORMER BLOCK  (one layer)
class TransformerBlock(nn.Module):
    """
    One GPT-style decoder block using pre-norm (LayerNorm before sub-layer).

    Forward pass:
      x = x + Attention(LayerNorm(x))    ← residual connection 1
      x = x + FFN(LayerNorm(x))          ← residual connection 2

    Pre-norm vs post-norm:
      The original "Attention is All You Need" paper used post-norm (norm after).
      GPT-2 and most modern LLMs use pre-norm because it trains more stably,
      especially at depth, as gradients flow more cleanly through the residuals.

    Residual connections:
      Allow gradients to bypass sub-layers entirely, preventing vanishing gradients
      in deep networks. Each block learns a correction δ, not a full transformation.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ffn  = FeedForward(config)

    def forward(self, x: torch.Tensor):
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


# FULL GPT MODEL
class GPT(nn.Module):
    """
    GPT-style causal language model.

    Architecture:
      tok_emb  : Embedding(vocab_size, d_model)          — token embeddings
      pos_emb  : Embedding(context_len, d_model)         — positional embeddings (learned)
      blocks   : N × TransformerBlock
      ln_f     : LayerNorm(d_model)                      — final layer norm
      lm_head  : Linear(d_model, vocab_size, bias=False) — language model head

    Weight tying (tok_emb.weight == lm_head.weight):
      The embedding matrix and the output projection share the same weights.
      This reduces parameters and consistently improves perplexity.
      Used in GPT-2, T5, LLaMA, and almost all modern LLMs.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config  = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f    = nn.LayerNorm(config.d_model)

        # LM head — no bias, weight-tied with token embedding
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Initialise weights (GPT-2 scheme)
        self.apply(self._init_weights)

        # Scale residual projections by 1/√(2·n_layers) for stability at depth
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        n_params = self.count_params()
        print(f"[GPT] initialised | layers={config.n_layers} | heads={config.n_heads} | "
              f"d_model={config.d_model} | params={n_params:,}")

    # ── Weight init ───────────────────────────────────────────────────────────
    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style: normal(0, 0.02) for weights, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        idx: torch.Tensor,                          # (B, T) token indices
        targets: Optional[torch.Tensor] = None,     # (B, T) shifted targets
    ):
        """
        Parameters
        ----------
        idx     : (B, T) long tensor of token IDs
        targets : (B, T) long tensor of next-token IDs (optional)
                  if provided, cross-entropy loss is computed and returned

        Returns
        -------
        logits           : (B, T, vocab_size)
        loss             : scalar cross-entropy loss, or None
        all_attn_weights : list of (B, H, T, T) tensors, one per layer
                           → used by dashboard for attention heatmaps
        """
        B, T = idx.shape
        assert T <= self.config.context_len, (
            f"Sequence length {T} exceeds model context_len {self.config.context_len}"
        )

        device = idx.device

        # ── Embeddings ───────────────────────────────────────────────────────
        tok  = self.tok_emb(idx)                                    # (B, T, C)
        pos  = self.pos_emb(torch.arange(T, device=device))        # (T, C) → broadcast
        x    = self.drop(tok + pos)                                 # (B, T, C)

        # ── Transformer blocks ───────────────────────────────────────────────
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x)
            all_attn_weights.append(attn_w)

        # ── Final norm + logits ──────────────────────────────────────────────
        x      = self.ln_f(x)                                       # (B, T, C)
        logits = self.lm_head(x)                                    # (B, T, vocab_size)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
            )

        return logits, loss, all_attn_weights

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_embeddings(self) -> torch.Tensor:
        """Return the token embedding matrix (vocab_size, d_model) — for PCA/t-SNE."""
        return self.tok_emb.weight.detach().cpu()

    def __repr__(self):
        return (f"GPT(vocab={self.config.vocab_size}, ctx={self.config.context_len}, "
                f"layers={self.config.n_layers}, heads={self.config.n_heads}, "
                f"d={self.config.d_model}, params={self.count_params():,})")

