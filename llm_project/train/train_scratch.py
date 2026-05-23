"""
train/train_scratch.py
======================
Phase 1 training loop — tiny GPT from scratch.

Writes three artefacts in Dash dashboard:
  training_log.csv   — loss / perplexity / LR / grad_norm at every log step
  attn_weights.npy   — (n_layers, n_heads, T, T) for attention heatmaps
  embeddings.npy     — (vocab_size, d_model) for PCA / t-SNE scatter

Run:
    cd llm_project
    python -m train.train_scratch

To use your own corpus, set DATA_PATH to any UTF-8 .txt file.
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Make sure parent directory is on path when run directly ───────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare    import CharTokenizer, load_corpus, build_dataloaders
from model.transformer import GPT, GPTConfig
from model.generate    import generate, GenerationConfig
from utils.logger      import TrainingLogger
from utils.checkpoint  import CheckpointManager


# HYPERPARAMETERS
CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "data_path":    None,           # path to .txt file; None → built-in demo corpus
    "train_split":  0.9,

    # ── Model ─────────────────────────────────────────────────────────────────
    "context_len":  128,
    "n_layers":     4,
    "n_heads":      4,
    "d_model":      128,
    "d_ff":         512,
    "dropout":      0.1,

    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size":   32,
    "max_iters":    3000,
    "eval_interval":100,
    "eval_iters":   50,
    "log_interval": 10,

    # ── Optimiser ─────────────────────────────────────────────────────────────
    "lr":           3e-4,
    "weight_decay": 0.1,
    "beta1":        0.9,
    "beta2":        0.95,
    "grad_clip":    1.0,

    # ── LR schedule (cosine + linear warmup) ──────────────────────────────────
    "warmup_iters":    200,
    "lr_decay_iters":  3000,
    "min_lr":          3e-5,

    # ── Outputs ───────────────────────────────────────────────────────────────
    "log_path":       "training_log.csv",
    "checkpoint_dir": "checkpoints",
    "attn_path":      "attn_weights.npy",
    "embed_path":     "embeddings.npy",
    "tokenizer_path": "checkpoints/tokenizer.pkl",

    # ── Wandb (optional) ──────────────────────────────────────────────────────
    "use_wandb":    False,
    "wandb_project":"llm_scratch",

    # ── Device ────────────────────────────────────────────────────────────────
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# LR SCHEDULE
def get_lr(step: int, cfg: dict) -> float:
    """
    Cosine decay with linear warmup.

    Region 1 — warmup  (0 → warmup_iters)      : LR rises linearly 0 → peak_lr
    Region 2 — decay   (warmup → decay_iters)  : LR follows cosine curve to min_lr
    Region 3 — flat    (decay_iters → ∞)       : LR stays at min_lr
    """
    lr, min_lr       = cfg["lr"], cfg["min_lr"]
    warmup, decay    = cfg["warmup_iters"], cfg["lr_decay_iters"]

    if step < warmup:
        return lr * step / warmup
    if step > decay:
        return min_lr
    progress = (step - warmup) / (decay - warmup)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)



# EVALUATION
@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: str, n_iters: int) -> dict:
    """
    Estimate validation loss and perplexity by averaging over n_iters batches.

    @torch.no_grad() disables gradient tracking → 2× faster, less memory.
    """
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        losses.append(loss.item())

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    model.train()
    return {
        "val_loss":       avg_loss,
        "val_perplexity": math.exp(min(avg_loss, 20)),   # cap to avoid overflow
    }


# DASHBOARD EXPORTS
@torch.no_grad()
def export_attention_weights(model: GPT, sample_x: torch.Tensor, path: str) -> None:
    """
    Run one forward pass and save all attention weight matrices to .npy.

    Saved shape: (n_layers, n_heads, T, T)

    Dashboard usage:
        weights = np.load("attn_weights.npy")
        # weights[layer_idx, head_idx] → (T, T) heatmap
    """
    model.eval()
    _, _, all_attn = model(sample_x)
    stacked = np.stack([w[0].cpu().numpy() for w in all_attn], axis=0)
    np.save(path, stacked)
    model.train()


@torch.no_grad()
def export_embeddings(model: GPT, path: str) -> None:
    """
    Save the token embedding matrix for PCA / t-SNE visualisation.

    Saved shape: (vocab_size, d_model)

    Dashboard usage:
        from sklearn.decomposition import PCA
        emb    = np.load("embeddings.npy")
        coords = PCA(n_components=2).fit_transform(emb)
        # → scatter plot, one point per vocabulary token
    """
    emb = model.get_embeddings().numpy()
    np.save(path, emb)


# MAIN TRAINING LOOP
def train() -> None:
    cfg    = CONFIG
    device = cfg["device"]
    print(f"\n{'='*60}")
    print(f"  Phase 1 — Training tiny GPT from scratch")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    text      = load_corpus(cfg["data_path"])
    tokenizer = CharTokenizer().fit(text)
    token_ids = tokenizer.encode(text)

    # Save tokenizer so inference scripts can reload it
    os.makedirs(os.path.dirname(os.path.abspath(cfg["tokenizer_path"])), exist_ok=True)
    tokenizer.save(cfg["tokenizer_path"])

    train_loader, val_loader = build_dataloaders(
        token_ids   = token_ids,
        context_len = cfg["context_len"],
        batch_size  = cfg["batch_size"],
        train_split = cfg["train_split"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = GPTConfig(
        vocab_size  = tokenizer.vocab_size,
        context_len = cfg["context_len"],
        n_layers    = cfg["n_layers"],
        n_heads     = cfg["n_heads"],
        d_model     = cfg["d_model"],
        d_ff        = cfg["d_ff"],
        dropout     = cfg["dropout"],
    )
    model = GPT(model_cfg).to(device)
    print(model)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Apply weight decay only to weight matrices (not biases or LayerNorm params)
    decay_params   = [p for n, p in model.named_parameters()
                      if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.dim() < 2  and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": cfg["weight_decay"]},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    print(f"[Optim] AdamW | decay={sum(p.numel() for p in decay_params):,} | "
          f"no-decay={sum(p.numel() for p in nodecay_params):,}")

    # ── Logger + checkpoint manager ───────────────────────────────────────────
    logger  = TrainingLogger(
        log_path  = cfg["log_path"],
        use_wandb = cfg["use_wandb"],
        wandb_cfg = {"project": cfg["wandb_project"]},
    )
    ckpt_mgr = CheckpointManager(directory=cfg["checkpoint_dir"], keep_last=3)

    # ── Infinite data iterator ────────────────────────────────────────────────
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0           = time.time()
    running_loss = 0.0
    best_val_loss = float("inf")
    last_x       = None   # keep a sample batch for exports

    print(f"\n[Train] Starting — max_iters={cfg['max_iters']:,}\n")

    for step in range(1, cfg["max_iters"] + 1):

        # Set LR for this step
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        x, y = get_batch()
        x, y = x.to(device), y.to(device)
        last_x = x[:1]   # save for dashboard exports

        _, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        grad_norm = 0.0
        if cfg["grad_clip"] > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), cfg["grad_clip"]
            ).item()

        optimizer.step()
        running_loss += loss.item()

        # ── Periodic logging ─────────────────────────────────────────────────
        if step % cfg["log_interval"] == 0:
            avg_loss = running_loss / cfg["log_interval"]
            running_loss = 0.0
            elapsed  = time.time() - t0

            logger.log(
                step        = step,
                train_loss  = avg_loss,
                lr          = lr,
                grad_norm   = grad_norm,
                elapsed_sec = elapsed,
            )
            print(f"step {step:5d} | loss {avg_loss:.4f} | "
                  f"ppl {math.exp(min(avg_loss,20)):.2f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm:.3f} | {elapsed:.1f}s")

        # ── Evaluation + export ───────────────────────────────────────────────
        if step % cfg["eval_interval"] == 0:
            val_metrics = evaluate(model, val_loader, device, cfg["eval_iters"])
            elapsed     = time.time() - t0

            logger.log(
                step        = step,
                train_loss  = loss.item(),
                val_loss    = val_metrics["val_loss"],
                lr          = lr,
                grad_norm   = grad_norm,
                elapsed_sec = elapsed,
            )

            print(f"\n── Eval @ step {step} {'─'*38}")
            print(f"   val_loss={val_metrics['val_loss']:.4f}  "
                  f"val_ppl={val_metrics['val_perplexity']:.2f}")

            # Save checkpoint if best
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                ckpt_mgr.save(model, optimizer, step,
                               val_metrics["val_loss"], cfg)

            # Export attention weights and embeddings for dashboard
            if last_x is not None:
                export_attention_weights(model, last_x, cfg["attn_path"])
                export_embeddings(model, cfg["embed_path"])

            # Quick generation sample
            gen_cfg = GenerationConfig(max_new_tokens=80, temperature=0.8, top_k=10)
            prompt  = torch.tensor(
                [tokenizer.encode("To be")], dtype=torch.long, device=device
            )
            out = generate(model, prompt, gen_cfg, device=device)
            print(f"   Sample: {tokenizer.decode(out[0].tolist())!r}")
            print(f"{'─'*54}\n")
            model.train()

    # ── Final exports ──────────────────────────────────────────────────────
    print("\n[Train] Complete.")
    if last_x is not None:
        export_attention_weights(model, last_x.to(device), cfg["attn_path"])
    export_embeddings(model, cfg["embed_path"])

    print(f"\nDashboard inputs ready:")
    print(f"  {cfg['log_path']:<30} ← pd.read_csv  → loss / ppl / LR curves")
    print(f"  {cfg['attn_path']:<30} ← np.load      → attention heatmaps")
    print(f"  {cfg['embed_path']:<30} ← np.load      → PCA / t-SNE scatter")
    print(f"  {cfg['checkpoint_dir']+'/':<30} ← torch.load   → resume / inference")

    logger.close()


if __name__ == "__main__":
    train()

