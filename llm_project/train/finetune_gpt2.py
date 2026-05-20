"""
train/finetune_gpt2.py
======================
Phase 2 — Fine-tune GPT-2 on a custom corpus using LoRA (Low-Rank Adaptation).

What is LoRA?
  Instead of updating all 117M parameters of GPT-2, LoRA injects small
  trainable rank-r matrices alongside the frozen weight matrices.
  Only the LoRA parameters are updated during training — typically < 1% of
  the full model. This enables fine-tuning on a laptop GPU or even CPU.

  For a weight matrix W ∈ R^{d×k}, LoRA adds:
      W' = W + B·A   where A ∈ R^{r×k}, B ∈ R^{d×r}, r << min(d,k)

Dependencies (Phase 2 only):
    pip install transformers peft datasets accelerate

Run:
    cd llm_project
    python -m train.finetune_gpt2
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger     import TrainingLogger
from utils.checkpoint import CheckpointManager

# ── Lazy imports for HuggingFace stack ───────────────────────────────────────
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_cosine_schedule_with_warmup
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    from torch.utils.data import Dataset, DataLoader
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[Warning] HuggingFace / PEFT not installed.")
    print("  Run:  pip install transformers peft datasets accelerate")



# CONFIG
CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "data_path":    None,           # path to .txt file; None → demo corpus
    "train_split":  0.9,

    # ── Base model ────────────────────────────────────────────────────────────
    "model_name":   "gpt2",         # 'gpt2' | 'gpt2-medium' | 'gpt2-large'
    "context_len":  256,

    # ── LoRA ──────────────────────────────────────────────────────────────────
    "lora_r":       8,              # rank — higher = more capacity, more params
    "lora_alpha":   32,             # scaling factor (alpha/r = effective scale)
    "lora_dropout": 0.1,
    # Which attention projections to adapt (c_attn = Q+K+V, c_proj = output)
    "lora_target_modules": ["c_attn", "c_proj"],

    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size":   8,
    "max_iters":    1000,
    "eval_interval":100,
    "eval_iters":   20,
    "log_interval": 10,

    # ── Optimiser ─────────────────────────────────────────────────────────────
    "lr":           2e-4,
    "weight_decay": 0.01,
    "warmup_iters": 100,
    "grad_clip":    1.0,

    # ── Outputs ───────────────────────────────────────────────────────────────
    "log_path":       "training_log_ft.csv",
    "checkpoint_dir": "checkpoints_ft",
    "adapter_dir":    "lora_adapter",   # where the LoRA adapter weights are saved

    # ── Wandb (optional) ──────────────────────────────────────────────────────
    "use_wandb":    False,
    "wandb_project":"llm_finetune",

    # ── Device ────────────────────────────────────────────────────────────────
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}



# DATASET
class GPT2TextDataset(Dataset):
    """
    Tokenises a plain-text corpus with the GPT-2 tokenizer and returns
    sliding-window (input, label) pairs where label = input shifted right.

    Parameters
    ----------
    text        : raw string corpus
    tokenizer   : GPT2TokenizerFast instance
    context_len : max tokens per sample
    """

    def __init__(self, text: str, tokenizer, context_len: int):
        self.context_len = context_len
        ids = tokenizer.encode(text)
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len - 1)

    def __getitem__(self, idx: int):
        x = self.data[idx       : idx + self.context_len]
        y = self.data[idx + 1   : idx + self.context_len + 1]
        return x, y


# LR SCHEDULE
def get_lr(step: int, cfg: dict) -> float:
    """Linear warmup → cosine decay."""
    lr, min_lr = cfg["lr"], cfg["lr"] * 0.1
    warmup     = cfg["warmup_iters"]
    total      = cfg["max_iters"]
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# EVALUATION
@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, n_iters: int) -> dict:
    """Estimate validation loss and perplexity."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_iters:
            break
        x, y = x.to(device), y.to(device)
        out  = model(input_ids=x, labels=y)
        losses.append(out.loss.item())
    avg_loss = float(np.mean(losses)) if losses else float("inf")
    model.train()
    return {"val_loss": avg_loss, "val_perplexity": math.exp(min(avg_loss, 20))}


# GENERATION HELPER
@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, max_new: int, device: str) -> str:
    """Generate text from a prompt using the fine-tuned model."""
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens  = max_new,
        temperature     = 0.8,
        top_k           = 50,
        top_p           = 0.95,
        do_sample       = True,
        pad_token_id    = tokenizer.eos_token_id,
    )
    model.train()
    return tokenizer.decode(out[0], skip_special_tokens=True)



# MAIN FINE-TUNING LOOP
def finetune() -> None:
    if not HF_AVAILABLE:
        print("Install required packages first:")
        print("  pip install transformers peft datasets accelerate")
        return

    cfg    = CONFIG
    device = cfg["device"]

    print(f"\n{'='*60}")
    print(f"  Phase 2 — Fine-tuning {cfg['model_name']} with LoRA")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Corpus ───────────────────────────────────────────────────────────────
    from data.prepare import load_corpus
    text = load_corpus(cfg["data_path"])

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[Tokenizer] vocab_size={tokenizer.vocab_size}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    n      = int(len(text) * cfg["train_split"])
    train_ds = GPT2TextDataset(text[:n], tokenizer, cfg["context_len"])
    val_ds   = GPT2TextDataset(text[n:], tokenizer, cfg["context_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                               shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                               shuffle=False, drop_last=True)
    print(f"[Data] train={len(train_ds):,}  val={len(val_ds):,}")

    # ── Base model (frozen) ───────────────────────────────────────────────────
    base_model = GPT2LMHeadModel.from_pretrained(cfg["model_name"])
    base_model.resize_token_embeddings(tokenizer.vocab_size)
    print(f"[Model] {cfg['model_name']} loaded | "
          f"params={sum(p.numel() for p in base_model.parameters()):,}")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = cfg["lora_r"],
        lora_alpha      = cfg["lora_alpha"],
        lora_dropout    = cfg["lora_dropout"],
        target_modules  = cfg["lora_target_modules"],
        bias            = "none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device)

    # Show trainable vs frozen parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable={trainable:,} / {total:,}  "
          f"({100*trainable/total:.2f}%)")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Logger + checkpoint ───────────────────────────────────────────────────
    logger   = TrainingLogger(log_path=cfg["log_path"], use_wandb=cfg["use_wandb"])
    ckpt_mgr = CheckpointManager(directory=cfg["checkpoint_dir"], keep_last=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0            = time.time()
    running_loss  = 0.0
    best_val_loss = float("inf")

    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    print(f"\n[Train] Starting — max_iters={cfg['max_iters']:,}\n")

    for step in range(1, cfg["max_iters"] + 1):

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch()
        x, y = x.to(device), y.to(device)

        out  = model(input_ids=x, labels=y)
        loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = 0.0
        if cfg["grad_clip"] > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                cfg["grad_clip"],
            ).item()

        optimizer.step()
        running_loss += loss.item()

        # ── Log ──────────────────────────────────────────────────────────────
        if step % cfg["log_interval"] == 0:
            avg_loss     = running_loss / cfg["log_interval"]
            running_loss = 0.0
            elapsed      = time.time() - t0

            logger.log(step=step, train_loss=avg_loss, lr=lr,
                       grad_norm=grad_norm, elapsed_sec=elapsed)
            print(f"step {step:5d} | loss {avg_loss:.4f} | "
                  f"ppl {math.exp(min(avg_loss,20)):.2f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm:.3f} | {elapsed:.1f}s")

        # ── Eval ─────────────────────────────────────────────────────────────
        if step % cfg["eval_interval"] == 0:
            val_metrics = evaluate(model, val_loader, device, cfg["eval_iters"])
            elapsed     = time.time() - t0

            logger.log(step=step, train_loss=loss.item(),
                       val_loss=val_metrics["val_loss"],
                       lr=lr, grad_norm=grad_norm, elapsed_sec=elapsed)

            print(f"\n── Eval @ step {step} {'─'*38}")
            print(f"   val_loss={val_metrics['val_loss']:.4f}  "
                  f"val_ppl={val_metrics['val_perplexity']:.2f}")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                # Save LoRA adapter weights (tiny — only the rank-r matrices)
                os.makedirs(cfg["adapter_dir"], exist_ok=True)
                model.save_pretrained(cfg["adapter_dir"])
                print(f"   LoRA adapter saved → {cfg['adapter_dir']}/")

            # Text sample
            sample = generate_sample(model, tokenizer, "To be or not", 60, device)
            print(f"   Sample: {sample!r}")
            print(f"{'─'*54}\n")
            model.train()

    print("\n[Finetune] Complete.")
    model.save_pretrained(cfg["adapter_dir"])
    print(f"Final LoRA adapter → {cfg['adapter_dir']}/")
    print("\nTo reload for inference:")
    print("  from transformers import GPT2LMHeadModel")
    print("  from peft import PeftModel")
    print(f"  base = GPT2LMHeadModel.from_pretrained('{cfg['model_name']}')")
    print(f"  model = PeftModel.from_pretrained(base, '{cfg['adapter_dir']}')")
    logger.close()


if __name__ == "__main__":
    finetune()

