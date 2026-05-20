"""
utils/logger.py
===============
Unified training logger.

Writes every metric row to a CSV file that your Dash dashboard
reads with pd.read_csv(). Optionally mirrors to Weights & Biases
if wandb is installed and WANDB_PROJECT is set in the config.

Usage:
    from utils.logger import TrainingLogger
    logger = TrainingLogger(log_path="training_log.csv", use_wandb=False)
    logger.log(step=10, train_loss=2.4, val_loss=2.6, lr=3e-4, grad_norm=0.9)
    logger.close()
"""

import csv
import os
import math
from typing import Optional, Any


# All columns written to the CSV (dashboard reads these by name)
CSV_FIELDS = [
    "step",
    "train_loss",
    "val_loss",
    "train_perplexity",
    "val_perplexity",
    "lr",
    "grad_norm",
    "elapsed_sec",
    "epoch",
]


class TrainingLogger:
    """
    Logs training metrics to CSV and optionally to Weights & Biases.

    Parameters
    ----------
    log_path   : path to the output CSV file
    use_wandb  : if True, tries to import wandb and log there too
    wandb_cfg  : dict passed to wandb.init() (project, name, config, etc.)
    """

    def __init__(
        self,
        log_path: str = "training_log.csv",
        use_wandb: bool = False,
        wandb_cfg: Optional[dict] = None,
    ):
        self.log_path  = log_path
        self.use_wandb = use_wandb
        self._wandb    = None

        # Create or overwrite the CSV and write the header row
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

        print(f"[Logger] CSV → {os.path.abspath(log_path)}")

        # Optional wandb setup
        if use_wandb:
            try:
                import wandb
                wandb.init(**(wandb_cfg or {}))
                self._wandb = wandb
                print("[Logger] wandb initialised")
            except ImportError:
                print("[Logger] wandb not installed — skipping. pip install wandb to enable.")
                self.use_wandb = False

    # Public API
    def log(
        self,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        elapsed_sec: float = 0.0,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Append one row to the CSV.

        Perplexity columns are computed automatically from loss values.
        None values are written as empty strings (Plotly handles NaN gracefully).
        """
        def ppl(loss):
            if loss is None:
                return ""
            try:
                return round(math.exp(loss), 4)
            except OverflowError:
                return float("inf")

        row = {
            "step":             step,
            "train_loss":       _fmt(train_loss),
            "val_loss":         _fmt(val_loss),
            "train_perplexity": ppl(train_loss),
            "val_perplexity":   ppl(val_loss),
            "lr":               _fmt(lr, decimals=8),
            "grad_norm":        _fmt(grad_norm),
            "elapsed_sec":      _fmt(elapsed_sec, decimals=1),
            "epoch":            epoch if epoch is not None else "",
        }

        # Write to CSV
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        # Mirror to wandb
        if self.use_wandb and self._wandb:
            wandb_row = {k: v for k, v in row.items() if v != ""}
            self._wandb.log(wandb_row, step=step)

    def close(self) -> None:
        """Finalise wandb run if active."""
        if self.use_wandb and self._wandb:
            self._wandb.finish()
            print("[Logger] wandb run finished")

    def __repr__(self):
        return f"TrainingLogger(log_path={self.log_path!r}, use_wandb={self.use_wandb})"



# Helpers
def _fmt(value, decimals: int = 5):
    """Round a float for CSV output; return empty string for None."""
    if value is None:
        return ""
    return round(float(value), decimals)

