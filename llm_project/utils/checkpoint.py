"""
utils/checkpoint.py
===================
Save and load model + optimiser checkpoints.

Every checkpoint stores:
  - model state dict
  - optimiser state dict
  - scheduler state dict (optional)
  - step / epoch counters
  - best validation loss seen so far
  - the full config dict so you always know what produced the weights

Usage:
    from utils.checkpoint import CheckpointManager
    mgr = CheckpointManager(directory="checkpoints", keep_last=3)
    mgr.save(model, optimizer, step=500, val_loss=2.31, config=cfg)
    step = mgr.load_latest(model, optimizer)
"""

import os
import glob
import torch
from typing import Optional


class CheckpointManager:
    """
    Manages a directory of .pt checkpoint files.

    Parameters
    ----------
    directory  : folder where checkpoints are written
    keep_last  : number of most-recent checkpoints to keep (older are deleted)
                 set to 0 to keep all
    """

    def __init__(self, directory: str = "checkpoints", keep_last: int = 3):
        self.directory  = directory
        self.keep_last  = keep_last
        self.best_loss  = float("inf")
        os.makedirs(directory, exist_ok=True)
        print(f"[Checkpoint] directory → {os.path.abspath(directory)}")


    # Save
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        val_loss: float,
        config: dict,
        scheduler=None,
        epoch: Optional[int] = None,
    ) -> str:
        """
        Save a checkpoint. Always saves the latest step.
        Also saves a separate best.pt if val_loss is the best seen.

        Returns the path of the file written.
        """
        payload = {
            "step":        step,
            "epoch":       epoch,
            "val_loss":    val_loss,
            "config":      config,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler else None,
        }

        # Numbered checkpoint
        path = os.path.join(self.directory, f"ckpt_step{step:06d}.pt")
        torch.save(payload, path)
        print(f"[Checkpoint] saved  → {path}  (val_loss={val_loss:.4f})")

        # Best checkpoint
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = os.path.join(self.directory, "best.pt")
            torch.save(payload, best_path)
            print(f"[Checkpoint] new best → {best_path}  (val_loss={val_loss:.4f})")

        # Prune old checkpoints
        if self.keep_last > 0:
            self._prune()

        return path


    # Load
    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        device: str = "cpu",
    ) -> int:
        """
        Load the most recent numbered checkpoint (not best.pt).
        Returns the step number so training can resume correctly.
        Returns 0 if no checkpoint exists.
        """
        path = self._latest_path()
        if path is None:
            print("[Checkpoint] no checkpoint found — starting from scratch")
            return 0
        return self._load(path, model, optimizer, scheduler, device)

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        device: str = "cpu",
    ) -> int:
        """Load best.pt — use this for final evaluation or inference."""
        path = os.path.join(self.directory, "best.pt")
        if not os.path.exists(path):
            print("[Checkpoint] best.pt not found")
            return 0
        return self._load(path, model, optimizer, scheduler, device)

    def load_path(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        device: str = "cpu",
    ) -> int:
        """Load a specific checkpoint file by path."""
        return self._load(path, model, optimizer, scheduler, device)


    # Internal helpers
    def _load(self, path, model, optimizer, scheduler, device) -> int:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if optimizer and ckpt.get("optim_state"):
            optimizer.load_state_dict(ckpt["optim_state"])
        if scheduler and ckpt.get("sched_state"):
            scheduler.load_state_dict(ckpt["sched_state"])
        step = ckpt.get("step", 0)
        print(f"[Checkpoint] loaded ← {path}  (step={step}, val_loss={ckpt.get('val_loss','?')})")
        return step

    def _latest_path(self) -> Optional[str]:
        """Return the path with the highest step number."""
        pattern = os.path.join(self.directory, "ckpt_step*.pt")
        files   = sorted(glob.glob(pattern))
        return files[-1] if files else None

    def _prune(self) -> None:
        """Delete numbered checkpoints beyond keep_last."""
        pattern = os.path.join(self.directory, "ckpt_step*.pt")
        files   = sorted(glob.glob(pattern))
        to_delete = files[: max(0, len(files) - self.keep_last)]
        for f in to_delete:
            os.remove(f)
            print(f"[Checkpoint] pruned  {f}")

    def list_checkpoints(self) -> list:
        """Return all checkpoint paths sorted by step."""
        pattern = os.path.join(self.directory, "ckpt_step*.pt")
        return sorted(glob.glob(pattern))

    def __repr__(self):
        return (f"CheckpointManager(directory={self.directory!r}, "
                f"keep_last={self.keep_last}, best_loss={self.best_loss:.4f})")

