from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    best_score: float | None = None,
    best_epoch: int | None = None,
    best_metrics: dict[str, float] | None = None,
    history: list[dict[str, Any]] | None = None,
    category: str | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics or {},
        "history": history or [],
        "category": category,
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
