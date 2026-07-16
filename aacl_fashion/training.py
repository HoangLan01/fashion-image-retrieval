from __future__ import annotations

import csv
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from contextlib import nullcontext

import torch
import yaml
from torch import nn
from tqdm import tqdm

from aacl_fashion.data.builders import build_eval_loaders, build_train_loader
from aacl_fashion.evaluation import evaluate_model
from aacl_fashion.losses import BatchSoftmaxLoss
from aacl_fashion.models import build_model
from aacl_fashion.utils.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from aacl_fashion.utils.device import resolve_device
from aacl_fashion.utils.seed import set_seed


def build_loss(config: dict[str, Any]) -> nn.Module:
    loss_config = config["loss"]
    return BatchSoftmaxLoss(
        temperature=float(loss_config.get("temperature", 1.0)),
        symmetric=bool(loss_config.get("symmetric", False)),
        label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    name = optimizer_config.get("name", "adamw").lower()

    if name == "adamw":
        parameter_groups = [
            {
                "name": "image_encoder",
                "params": [p for p in model.image_encoder.parameters() if p.requires_grad],
                "lr": float(optimizer_config.get("image_lr", 1e-5)),
            },
            {
                "name": "text_encoder",
                "params": [p for p in model.text_encoder.parameters() if p.requires_grad],
                "lr": float(optimizer_config.get("text_lr", 1e-5)),
            },
            {
                "name": "composition",
                "params": [p for p in model.composition.parameters() if p.requires_grad],
                "lr": float(optimizer_config.get("composition_lr", 3e-4)),
            },
        ]
        parameter_groups = [group for group in parameter_groups if group["params"]]
        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
        )

    if name == "sgd":
        return torch.optim.SGD(
            [{"name": "all", "params": [p for p in model.parameters() if p.requires_grad]}],
            lr=float(optimizer_config.get("lr", 0.035)),
            momentum=float(optimizer_config.get("momentum", 0.9)),
            weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    scheduler_config = config["scheduler"]
    name = scheduler_config.get("name", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"].get("epochs", 60)),
            eta_min=float(scheduler_config.get("eta_min", 1e-6)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 10)),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def _move_training_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    query_images = batch["query_image"].to(device, non_blocking=True)
    target_images = batch["target_image"].to(device, non_blocking=True)
    captions = list(batch["captions"])
    return query_images, target_images, captions


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def resolve_run_output_dir(config: dict[str, Any], category: str) -> Path:
    training_config = config["training"]
    output_dir = Path(training_config.get("output_dir", "outputs"))
    run_name = str(training_config.get("run_name", "")).strip()
    if run_name:
        output_dir = output_dir / run_name
    return output_dir / category


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in history)
    _write_text_atomic(output_dir / "metrics.jsonl", jsonl)

    fieldnames = sorted({key for row in history for key in row})
    if not fieldnames:
        return
    temporary_path = output_dir / ".metrics.csv.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        temporary_path.replace(output_dir / "metrics.csv")
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        values[f"lr_{name}"] = float(group["lr"])
    return values


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gib": round(props.total_memory / 1024**3, 3),
                "gpu_visible_count": torch.cuda.device_count(),
            }
        )
    return metadata


def _prepare_fresh_output(output_dir: Path, overwrite: bool) -> None:
    tracked_files = (
        "latest.pt",
        "best.pt",
        "metrics.csv",
        "metrics.jsonl",
        "run.json",
        "run_summary.json",
        "config.resolved.yaml",
    )
    existing = [output_dir / name for name in tracked_files if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Run artifacts already exist in {output_dir}: {names}. "
            "Use --resume auto to continue or --overwrite to start this run again."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def _load_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    category: str,
    config: dict[str, Any],
) -> tuple[int, float, int | None, dict[str, float], list[dict[str, Any]]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    checkpoint_category = checkpoint.get("category")
    if checkpoint_category is not None and checkpoint_category != category:
        raise ValueError(
            f"Checkpoint category is '{checkpoint_category}', but requested category is '{category}'."
        )

    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config and checkpoint_config.get("model") != config.get("model"):
        raise ValueError("Checkpoint model config does not match the current model config.")

    model.load_state_dict(checkpoint["model"])
    if "optimizer" not in checkpoint:
        raise ValueError("Resume checkpoint does not contain optimizer state.")
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if "scheduler" not in checkpoint:
            raise ValueError("Resume checkpoint does not contain scheduler state.")
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    last_epoch = int(checkpoint.get("epoch") or 0)
    history = list(checkpoint.get("history") or [])
    history = [row for row in history if int(row.get("epoch", 0)) <= last_epoch]
    best_metrics = dict(checkpoint.get("best_metrics") or {})
    best_score_value = checkpoint.get("best_score")
    if best_score_value is None:
        fallback = {key: value for key, value in checkpoint.get("metrics", {}).items() if key.startswith("R@")}
        best_metrics = best_metrics or dict(checkpoint.get("metrics") or {})
        best_score = sum(fallback.values()) / len(fallback) if fallback else -1.0
    else:
        best_score = float(best_score_value)
    best_epoch_value = checkpoint.get("best_epoch")
    best_epoch = int(best_epoch_value) if best_epoch_value is not None else None
    restore_rng_state(checkpoint.get("rng_state"))
    return last_epoch + 1, best_score, best_epoch, best_metrics, history


def train_one_category(
    config: dict[str, Any],
    category: str,
    resume: str | Path | None = None,
    overwrite: bool = False,
    device_name: str | None = None,
) -> dict[str, float]:
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(device_name)
    model = build_model(config["model"]).to(device)
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    train_loader = build_train_loader(config, category)
    query_loader, gallery_loader = build_eval_loaders(config, category)

    training_config = config["training"]
    evaluation_config = config["evaluation"]
    epochs = int(training_config.get("epochs", 60))
    grad_accumulation = int(training_config.get("grad_accumulation", 1))
    amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    clip_grad_norm = float(training_config.get("clip_grad_norm", 1.0))
    eval_every = int(training_config.get("eval_every", 5))
    checkpoint_every = int(training_config.get("checkpoint_every", 1))
    if eval_every <= 0 or checkpoint_every <= 0:
        raise ValueError("eval_every and checkpoint_every must be positive integers.")
    output_dir = resolve_run_output_dir(config, category)
    recall_ks = tuple(int(k) for k in evaluation_config.get("recall_ks", [10, 50]))

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score = -1.0
    best_epoch: int | None = None
    best_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    start_epoch = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path: Path | None = None
    if resume is not None:
        resume_path = output_dir / "latest.pt" if str(resume) == "auto" else Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_score, best_epoch, best_metrics, history = _load_training_state(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            category=category,
            config=config,
        )
        _write_history(output_dir, history)
    else:
        _prepare_fresh_output(output_dir, overwrite=overwrite)

    resolved_config = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    _write_text_atomic(output_dir / "config.resolved.yaml", resolved_config)
    run_metadata = {
        "category": category,
        "config_path": config.get("_meta", {}).get("config_path"),
        "output_dir": str(output_dir),
        "seed": int(config.get("seed", 42)),
        "started_at_utc": _utc_now(),
        "start_epoch": start_epoch,
        "target_epochs": epochs,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "amp": amp,
        "micro_batch_size": int(training_config.get("batch_size", 32)),
        "grad_accumulation": grad_accumulation,
        "optimizer_effective_batch_size": int(training_config.get("batch_size", 32)) * grad_accumulation,
        "contrastive_in_batch_size": int(training_config.get("batch_size", 32)),
        "train_records": len(train_loader.dataset),
        "train_batches_per_epoch": len(train_loader),
        "validation_queries": len(query_loader.dataset),
        "gallery_images": len(gallery_loader.dataset),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        **_device_metadata(device),
    }
    _write_json(output_dir / "run.json", run_metadata)

    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}, which is not below configured epochs={epochs}."
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        epoch_learning_rates = _learning_rates(optimizer)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"{category} epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress, start=1):
            query_images, target_images, captions = _move_training_batch(batch, device)
            with _autocast(device, amp):
                query_embeddings, target_embeddings = model(query_images, target_images, captions)
                loss = criterion(query_embeddings, target_embeddings) / grad_accumulation

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accumulation

            should_step = step % grad_accumulation == 0 or step == len(train_loader)
            if should_step:
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=running_loss / step)

        metrics: dict[str, float] = {"loss": running_loss / max(1, len(train_loader))}
        evaluated = epoch % eval_every == 0 or epoch == epochs
        is_best = False
        if evaluated:
            eval_metrics = evaluate_model(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                device=device,
                recall_ks=recall_ks,
                exclude_query=bool(evaluation_config.get("exclude_query", True)),
                amp=amp,
            )
            metrics.update(eval_metrics)
            score = sum(eval_metrics.values()) / max(1, len(eval_metrics))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics
                is_best = True

        if scheduler is not None:
            scheduler.step()

        history_row: dict[str, Any] = {
            "epoch": epoch,
            "timestamp_utc": _utc_now(),
            "duration_seconds": round(time.perf_counter() - epoch_started, 3),
            "train_loss": float(metrics["loss"]),
            "evaluated": evaluated,
            **epoch_learning_rates,
        }
        for key, value in metrics.items():
            if key != "loss":
                history_row[key] = float(value)
        if device.type == "cuda":
            history_row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
            history_row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 1024**3, 3)
        history.append(history_row)
        _write_history(output_dir, history)

        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "scaler": scaler,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "history": history,
            "category": category,
        }
        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(output_dir / "latest.pt", **checkpoint_kwargs)
        if is_best:
            save_checkpoint(output_dir / "best.pt", **checkpoint_kwargs)

        print(f"[{category}] epoch={epoch} metrics={history_row}")

    summary = {
        **run_metadata,
        "finished_at_utc": _utc_now(),
        "completed_epochs": epochs,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_checkpoint": str(output_dir / "best.pt"),
        "latest_checkpoint": str(output_dir / "latest.pt"),
        "history_csv": str(output_dir / "metrics.csv"),
        "history_jsonl": str(output_dir / "metrics.jsonl"),
    }
    _write_json(output_dir / "run_summary.json", summary)

    return best_metrics
