from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import nullcontext

import torch
from torch import nn
from tqdm import tqdm

from aacl_fashion.data.builders import build_eval_loaders, build_train_loader
from aacl_fashion.evaluation import evaluate_model
from aacl_fashion.losses import BatchSoftmaxLoss
from aacl_fashion.models import build_model
from aacl_fashion.utils.checkpoint import save_checkpoint
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
                "params": [p for p in model.image_encoder.parameters() if p.requires_grad],
                "lr": float(optimizer_config.get("image_lr", 1e-5)),
            },
            {
                "params": [p for p in model.text_encoder.parameters() if p.requires_grad],
                "lr": float(optimizer_config.get("text_lr", 1e-5)),
            },
            {
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
            [p for p in model.parameters() if p.requires_grad],
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


def train_one_category(config: dict[str, Any], category: str) -> dict[str, float]:
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    output_dir = Path(training_config.get("output_dir", "outputs")) / category
    recall_ks = tuple(int(k) for k in evaluation_config.get("recall_ks", [10, 50]))

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
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

        if scheduler is not None:
            scheduler.step()

        metrics: dict[str, float] = {"loss": running_loss / max(1, len(train_loader))}
        if epoch % eval_every == 0 or epoch == epochs:
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
            save_checkpoint(
                output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
            if score > best_score:
                best_score = score
                best_metrics = metrics
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    config=config,
                )

    return best_metrics
