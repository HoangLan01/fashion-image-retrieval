from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "dataset": {
        "name": "fashioniq",
        "root": "data/fashioniq",
        "categories": ["dress", "shirt", "toptee"],
        "caption_mode": "concat",
        "image_size": 224,
        "resize_size": 256,
        "num_workers": 4,
    },
    "model": {
        "embedding_dim": 768,
        "image_encoder": {
            "type": "swin",
            "name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            "pretrained": True,
            "out_indices": [2, 3],
            "pool_size": 7,
            "freeze": False,
        },
        "text_encoder": {
            "type": "distilbert",
            "name": "distilbert-base-uncased",
            "max_length": 128,
            "freeze": False,
        },
        "composition": {
            "num_blocks": 3,
            "num_heads": 8,
            "ffn_multiplier": 4,
            "dropout": 0.1,
        },
    },
    "loss": {
        "temperature": 0.07,
        "symmetric": True,
        "label_smoothing": 0.1,
    },
    "optimizer": {
        "name": "adamw",
        "image_lr": 1e-5,
        "text_lr": 1e-5,
        "composition_lr": 3e-4,
        "weight_decay": 0.01,
        "momentum": 0.9,
    },
    "scheduler": {
        "name": "cosine",
        "step_size": 10,
        "gamma": 0.1,
        "eta_min": 1e-6,
    },
    "training": {
        "epochs": 60,
        "batch_size": 32,
        "grad_accumulation": 1,
        "amp": True,
        "clip_grad_norm": 1.0,
        "eval_every": 5,
        "output_dir": "outputs",
    },
    "evaluation": {
        "batch_size": 64,
        "recall_ks": [10, 50],
        "exclude_query": True,
    },
}


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    config = deepcopy(DEFAULT_CONFIG)
    deep_update(config, user_config)
    return config


def resolve_categories(config: dict[str, Any], category: str | None) -> list[str]:
    categories = list(config["dataset"].get("categories", []))
    if category is None or category == "all":
        return categories
    if category not in categories:
        raise ValueError(f"Unknown category '{category}'. Expected one of: {categories}")
    return [category]
