from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


class SwinImageEncoder(nn.Module):
    """Swin feature wrapper returning 98 projected tokens by default."""

    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 768,
        pretrained: bool = True,
        out_indices: Sequence[int] = (2, 3),
        pool_size: int = 7,
        freeze: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use SwinImageEncoder: pip install timm") from exc

        self.pool_size = pool_size
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
            cache_dir=cache_dir,
        )
        channels = self.backbone.feature_info.channels()
        self.projections = nn.ModuleList([nn.Linear(channel, embedding_dim) for channel in channels])

        if freeze:
            _set_trainable(self.backbone, False)

    def _feature_to_tokens(
        self,
        feature: torch.Tensor,
        channels: int,
        projection: nn.Linear,
    ) -> torch.Tensor:
        if feature.ndim == 4:
            if feature.shape[1] != channels and feature.shape[-1] == channels:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            pooled = F.adaptive_avg_pool2d(feature, (self.pool_size, self.pool_size))
            tokens = pooled.flatten(2).transpose(1, 2)
        elif feature.ndim == 3:
            tokens = feature if feature.shape[-1] == channels else feature.transpose(1, 2)
            target_tokens = self.pool_size * self.pool_size
            if tokens.shape[1] != target_tokens:
                tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), target_tokens).transpose(1, 2)
        else:
            raise ValueError(f"Unsupported Swin feature shape: {tuple(feature.shape)}")
        return projection(tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        tokens = []
        for feature, projection in zip(features, self.projections, strict=True):
            tokens.append(self._feature_to_tokens(feature, projection.in_features, projection))
        return torch.cat(tokens, dim=1)


class MockImageEncoder(nn.Module):
    """Small local encoder used by tests and synthetic smoke runs."""

    def __init__(self, embedding_dim: int = 768, pool_size: int = 7) -> None:
        super().__init__()
        self.pool_size = pool_size
        hidden_dim = min(128, embedding_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3_projection = nn.Linear(hidden_dim, embedding_dim)
        self.stage4_projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        pooled = F.adaptive_avg_pool2d(features, (self.pool_size, self.pool_size))
        tokens = pooled.flatten(2).transpose(1, 2)
        stage3 = self.stage3_projection(tokens)
        stage4 = self.stage4_projection(tokens + math.sqrt(0.5))
        return torch.cat([stage3, stage4], dim=1)


def build_image_encoder(config: dict) -> nn.Module:
    encoder_type = config.get("type", "swin")
    embedding_dim = int(config.get("embedding_dim", config.get("projection_dim", 768)))
    pool_size = int(config.get("pool_size", 7))

    if encoder_type == "mock":
        return MockImageEncoder(embedding_dim=embedding_dim, pool_size=pool_size)
    if encoder_type != "swin":
        raise ValueError(f"Unsupported image encoder type: {encoder_type}")

    return SwinImageEncoder(
        model_name=config.get("name", "swin_base_patch4_window7_224.ms_in22k_ft_in1k"),
        embedding_dim=embedding_dim,
        pretrained=bool(config.get("pretrained", True)),
        out_indices=config.get("out_indices", (2, 3)),
        pool_size=pool_size,
        freeze=bool(config.get("freeze", False)),
        cache_dir=config.get("cache_dir"),
    )
