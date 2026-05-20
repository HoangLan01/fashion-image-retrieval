from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BatchSoftmaxLoss(nn.Module):
    """Batch classification loss with optional symmetric InfoNCE direction."""

    def __init__(
        self,
        temperature: float = 1.0,
        symmetric: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.symmetric = symmetric
        self.label_smoothing = label_smoothing

    def forward(self, query_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        target_embeddings = F.normalize(target_embeddings, dim=-1)
        logits = query_embeddings @ target_embeddings.T / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        if self.symmetric:
            loss = 0.5 * (
                loss
                + F.cross_entropy(logits.T, targets, label_smoothing=self.label_smoothing)
            )
        return loss
