from __future__ import annotations

import math
from typing import Any

import torch


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def minmax_normalize(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Min-max normalize the final dimension while keeping masked tokens at zero."""
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=values.device)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(values)
        minimum = values.masked_fill(~mask, torch.inf).amin(dim=-1, keepdim=True)
        maximum = values.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
    else:
        minimum = values.amin(dim=-1, keepdim=True)
        maximum = values.amax(dim=-1, keepdim=True)
    scale = (maximum - minimum).clamp_min(eps)
    normalized = (values - minimum) / scale
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        normalized = normalized.masked_fill(~mask, 0.0)
    return normalized


def compute_attention_flow(
    attention_weights: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Multiply alpha across blocks in log-space.

    Args:
        attention_weights: Tensor shaped ``[batch, block, head, token]``.
        attention_mask: Optional valid-token mask shaped ``[batch, token]``.

    Returns:
        Per-head relative flow shaped ``[batch, head, token]`` in ``[0, 1]``.
    """
    if attention_weights.ndim != 4:
        raise ValueError("attention_weights must have shape [batch, block, head, token].")
    log_flow = attention_weights.float().clamp_min(eps).log().sum(dim=1)
    expanded_mask = None
    if attention_mask is not None:
        if attention_mask.shape != (attention_weights.shape[0], attention_weights.shape[-1]):
            raise ValueError("attention_mask must have shape [batch, token].")
        expanded_mask = attention_mask[:, None, :].to(dtype=torch.bool, device=log_flow.device)
        log_flow = log_flow.masked_fill(~expanded_mask, -torch.inf)
    maximum = log_flow.amax(dim=-1, keepdim=True)
    relative_flow = torch.exp(log_flow - maximum)
    relative_flow = torch.nan_to_num(relative_flow, nan=0.0, posinf=0.0, neginf=0.0)
    return minmax_normalize(relative_flow, mask=expanded_mask)


def split_image_attention_maps(
    flow: torch.Tensor,
    image_token_count: int,
    pool_size: int = 7,
) -> dict[str, torch.Tensor]:
    """Split concatenated Swin Stage 3/4 tokens into aligned spatial maps."""
    expected_tokens = 2 * pool_size * pool_size
    if image_token_count != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} image tokens (two {pool_size}x{pool_size} stages), "
            f"found {image_token_count}."
        )
    if flow.ndim != 3 or flow.shape[-1] < image_token_count:
        raise ValueError("flow must have shape [batch, head, token] and contain all image tokens.")

    stage_tokens = pool_size * pool_size
    stage3_per_head = flow[..., :stage_tokens].reshape(
        flow.shape[0], flow.shape[1], pool_size, pool_size
    )
    stage4_per_head = flow[..., stage_tokens:image_token_count].reshape(
        flow.shape[0], flow.shape[1], pool_size, pool_size
    )
    stage3 = minmax_normalize(stage3_per_head.mean(dim=1).flatten(1)).reshape(
        flow.shape[0], pool_size, pool_size
    )
    stage4 = minmax_normalize(stage4_per_head.mean(dim=1).flatten(1)).reshape(
        flow.shape[0], pool_size, pool_size
    )
    average = minmax_normalize(((stage3 + stage4) * 0.5).flatten(1)).reshape(
        flow.shape[0], pool_size, pool_size
    )
    return {
        "stage3_per_head": stage3_per_head,
        "stage4_per_head": stage4_per_head,
        "stage3": stage3,
        "stage4": stage4,
        "average": average,
    }


def merge_wordpiece_scores(
    token_labels: list[str],
    scores: torch.Tensor | list[float],
) -> list[dict[str, Any]]:
    """Merge DistilBERT ``##`` pieces and average their flow scores."""
    score_values = scores.detach().cpu().tolist() if isinstance(scores, torch.Tensor) else list(scores)
    if len(token_labels) != len(score_values):
        raise ValueError("token_labels and scores must have the same length.")

    words: list[dict[str, Any]] = []
    for token, score in zip(token_labels, score_values, strict=True):
        if token in SPECIAL_TOKENS:
            continue
        if token.startswith("##") and words:
            words[-1]["word"] += token[2:]
            words[-1]["piece_scores"].append(float(score))
        else:
            words.append({"word": token, "piece_scores": [float(score)]})
    for item in words:
        item["score"] = sum(item["piece_scores"]) / len(item["piece_scores"])
        del item["piece_scores"]
    if words:
        values = torch.tensor([item["score"] for item in words], dtype=torch.float32)
        normalized = minmax_normalize(values.unsqueeze(0)).squeeze(0).tolist()
        for item, score in zip(words, normalized, strict=True):
            item["score"] = float(score)
    return words


def map_similarity(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Return Pearson correlation and Jensen-Shannon divergence for two maps."""
    a = first.detach().float().flatten()
    b = second.detach().float().flatten()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = a_centered.norm() * b_centered.norm()
    pearson = float((a_centered @ b_centered / denominator).item()) if denominator > 0 else 0.0

    p = (a.clamp_min(0) + 1e-12)
    q = (b.clamp_min(0) + 1e-12)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * torch.log(p / midpoint)) + 0.5 * torch.sum(q * torch.log(q / midpoint))
    return {"pearson": pearson, "jensen_shannon": float(js.item() / math.log(2.0))}
