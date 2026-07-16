from __future__ import annotations

import itertools

import numpy as np
import torch
from torch.nn import functional as F


def select_patch_mask(
    attention_map: torch.Tensor,
    ratio: float,
    highest: bool,
) -> torch.Tensor:
    """Select the highest- or lowest-flow patches with a fixed area."""
    if attention_map.ndim != 2:
        raise ValueError("attention_map must be a 2D patch grid.")
    if not 0.0 < ratio < 0.5:
        raise ValueError("ratio must be between 0 and 0.5.")
    patch_count = attention_map.numel()
    selected_count = max(1, min(patch_count, int(round(patch_count * ratio))))
    flat = attention_map.detach().float().flatten()
    indices = torch.topk(flat, k=selected_count, largest=highest, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    return mask.reshape_as(attention_map)


def apply_patch_mask(
    image: torch.Tensor,
    patch_mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Occlude selected patches in a normalized CHW image.

    A fill value of zero corresponds to the dataset mean after ImageNet
    normalization.
    """
    if image.ndim != 3:
        raise ValueError("image must have shape [channel, height, width].")
    if patch_mask.ndim != 2:
        raise ValueError("patch_mask must be a 2D patch grid.")
    pixel_mask = F.interpolate(
        patch_mask[None, None].float(),
        size=image.shape[-2:],
        mode="nearest",
    ).squeeze(0).squeeze(0).bool()
    output = image.clone()
    output[:, pixel_mask] = fill_value
    return output


def paired_bootstrap_ci(
    differences: np.ndarray | list[float],
    samples: int = 5000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty 1D array.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def paired_sign_flip_pvalue(differences: np.ndarray | list[float]) -> float:
    """Exact two-sided paired randomization test for up to 20 pairs."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty 1D array.")
    if values.size > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 pairs.")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistic = abs(float((values * np.asarray(signs)).mean()))
        extreme += statistic >= observed - 1e-12
        total += 1
    return extreme / total
