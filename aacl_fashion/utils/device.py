from __future__ import annotations

import torch


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve and activate the device used by a CLI entry point.

    PyTorch accepts an index-less ``cuda`` device for tensor moves, but
    ``torch.cuda.set_device`` requires an explicit logical device index.
    Logical index 0 is the selected GPU when CUDA_VISIBLE_DEVICES exposes a
    single device.
    """
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device.index)
    return device
