from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticRetrievalDataset(Dataset):
    """Small deterministic dataset for CPU smoke tests."""

    def __init__(
        self,
        size: int = 16,
        image_size: int = 64,
        category: str = "dress",
        split: str = "train",
    ) -> None:
        self.size = size
        self.image_size = image_size
        self.category = category
        self.split = split

    def __len__(self) -> int:
        return self.size

    def _image(self, idx: int, offset: int = 0) -> torch.Tensor:
        generator = torch.Generator().manual_seed(idx + offset)
        return torch.rand(3, self.image_size, self.image_size, generator=generator)

    def __getitem__(self, idx: int) -> dict[str, object]:
        target_idx = (idx + 1) % self.size
        return {
            "query_image": self._image(idx),
            "target_image": self._image(target_idx, offset=10_000),
            "captions": f"make item {idx} match target {target_idx}",
            "query_id": f"{self.category}_{idx}",
            "target_id": f"{self.category}_{target_idx}",
            "category": self.category,
        }


class SyntheticGalleryDataset(Dataset):
    def __init__(self, size: int = 16, image_size: int = 64, category: str = "dress") -> None:
        self.size = size
        self.image_size = image_size
        self.category = category

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(idx + 10_000)
        return {
            "image": torch.rand(3, self.image_size, self.image_size, generator=generator),
            "image_id": f"{self.category}_{idx}",
            "category": self.category,
        }
