from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


CAPTION_PATTERNS = (
    "cap.{category}.{split}.json",
    "{category}.{split}.json",
    "{split}.{category}.json",
)

SPLIT_PATTERNS = (
    "split.{category}.{split}.json",
    "{category}.{split}.json",
    "{split}.{category}.json",
)

IMAGE_SUFFIXES = ("", ".jpg", ".jpeg", ".png", ".webp")


def _find_existing(base_dir: Path, patterns: tuple[str, ...], category: str, split: str) -> Path:
    tried = []
    for pattern in patterns:
        candidate = base_dir / pattern.format(category=category, split=split)
        tried.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Expected one of these files:\n  " + "\n  ".join(tried))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_id(image_id: str) -> str:
    return Path(str(image_id)).stem


class FashionIQDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        transform=None,
        caption_mode: str = "concat",
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.caption_mode = caption_mode
        self.image_root = self.root / "images"

        self._validate_root()
        caption_path = _find_existing(self.root / "captions", CAPTION_PATTERNS, category, split)
        raw_records = _load_json(caption_path)
        self.records = [self._normalize_record(record) for record in raw_records]

    def _validate_root(self) -> None:
        expected = [self.root / "images", self.root / "captions", self.root / "image_splits"]
        missing = [path for path in expected if not path.exists()]
        if missing:
            missing_text = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "FashionIQ data is incomplete. Expected structure: "
                "data/fashioniq/{images,captions,image_splits}. Missing:\n  "
                + missing_text
            )

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        query_id = record.get("candidate") or record.get("query") or record.get("source")
        target_id = record.get("target")
        captions = record.get("captions") or record.get("caption") or record.get("text")

        if query_id is None or target_id is None or captions is None:
            raise ValueError(
                "FashionIQ caption records must contain candidate/query, target, and captions fields."
            )
        if isinstance(captions, str):
            captions = [captions]

        return {
            "query_id": _normalize_id(query_id),
            "target_id": _normalize_id(target_id),
            "captions": [str(caption) for caption in captions],
        }

    def _resolve_image(self, image_id: str) -> Path:
        raw_path = self.image_root / image_id
        for suffix in IMAGE_SUFFIXES:
            candidate = raw_path if suffix == "" else self.image_root / f"{image_id}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image '{image_id}' not found under {self.image_root}")

    def _format_caption(self, captions: list[str]) -> str:
        if self.caption_mode == "random":
            return random.choice(captions)
        if self.caption_mode == "first":
            return captions[0]
        if self.caption_mode == "concat":
            return " [SEP] ".join(captions)
        raise ValueError("caption_mode must be one of: concat, random, first")

    def _load_image(self, image_id: str):
        image = Image.open(self._resolve_image(image_id)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.records[idx]
        return {
            "query_image": self._load_image(record["query_id"]),
            "target_image": self._load_image(record["target_id"]),
            "captions": self._format_caption(record["captions"]),
            "query_id": record["query_id"],
            "target_id": record["target_id"],
            "category": self.category,
        }


class FashionIQGalleryDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.image_root = self.root / "images"
        self._validate_root()

        split_path = _find_existing(self.root / "image_splits", SPLIT_PATTERNS, category, split)
        raw_ids = _load_json(split_path)
        self.image_ids = [_normalize_id(image_id) for image_id in raw_ids]

    def _validate_root(self) -> None:
        expected = [self.root / "images", self.root / "image_splits"]
        missing = [path for path in expected if not path.exists()]
        if missing:
            missing_text = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError("FashionIQ gallery data is incomplete. Missing:\n  " + missing_text)

    def _resolve_image(self, image_id: str) -> Path:
        raw_path = self.image_root / image_id
        for suffix in IMAGE_SUFFIXES:
            candidate = raw_path if suffix == "" else self.image_root / f"{image_id}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image '{image_id}' not found under {self.image_root}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        image_id = self.image_ids[idx]
        image = Image.open(self._resolve_image(image_id)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "image_id": image_id,
            "category": self.category,
        }
