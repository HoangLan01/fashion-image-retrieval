from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from aacl_fashion.data.fashioniq_dataset import FashionIQDataset, FashionIQGalleryDataset
from aacl_fashion.data.synthetic_dataset import SyntheticGalleryDataset, SyntheticRetrievalDataset
from aacl_fashion.data.transforms import build_transforms


def build_train_loader(config: dict, category: str) -> DataLoader:
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset_name = dataset_config.get("name", "fashioniq")

    if dataset_name == "synthetic":
        dataset = SyntheticRetrievalDataset(
            size=int(dataset_config.get("size", 16)),
            image_size=int(dataset_config.get("image_size", 64)),
            category=category,
            split="train",
        )
    elif dataset_name == "fashioniq":
        dataset = FashionIQDataset(
            root=dataset_config["root"],
            category=category,
            split=dataset_config.get("train_split", "train"),
            transform=build_transforms(
                train=True,
                image_size=int(dataset_config.get("image_size", 224)),
                resize_size=int(dataset_config.get("resize_size", 256)),
            ),
            caption_mode=dataset_config.get("caption_mode", "concat"),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(dataset_config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def build_eval_loaders(config: dict, category: str) -> tuple[DataLoader, DataLoader]:
    dataset_config = config["dataset"]
    evaluation_config = config["evaluation"]
    dataset_name = dataset_config.get("name", "fashioniq")

    if dataset_name == "synthetic":
        size = int(dataset_config.get("size", 16))
        image_size = int(dataset_config.get("image_size", 64))
        query_dataset = SyntheticRetrievalDataset(size=size, image_size=image_size, category=category, split="val")
        gallery_dataset = SyntheticGalleryDataset(size=size, image_size=image_size, category=category)
    elif dataset_name == "fashioniq":
        transform = build_transforms(
            train=False,
            image_size=int(dataset_config.get("image_size", 224)),
            resize_size=int(dataset_config.get("resize_size", 256)),
        )
        split = dataset_config.get("val_split", "val")
        query_dataset = FashionIQDataset(
            root=dataset_config["root"],
            category=category,
            split=split,
            transform=transform,
            caption_mode=dataset_config.get("caption_mode", "concat"),
        )
        gallery_dataset = FashionIQGalleryDataset(
            root=dataset_config["root"],
            category=category,
            split=split,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    common = {
        "batch_size": int(evaluation_config.get("batch_size", 64)),
        "shuffle": False,
        "num_workers": int(dataset_config.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
    }
    return DataLoader(query_dataset, **common), DataLoader(gallery_dataset, **common)
