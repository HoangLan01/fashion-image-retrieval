from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import torch
from tqdm import tqdm

from aacl_fashion.utils.metrics import recall_at_ks


def _to_device(batch: dict, key: str, device: torch.device) -> torch.Tensor:
    return batch[key].to(device, non_blocking=True)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


@torch.no_grad()
def encode_gallery(
    model: torch.nn.Module,
    gallery_loader,
    device: torch.device,
    amp: bool = False,
) -> tuple[torch.Tensor, list[str]]:
    model.eval()
    embeddings = []
    image_ids: list[str] = []

    for batch in tqdm(gallery_loader, desc="Encode gallery", leave=False):
        images = _to_device(batch, "image", device)
        with _autocast(device, amp):
            batch_embeddings = model.encode_image(images)
        embeddings.append(batch_embeddings.detach().cpu())
        image_ids.extend(batch["image_id"])

    return torch.cat(embeddings, dim=0), image_ids


@torch.no_grad()
def encode_queries(
    model: torch.nn.Module,
    query_loader,
    device: torch.device,
    amp: bool = False,
) -> tuple[torch.Tensor, list[str], list[str]]:
    model.eval()
    embeddings = []
    query_ids: list[str] = []
    target_ids: list[str] = []

    for batch in tqdm(query_loader, desc="Encode queries", leave=False):
        query_images = _to_device(batch, "query_image", device)
        captions = list(batch["captions"])
        with _autocast(device, amp):
            batch_embeddings = model.encode_query(query_images, captions)
        embeddings.append(batch_embeddings.detach().cpu())
        query_ids.extend(batch["query_id"])
        target_ids.extend(batch["target_id"])

    return torch.cat(embeddings, dim=0), query_ids, target_ids


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    query_loader,
    gallery_loader,
    device: torch.device,
    recall_ks: Sequence[int] = (10, 50),
    exclude_query: bool = True,
    amp: bool = False,
) -> dict[str, float]:
    gallery_embeddings, gallery_ids = encode_gallery(model, gallery_loader, device, amp=amp)
    query_embeddings, query_ids, target_ids = encode_queries(model, query_loader, device, amp=amp)

    return recall_at_ks(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        target_ids=target_ids,
        gallery_ids=gallery_ids,
        ks=recall_ks,
        query_ids=query_ids,
        exclude_query=exclude_query,
    )
