from __future__ import annotations

from collections.abc import Sequence

import torch


def recall_at_ks(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    ks: Sequence[int] = (10, 50),
    query_ids: Sequence[str] | None = None,
    exclude_query: bool = True,
) -> dict[str, float]:
    if query_embeddings.numel() == 0:
        return {f"R@{k}": 0.0 for k in ks}

    sims = query_embeddings @ gallery_embeddings.T
    gallery_index = {image_id: idx for idx, image_id in enumerate(gallery_ids)}

    if exclude_query and query_ids is not None:
        for row, query_id in enumerate(query_ids):
            gallery_pos = gallery_index.get(query_id)
            if gallery_pos is not None:
                sims[row, gallery_pos] = -torch.inf

    results: dict[str, float] = {}
    max_gallery = max(1, gallery_embeddings.shape[0])

    for k in ks:
        actual_k = min(k, max_gallery)
        topk = sims.topk(actual_k, dim=1).indices
        hits = 0
        for row, target_id in enumerate(target_ids):
            target_pos = gallery_index.get(target_id)
            if target_pos is not None and (topk[row] == target_pos).any().item():
                hits += 1
        results[f"R@{k}"] = hits / len(target_ids) * 100.0

    return results
