from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F
from tqdm import tqdm

from aacl_fashion.config import load_config
from aacl_fashion.data.builders import build_eval_loaders
from aacl_fashion.evaluation import encode_gallery
from aacl_fashion.models import build_model
from aacl_fashion.occlusion import paired_bootstrap_ci, paired_sign_flip_pvalue
from aacl_fashion.utils.checkpoint import load_checkpoint
from aacl_fashion.utils.device import resolve_device
from aacl_fashion.utils.metrics import recall_at_ks
from aacl_fashion.utils.seed import set_seed


VARIANTS = ("full", "shuffled", "uniform")


def _sign_flip_pvalue(
    differences: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, str]:
    """Use the exact test for small probes and Monte Carlo for full validation sets."""
    if differences.size <= 20:
        return paired_sign_flip_pvalue(differences), "exact"
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk_size = 256
    while completed < samples:
        current = min(chunk_size, samples - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, differences.size))
        statistics = np.abs((signs * differences[None, :]).mean(axis=1))
        extreme += int(np.count_nonzero(statistics >= observed - 1e-12))
        completed += current
    return float((extreme + 1) / (samples + 1)), f"monte_carlo_{samples}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate inference-only interventions on AACL global context."
    )
    parser.add_argument("--config", default="configs/fashioniq_l40_shared.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-root", default="outputs/fashioniq_improved/l40_shared_seed42"
    )
    parser.add_argument("--category", default="shirt", choices=["dress", "shirt", "toptee"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output-root", default="outputs/context_intervention")
    parser.add_argument(
        "--table-output", default="outputs/report_assets/table_global_context_metrics.md"
    )
    parser.add_argument(
        "--figure-output", default="outputs/report_assets/fig_context_intervention.png"
    )
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


@torch.no_grad()
def _encode_queries(
    model: torch.nn.Module,
    query_loader,
    device: torch.device,
    variant: str,
    amp: bool,
) -> tuple[torch.Tensor, list[str], list[str]]:
    embeddings: list[torch.Tensor] = []
    query_ids: list[str] = []
    target_ids: list[str] = []
    model.eval()
    for batch in tqdm(query_loader, desc=f"Encode queries ({variant})", leave=False):
        images = batch["query_image"].to(device, non_blocking=True)
        captions = list(batch["captions"])
        autocast = torch.amp.autocast("cuda") if amp else nullcontext()
        with autocast:
            if variant == "full":
                output = model.encode_query(images, captions)
            else:
                output = model.encode_query_with_context_intervention(
                    images, captions, intervention=variant
                )
        embeddings.append(output.detach().cpu().float())
        query_ids.extend(str(value) for value in batch["query_id"])
        target_ids.extend(str(value) for value in batch["target_id"])
    return torch.cat(embeddings), query_ids, target_ids


def _ranking_state(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_ids: list[str],
    target_ids: list[str],
    gallery_ids: list[str],
    top_k: int = 5,
) -> tuple[np.ndarray, list[list[str]]]:
    scores = query_embeddings @ gallery_embeddings.T
    gallery_index = {str(image_id): index for index, image_id in enumerate(gallery_ids)}
    for row, query_id in enumerate(query_ids):
        position = gallery_index.get(str(query_id))
        if position is not None:
            scores[row, position] = -torch.inf
    ranks = []
    top_ids = []
    actual_k = min(top_k, scores.shape[1])
    indices = scores.topk(actual_k, dim=1).indices
    for row, target_id in enumerate(target_ids):
        target_position = gallery_index.get(str(target_id))
        if target_position is None:
            ranks.append(float("nan"))
        else:
            target_score = scores[row, target_position]
            ranks.append(float((scores[row] > target_score).sum().item() + 1))
        top_ids.append([str(gallery_ids[int(index)]) for index in indices[row]])
    return np.asarray(ranks, dtype=np.float64), top_ids


def _summarize_variant(
    variant: str,
    embeddings: torch.Tensor,
    full_embeddings: torch.Tensor,
    ranks: np.ndarray,
    full_ranks: np.ndarray,
    top_ids: list[list[str]],
    full_top_ids: list[list[str]],
    metrics: dict[str, float],
    full_metrics: dict[str, float],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    valid = np.isfinite(ranks) & np.isfinite(full_ranks)
    rank_change = ranks[valid] - full_ranks[valid]
    cosine = F.cosine_similarity(embeddings, full_embeddings).numpy()
    overlaps = np.asarray(
        [len(set(current) & set(reference)) for current, reference in zip(top_ids, full_top_ids, strict=True)],
        dtype=np.float64,
    )
    if variant == "full":
        ci = [0.0, 0.0]
        p_value = 1.0
        p_method = "not_applicable"
    else:
        ci = list(
            paired_bootstrap_ci(rank_change, samples=bootstrap_samples, seed=seed)
        )
        p_value, p_method = _sign_flip_pvalue(rank_change, bootstrap_samples, seed)
    reciprocal = np.where(valid, 1.0 / ranks, np.nan)
    return {
        "R@10": float(metrics.get("R@10", 0.0)),
        "R@50": float(metrics.get("R@50", 0.0)),
        "delta_R@10_vs_full": float(metrics.get("R@10", 0.0) - full_metrics.get("R@10", 0.0)),
        "delta_R@50_vs_full": float(metrics.get("R@50", 0.0) - full_metrics.get("R@50", 0.0)),
        "MRR": float(np.nanmean(reciprocal)),
        "median_target_rank": float(np.nanmedian(ranks)),
        "mean_target_rank_change_vs_full": float(rank_change.mean()),
        "target_rank_change_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        "target_rank_change_sign_flip_pvalue": float(p_value),
        "target_rank_change_sign_flip_method": p_method,
        "mean_embedding_cosine_to_full": float(cosine.mean()),
        "mean_top5_overlap_with_full": float(overlaps.mean()),
        "unchanged_top1_fraction": float(
            np.mean(
                [
                    current[0] == reference[0]
                    for current, reference in zip(top_ids, full_top_ids, strict=True)
                ]
            )
        ),
    }


def _write_table(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Global-context intervention (inference-only)",
        "",
        "Shuffled cyclically exchanges each block's context vector between samples; uniform replaces learned token weights with equal weights over valid tokens.",
        "",
        "| Variant | R@10 | R@50 | ΔR@10 | ΔR@50 | Median rank | Cosine→full | Top-5 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"full": "Full AACL", "shuffled": "Shuffled context", "uniform": "Uniform context"}
    for variant in VARIANTS:
        item = summaries[variant]
        lines.append(
            f"| {labels[variant]} | {item['R@10']:.4f} | {item['R@50']:.4f} | "
            f"{item['delta_R@10_vs_full']:+.4f} | {item['delta_R@50_vs_full']:+.4f} | "
            f"{item['median_target_rank']:.1f} | {item['mean_embedding_cosine_to_full']:.4f} | "
            f"{item['mean_top5_overlap_with_full']:.2f}/5 |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_figure(path: Path, summaries: dict[str, dict[str, Any]], category: str, n: int) -> None:
    canvas = Image.new("RGB", (1040, 560), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), "AACL global-context intervention", fill=(20, 20, 20), font=_font(25, True))
    draw.text(
        (18, 50),
        f"{category} | N={n} validation queries | same checkpoint/gallery | inference-only",
        fill=(65, 65, 65),
        font=_font(14),
    )
    colors = {"full": (55, 105, 175), "shuffled": (205, 70, 70), "uniform": (90, 155, 90)}
    labels = {"full": "Full", "shuffled": "Shuffled", "uniform": "Uniform"}
    panels = (("Recall@10", "R@10"), ("Recall@50", "R@50"))
    for panel_index, (title, key) in enumerate(panels):
        left = 30 + panel_index * 505
        top, right, bottom = 95, left + 475, 520
        draw.rectangle((left, top, right, bottom), outline=(190, 190, 190))
        draw.text((left + 12, top + 10), title, fill=(30, 30, 30), font=_font(18, True))
        values = [summaries[v][key] for v in VARIANTS]
        maximum = max(max(values) * 1.25, 1.0)
        chart_top, chart_bottom = top + 55, bottom - 55
        baseline_y = chart_bottom
        draw.line((left + 45, baseline_y, right - 20, baseline_y), fill=(90, 90, 90))
        for index, variant in enumerate(VARIANTS):
            x0 = left + 75 + index * 125
            height = int(values[index] / maximum * (chart_bottom - chart_top))
            y0 = chart_bottom - height
            draw.rectangle((x0, y0, x0 + 65, chart_bottom), fill=colors[variant])
            draw.text((x0 + 4, y0 - 22), f"{values[index]:.2f}", fill=(30, 30, 30), font=_font(13))
            draw.text((x0 - 3, chart_bottom + 12), labels[variant], fill=(30, 30, 30), font=_font(13))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(args.checkpoint_root) / args.category / "best.pt"
    )
    model = build_model(config["model"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config and checkpoint_config.get("model") != config.get("model"):
        raise ValueError("Checkpoint model config does not match the evaluation config.")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    query_loader, gallery_loader = build_eval_loaders(config, args.category)
    if len(query_loader.dataset) % int(query_loader.batch_size) == 1:
        raise ValueError(
            "The final query batch would contain one sample, which cannot use shuffled context. "
            "Change evaluation.batch_size in a dedicated config."
        )
    gallery_embeddings, gallery_ids = encode_gallery(model, gallery_loader, device, amp=amp)
    gallery_embeddings = gallery_embeddings.float()

    embeddings_by_variant: dict[str, torch.Tensor] = {}
    ranks_by_variant: dict[str, np.ndarray] = {}
    top_ids_by_variant: dict[str, list[list[str]]] = {}
    metrics_by_variant: dict[str, dict[str, float]] = {}
    query_ids: list[str] = []
    target_ids: list[str] = []
    recall_ks = tuple(int(value) for value in config["evaluation"].get("recall_ks", [10, 50]))
    for variant in VARIANTS:
        embeddings, current_query_ids, current_target_ids = _encode_queries(
            model, query_loader, device, variant, amp
        )
        if query_ids and (current_query_ids != query_ids or current_target_ids != target_ids):
            raise ValueError("Query order changed between context variants.")
        query_ids, target_ids = current_query_ids, current_target_ids
        embeddings_by_variant[variant] = embeddings
        metrics_by_variant[variant] = recall_at_ks(
            embeddings,
            gallery_embeddings,
            target_ids,
            gallery_ids,
            ks=recall_ks,
            query_ids=query_ids,
            exclude_query=True,
        )
        ranks_by_variant[variant], top_ids_by_variant[variant] = _ranking_state(
            embeddings, gallery_embeddings, query_ids, target_ids, gallery_ids
        )

    summaries = {
        variant: _summarize_variant(
            variant,
            embeddings_by_variant[variant],
            embeddings_by_variant["full"],
            ranks_by_variant[variant],
            ranks_by_variant["full"],
            top_ids_by_variant[variant],
            top_ids_by_variant["full"],
            metrics_by_variant[variant],
            metrics_by_variant["full"],
            args.bootstrap_samples,
            seed + index,
        )
        for index, variant in enumerate(VARIANTS)
    }

    output_dir = Path(args.output_root) / args.category / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (query_id, target_id) in enumerate(zip(query_ids, target_ids, strict=True)):
        for variant in VARIANTS:
            rows.append(
                {
                    "query_id": query_id,
                    "target_id": target_id,
                    "variant": variant,
                    "target_rank": ranks_by_variant[variant][index],
                    "target_rank_change_vs_full": ranks_by_variant[variant][index] - ranks_by_variant["full"][index],
                    "embedding_cosine_to_full": float(
                        F.cosine_similarity(
                            embeddings_by_variant[variant][index : index + 1],
                            embeddings_by_variant["full"][index : index + 1],
                        ).item()
                    ),
                    "top5_overlap_with_full": len(
                        set(top_ids_by_variant[variant][index]) & set(top_ids_by_variant["full"][index])
                    ),
                    "top1_id": top_ids_by_variant[variant][index][0],
                }
            )
    with (output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "category": args.category,
        "n_queries": len(query_ids),
        "evaluation_batch_size": int(query_loader.batch_size),
        "seed": seed,
        "inference_only": True,
        "definitions": {
            "shuffled": "At every block, cyclically shift learned context vectors by one sample within each fixed evaluation batch.",
            "uniform": "At every block, replace learned token attention with equal weights over valid image and text tokens.",
        },
        "variants": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_table(Path(args.table_output), summaries)
    _draw_figure(Path(args.figure_output), summaries, args.category, len(query_ids))
    print(f"Device: {device}; checkpoint: {checkpoint_path}")
    print(f"Wrote summary: {output_dir / 'summary.json'}")
    print(f"Wrote per-query results: {output_dir / 'per_query.csv'}")
    print(f"Wrote table: {args.table_output}")
    print(f"Wrote figure: {args.figure_output}")


if __name__ == "__main__":
    main()
