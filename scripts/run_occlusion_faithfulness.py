from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from aacl_fashion.attention import compute_attention_flow, split_image_attention_maps
from aacl_fashion.config import load_config
from aacl_fashion.data.builders import build_eval_loaders
from aacl_fashion.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from aacl_fashion.evaluation import encode_gallery
from aacl_fashion.models import build_model
from aacl_fashion.occlusion import (
    apply_patch_mask,
    paired_bootstrap_ci,
    paired_sign_flip_pvalue,
    select_patch_mask,
)
from aacl_fashion.utils.checkpoint import load_checkpoint
from aacl_fashion.utils.device import resolve_device
from aacl_fashion.utils.seed import set_seed


RESULT_FIELDS = (
    "category",
    "probe_id",
    "query_id",
    "target_id",
    "prompt",
    "mask_ratio",
    "mask_type",
    "masked_patch_count",
    "reference_top1_id",
    "baseline_reference_similarity",
    "occluded_reference_similarity",
    "reference_similarity_drop",
    "occluded_reference_rank",
    "reference_rank_increase",
    "baseline_target_similarity",
    "occluded_target_similarity",
    "target_similarity_drop",
    "baseline_target_rank",
    "occluded_target_rank",
    "target_rank_increase",
    "query_embedding_cosine",
    "query_embedding_cosine_drop",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare equal-area high- and low-attention occlusion on fixed FashionIQ probes."
    )
    parser.add_argument("--config", default="configs/fashioniq_l40_shared.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-root", default="outputs/fashioniq_improved/l40_shared_seed42"
    )
    parser.add_argument("--category", default="shirt", choices=["dress", "shirt", "toptee"])
    parser.add_argument("--prompts-manifest", default="outputs/hallucination/prompts.json")
    parser.add_argument("--num-probes", type=int, default=10)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/occlusion")
    parser.add_argument(
        "--report-output", default="outputs/report_assets/fig_occlusion_comparison.png"
    )
    parser.add_argument(
        "--table-output", default="outputs/report_assets/table_occlusion_faithfulness.md"
    )
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _model_input_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().float().clone()
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    image = (image * std + mean).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _rank(scores: torch.Tensor, position: int) -> int:
    return int((scores > scores[position]).sum().item()) + 1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_metric(
    rows: list[dict[str, Any]],
    ratio: float,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = [row for row in rows if abs(float(row["mask_ratio"]) - ratio) < 1e-9]
    high = {row["probe_id"]: float(row[metric]) for row in selected if row["mask_type"] == "high"}
    low = {row["probe_id"]: float(row[metric]) for row in selected if row["mask_type"] == "low"}
    if set(high) != set(low):
        raise ValueError(f"High/low probe mismatch at ratio={ratio} for {metric}.")
    probe_ids = sorted(high)
    high_values = np.asarray([high[probe_id] for probe_id in probe_ids])
    low_values = np.asarray([low[probe_id] for probe_id in probe_ids])
    differences = high_values - low_values
    lower, upper = paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "n": len(probe_ids),
        "mean_high": float(high_values.mean()),
        "mean_low": float(low_values.mean()),
        "mean_high_minus_low": float(differences.mean()),
        "bootstrap_95_ci": [lower, upper],
        "sign_flip_pvalue": paired_sign_flip_pvalue(differences),
        "per_probe_high_minus_low": {
            probe_id: float(value) for probe_id, value in zip(probe_ids, differences, strict=True)
        },
    }


def _draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ratios: list[float],
    high_values: list[float],
    low_values: list[float],
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(190, 190, 190), width=1)
    draw.text((left + 10, top + 8), title, fill=(25, 25, 25), font=_font(17, True))
    chart_top = top + 42
    chart_bottom = bottom - 38
    chart_left = left + 48
    chart_right = right - 16
    values = high_values + low_values
    minimum = min(min(values), 0.0)
    maximum = max(max(values), 0.0)
    span = max(maximum - minimum, 1e-6)
    padding = span * 0.15
    plot_min = minimum - padding if minimum < 0.0 else 0.0
    plot_max = maximum + padding
    plot_span = max(plot_max - plot_min, 1e-6)
    zero_y = int(chart_top + (plot_max / plot_span) * (chart_bottom - chart_top))
    draw.line((chart_left, zero_y, chart_right, zero_y), fill=(90, 90, 90), width=1)
    group_width = (chart_right - chart_left) / len(ratios)
    bar_width = int(group_width * 0.25)
    for index, (ratio, high, low) in enumerate(
        zip(ratios, high_values, low_values, strict=True)
    ):
        center = int(chart_left + (index + 0.5) * group_width)
        for offset, value, color in (
            (-bar_width, high, (205, 70, 70)),
            (0, low, (65, 115, 190)),
        ):
            x0 = center + offset
            value_y = int(
                chart_top + ((plot_max - value) / plot_span) * (chart_bottom - chart_top)
            )
            y0, y1 = sorted((zero_y, value_y))
            if y0 == y1:
                y1 += 1
            draw.rectangle((x0, y0, x0 + bar_width - 2, y1), fill=color)
            label_y = max(chart_top, value_y - 17) if value >= 0 else min(chart_bottom - 13, value_y + 2)
            draw.text(
                (x0 - 2, label_y),
                f"{value:.3f}",
                fill=(35, 35, 35),
                font=_font(11),
            )
        draw.text(
            (center - 17, bottom - 29),
            f"{int(ratio * 100)}%",
            fill=(35, 35, 35),
            font=_font(13),
        )
    draw.rectangle((right - 155, top + 10, right - 142, top + 23), fill=(205, 70, 70))
    draw.text((right - 137, top + 8), "high", fill=(40, 40, 40), font=_font(12))
    draw.rectangle((right - 85, top + 10, right - 72, top + 23), fill=(65, 115, 190))
    draw.text((right - 67, top + 8), "low", fill=(40, 40, 40), font=_font(12))


def _build_report_figure(
    summary: dict[str, Any],
    source_image: Image.Image,
    high_image: Image.Image,
    low_image: Image.Image,
    preview_ratio: float,
    output_path: Path,
) -> None:
    ratios = [float(value) for value in summary["ratios"]]
    summary_by_ratio = summary["by_ratio"]
    canvas = Image.new("RGB", (1120, 760), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 12),
        "AACL occlusion faithfulness: high vs low attention",
        fill=(15, 15, 15),
        font=_font(24, True),
    )
    draw.text(
        (16, 48),
        f"{summary['category']} | N={summary['n_probes']} fixed probes | "
        "equal-area ImageNet-mean masks | paired test",
        fill=(60, 60, 60),
        font=_font(14),
    )
    image_y = 94
    for index, (label, image) in enumerate(
        (
            ("Source", source_image),
            (f"High {int(preview_ratio * 100)}%", high_image),
            (f"Low {int(preview_ratio * 100)}%", low_image),
        )
    ):
        x = 80 + index * 330
        draw.text((x, image_y), label, fill=(25, 25, 25), font=_font(16, True))
        canvas.paste(image, (x, image_y + 28))

    high_sim = [
        summary_by_ratio[f"{int(r * 100)}%"]["reference_similarity_drop"]["mean_high"]
        for r in ratios
    ]
    low_sim = [
        summary_by_ratio[f"{int(r * 100)}%"]["reference_similarity_drop"]["mean_low"]
        for r in ratios
    ]
    high_rank = [
        summary_by_ratio[f"{int(r * 100)}%"]["reference_rank_increase"]["mean_high"]
        for r in ratios
    ]
    low_rank = [
        summary_by_ratio[f"{int(r * 100)}%"]["reference_rank_increase"]["mean_low"]
        for r in ratios
    ]
    _draw_bar_chart(
        draw,
        (30, 390, 550, 735),
        "Top-1 reference similarity drop",
        ratios,
        high_sim,
        low_sim,
    )
    _draw_bar_chart(
        draw,
        (570, 390, 1090, 735),
        "Top-1 reference rank increase",
        ratios,
        high_rank,
        low_rank,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not 1 <= args.num_probes <= 20:
        raise ValueError("--num-probes must be between 1 and 20 for the exact sign-flip test.")
    if any(not 0.0 < ratio < 0.5 for ratio in args.ratios):
        raise ValueError("Every mask ratio must be between 0 and 0.5.")
    ratios = sorted(set(float(ratio) for ratio in args.ratios))

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
    model.load_state_dict(checkpoint["model"])
    model.eval()

    query_loader, gallery_loader = build_eval_loaders(config, args.category)
    query_dataset = query_loader.dataset
    if config["dataset"].get("name") == "synthetic":
        probes = [
            {
                "probe_id": f"{args.category}_q{index + 1:02d}",
                "category": args.category,
                "query_index": index,
                "query_id": query_dataset[index]["query_id"],
                "original_caption": query_dataset[index]["captions"],
            }
            for index in range(args.num_probes)
        ]
    else:
        manifest = json.loads(Path(args.prompts_manifest).read_text(encoding="utf-8"))
        probes = sorted(
            [probe for probe in manifest["probes"] if probe["category"] == args.category],
            key=lambda probe: probe["probe_id"],
        )[: args.num_probes]
        if len(probes) != args.num_probes:
            raise ValueError(f"Found only {len(probes)} probes for {args.category}.")

    samples = [query_dataset[int(probe["query_index"])] for probe in probes]
    for probe, sample in zip(probes, samples, strict=True):
        if str(sample["query_id"]) != str(probe["query_id"]):
            raise ValueError(f"Manifest mismatch for {probe['probe_id']}.")
    images = torch.stack([sample["query_image"] for sample in samples]).to(device)
    captions = [str(probe["original_caption"]) for probe in probes]

    autocast = torch.amp.autocast("cuda") if amp else nullcontext()
    with autocast:
        baseline_output = model.encode_query_with_attention(images, captions)
    weights = baseline_output.attention_weights.detach().cpu().float()
    text_mask = baseline_output.text_features.attention_mask.detach().cpu()
    combined_mask = torch.cat(
        [
            torch.ones(len(probes), baseline_output.image_token_count, dtype=text_mask.dtype),
            text_mask,
        ],
        dim=1,
    )
    flow = compute_attention_flow(weights, attention_mask=combined_mask)
    pool_size = int(config["model"]["image_encoder"].get("pool_size", 7))
    attention_maps = split_image_attention_maps(
        flow,
        image_token_count=baseline_output.image_token_count,
        pool_size=pool_size,
    )["average"]

    gallery_embeddings, gallery_ids = encode_gallery(model, gallery_loader, device, amp=amp)
    gallery_embeddings = gallery_embeddings.float()
    gallery_index = {str(image_id): index for index, image_id in enumerate(gallery_ids)}
    baseline_embeddings = baseline_output.embedding.detach().cpu().float()
    baseline_scores = baseline_embeddings @ gallery_embeddings.T

    baselines: list[dict[str, Any]] = []
    for index, (probe, sample) in enumerate(zip(probes, samples, strict=True)):
        source_position = gallery_index.get(str(sample["query_id"]))
        if source_position is not None:
            baseline_scores[index, source_position] = -torch.inf
        top1_position = int(baseline_scores[index].argmax().item())
        target_position = gallery_index.get(str(sample["target_id"]))
        if target_position is None:
            raise ValueError(f"Target {sample['target_id']} is absent from the gallery.")
        baselines.append(
            {
                "reference_position": top1_position,
                "reference_id": str(gallery_ids[top1_position]),
                "reference_similarity": float(baseline_scores[index, top1_position].item()),
                "target_position": target_position,
                "target_similarity": float(baseline_scores[index, target_position].item()),
                "target_rank": _rank(baseline_scores[index], target_position),
            }
        )

    conditions: list[dict[str, Any]] = []
    masks_for_npz = []
    for probe_index, (probe, sample) in enumerate(zip(probes, samples, strict=True)):
        for ratio in ratios:
            pair_masks = []
            for mask_type, highest in (("high", True), ("low", False)):
                patch_mask = select_patch_mask(attention_maps[probe_index], ratio, highest=highest)
                conditions.append(
                    {
                        "probe_index": probe_index,
                        "probe": probe,
                        "sample": sample,
                        "ratio": ratio,
                        "mask_type": mask_type,
                        "patch_mask": patch_mask,
                        "image": apply_patch_mask(sample["query_image"], patch_mask, fill_value=0.0),
                        "caption": captions[probe_index],
                    }
                )
                pair_masks.append(patch_mask.numpy())
            masks_for_npz.append(np.stack(pair_masks))

    occluded_embeddings: list[torch.Tensor] = []
    for start in range(0, len(conditions), args.batch_size):
        batch = conditions[start : start + args.batch_size]
        batch_images = torch.stack([item["image"] for item in batch]).to(device)
        batch_captions = [item["caption"] for item in batch]
        with torch.amp.autocast("cuda") if amp else nullcontext():
            embeddings = model.encode_query(batch_images, batch_captions)
        occluded_embeddings.append(embeddings.detach().cpu().float())
    all_occluded_embeddings = torch.cat(occluded_embeddings, dim=0)
    occluded_scores = all_occluded_embeddings @ gallery_embeddings.T

    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        probe_index = int(condition["probe_index"])
        sample = condition["sample"]
        baseline = baselines[probe_index]
        source_position = gallery_index.get(str(sample["query_id"]))
        if source_position is not None:
            occluded_scores[condition_index, source_position] = -torch.inf
        reference_position = int(baseline["reference_position"])
        target_position = int(baseline["target_position"])
        reference_similarity = float(occluded_scores[condition_index, reference_position].item())
        target_similarity = float(occluded_scores[condition_index, target_position].item())
        cosine = float(
            F.cosine_similarity(
                baseline_embeddings[probe_index : probe_index + 1],
                all_occluded_embeddings[condition_index : condition_index + 1],
            ).item()
        )
        rows.append(
            {
                "category": args.category,
                "probe_id": condition["probe"]["probe_id"],
                "query_id": sample["query_id"],
                "target_id": sample["target_id"],
                "prompt": condition["caption"],
                "mask_ratio": condition["ratio"],
                "mask_type": condition["mask_type"],
                "masked_patch_count": int(condition["patch_mask"].sum().item()),
                "reference_top1_id": baseline["reference_id"],
                "baseline_reference_similarity": baseline["reference_similarity"],
                "occluded_reference_similarity": reference_similarity,
                "reference_similarity_drop": baseline["reference_similarity"] - reference_similarity,
                "occluded_reference_rank": _rank(occluded_scores[condition_index], reference_position),
                "reference_rank_increase": _rank(occluded_scores[condition_index], reference_position) - 1,
                "baseline_target_similarity": baseline["target_similarity"],
                "occluded_target_similarity": target_similarity,
                "target_similarity_drop": baseline["target_similarity"] - target_similarity,
                "baseline_target_rank": baseline["target_rank"],
                "occluded_target_rank": _rank(occluded_scores[condition_index], target_position),
                "target_rank_increase": _rank(occluded_scores[condition_index], target_position) - baseline["target_rank"],
                "query_embedding_cosine": cosine,
                "query_embedding_cosine_drop": 1.0 - cosine,
            }
        )

    run_name = f"{args.category}_probe{args.num_probes}"
    output_dir = Path(args.output_root) / args.category / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "results.csv", rows)
    np.savez_compressed(
        output_dir / "occlusion_masks_and_attention.npz",
        average_attention=attention_maps.numpy(),
        patch_masks=np.stack(masks_for_npz).reshape(
            len(probes), len(ratios), 2, pool_size, pool_size
        ),
        ratios=np.asarray(ratios),
    )

    metric_names = (
        "reference_similarity_drop",
        "reference_rank_increase",
        "query_embedding_cosine_drop",
        "target_similarity_drop",
        "target_rank_increase",
    )
    summary_by_ratio: dict[str, Any] = {}
    for ratio_index, ratio in enumerate(ratios):
        summary_by_ratio[f"{int(round(ratio * 100))}%"] = {
            metric: _paired_metric(
                rows,
                ratio,
                metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=seed + ratio_index,
            )
            for metric in metric_names
        }
    summary = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "category": args.category,
        "probe_ids": [probe["probe_id"] for probe in probes],
        "n_probes": len(probes),
        "ratios": ratios,
        "mask_definition": (
            "Select equal-count highest/lowest patches from the 7x7 average-stage attention map; "
            "set their pixels to ImageNet mean (zero after normalization)."
        ),
        "primary_endpoint": (
            "Drop in similarity and increase in rank of the unoccluded query's top-1 gallery result."
        ),
        "secondary_endpoint": "FashionIQ ground-truth target similarity/rank change.",
        "inference_only": True,
        "statistics": "Paired high-minus-low bootstrap 95% CI and exact two-sided sign-flip p-value.",
        "by_ratio": summary_by_ratio,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    preview_ratio = min(ratios, key=lambda value: abs(value - 0.20))
    preview_conditions = [
        condition
        for condition in conditions
        if condition["probe_index"] == 0 and abs(condition["ratio"] - preview_ratio) < 1e-9
    ]
    source_image = _model_input_image(samples[0]["query_image"])
    preview_images = {condition["mask_type"]: _model_input_image(condition["image"]) for condition in preview_conditions}
    source_image.save(output_dir / "preview_source.png")
    preview_images["high"].save(output_dir / "preview_high_mask.png")
    preview_images["low"].save(output_dir / "preview_low_mask.png")

    report_output = Path(args.report_output)
    _build_report_figure(
        summary,
        source_image,
        preview_images["high"],
        preview_images["low"],
        preview_ratio,
        report_output,
    )

    table_lines = [
        "# Occlusion faithfulness",
        "",
        "Primary endpoint uses each unoccluded query's top-1 result as the fixed reference.",
        "",
        "| Mask ratio | Δsim high | Δsim low | high−low [bootstrap 95% CI] | Δrank high | Δrank low | p (Δsim) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in ratios:
        item = summary_by_ratio[f"{int(ratio * 100)}%"]
        sim = item["reference_similarity_drop"]
        rank = item["reference_rank_increase"]
        table_lines.append(
            f"| {int(ratio * 100)}% | {sim['mean_high']:.4f} | {sim['mean_low']:.4f} | "
            f"{sim['mean_high_minus_low']:.4f} [{sim['bootstrap_95_ci'][0]:.4f}; {sim['bootstrap_95_ci'][1]:.4f}] | "
            f"{rank['mean_high']:.2f} | {rank['mean_low']:.2f} | {sim['sign_flip_pvalue']:.4f} |"
        )
    table_output = Path(args.table_output)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    print(f"Device: {device}; checkpoint: {checkpoint_path}")
    print(f"Wrote {len(rows)} paired-condition rows: {output_dir / 'results.csv'}")
    print(f"Wrote summary: {output_dir / 'summary.json'}")
    print(f"Wrote report figure: {report_output}")
    print(f"Wrote report table: {table_output}")


if __name__ == "__main__":
    main()
