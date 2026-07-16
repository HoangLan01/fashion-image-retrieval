from __future__ import annotations

import argparse
import json
import sys
import textwrap
from contextlib import nullcontext
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from aacl_fashion.attention import (
    compute_attention_flow,
    map_similarity,
    merge_wordpiece_scores,
    split_image_attention_maps,
)
from aacl_fashion.config import load_config
from aacl_fashion.data.builders import build_eval_loaders
from aacl_fashion.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from aacl_fashion.evaluation import encode_gallery
from aacl_fashion.models import build_model
from aacl_fashion.utils.checkpoint import load_checkpoint
from aacl_fashion.utils.device import resolve_device
from aacl_fashion.utils.seed import set_seed


DEFAULT_PROMPTS = {
    "dress": ("Make the dress longer.", "Make the dress have longer sleeves."),
    "shirt": ("Make the shirt have longer sleeves.", "Make the shirt have a different graphic."),
    "toptee": ("Make the top have longer sleeves.", "Make the top have a different graphic."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract additive-attention flow and build counterfactual AACL visualizations."
    )
    parser.add_argument("--config", default="configs/fashioniq_l40_shared.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-root", default="outputs/fashioniq_improved/l40_shared_seed42"
    )
    parser.add_argument("--category", default="shirt", choices=["dress", "shirt", "toptee"])
    parser.add_argument("--probe-id", default=None, help="Probe ID from outputs/hallucination/prompts.json.")
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument("--prompts-manifest", default="outputs/hallucination/prompts.json")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Counterfactual prompt; repeat at least twice. Defaults to a fixed pair.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/attention")
    parser.add_argument(
        "--report-output", default="outputs/report_assets/fig_attention_counterfactual.png"
    )
    parser.add_argument("--save-heads", action="store_true")
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


def _heat_colors(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.8 * values, 0.0, 1.0)
    green = np.clip(1.8 * (values - 0.28), 0.0, 1.0)
    blue = np.clip(1.2 * (0.55 - values), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def _overlay(source: Image.Image, heatmap: torch.Tensor) -> Image.Image:
    small = np.asarray(heatmap.detach().cpu(), dtype=np.float32)
    heat = Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8), mode="L")
    heat = heat.resize(source.size, Image.Resampling.BILINEAR)
    values = np.asarray(heat, dtype=np.float32) / 255.0
    colors = (_heat_colors(values) * 255.0).astype(np.uint8)
    source_values = np.asarray(source, dtype=np.float32)
    alpha = (0.18 + 0.48 * values)[..., None]
    blended = source_values * (1.0 - alpha) + colors.astype(np.float32) * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _wrap(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_token_flow(
    draw: ImageDraw.ImageDraw,
    tokens: list[dict[str, Any]],
    x: int,
    y: int,
    max_width: int,
) -> None:
    font = _font(13)
    label_font = _font(13, bold=True)
    draw.text((x, y), "Text flow:", fill=(35, 35, 35), font=label_font)
    cursor_x = x + 75
    cursor_y = y - 2
    for token in tokens:
        word = str(token["word"])
        score = float(token["score"])
        box_width = int(draw.textlength(word, font=font)) + 12
        if cursor_x + box_width > x + max_width:
            cursor_x = x + 75
            cursor_y += 24
        fill = (255, int(245 - 170 * score), int(245 - 190 * score))
        draw.rounded_rectangle(
            (cursor_x, cursor_y, cursor_x + box_width, cursor_y + 20),
            radius=4,
            fill=fill,
            outline=(210, 120, 120),
        )
        draw.text((cursor_x + 6, cursor_y + 2), word, fill=(45, 20, 20), font=font)
        cursor_x += box_width + 5


def _head_grid(
    source: Image.Image,
    maps: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    # maps: [head, H, W]
    columns = 4
    tile = source.width
    rows = (maps.shape[0] + columns - 1) // columns
    header = 38
    label = 24
    canvas = Image.new("RGB", (columns * tile, header + rows * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(20, 20, 20), font=_font(18, bold=True))
    for head in range(maps.shape[0]):
        x = (head % columns) * tile
        y = header + (head // columns) * (tile + label)
        canvas.paste(_overlay(source, maps[head]), (x, y))
        draw.text((x + 6, y + tile + 3), f"head {head + 1}", fill=(25, 25, 25), font=_font(13))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    prompts = list(args.prompts or DEFAULT_PROMPTS[args.category])
    if len(prompts) < 2:
        raise ValueError("Provide at least two --prompt values for a counterfactual comparison.")
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(args.device)
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"

    manifest = json.loads(Path(args.prompts_manifest).read_text(encoding="utf-8"))
    probe_id = args.probe_id or f"{args.category}_q01"
    probe = next(
        (row for row in manifest["probes"] if row["probe_id"] == probe_id),
        None,
    )
    if probe is None and args.query_index is None:
        raise ValueError(f"Probe {probe_id!r} not found; provide --query-index explicitly.")
    query_index = int(args.query_index if args.query_index is not None else probe["query_index"])

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
    sample = query_dataset[query_index]
    query_id = str(sample["query_id"])
    target_id = str(sample["target_id"])
    if probe is not None and query_id != str(probe["query_id"]):
        raise ValueError(f"Manifest mismatch: expected {probe['query_id']}, found {query_id}.")

    query_images = sample["query_image"].unsqueeze(0).repeat(len(prompts), 1, 1, 1).to(device)
    autocast = torch.amp.autocast("cuda") if amp else nullcontext()
    with autocast:
        attention_output = model.encode_query_with_attention(query_images, prompts)

    weights = attention_output.attention_weights.detach().cpu().float()
    text_mask = attention_output.text_features.attention_mask.detach().cpu()
    image_mask = torch.ones(
        len(prompts), attention_output.image_token_count, dtype=text_mask.dtype
    )
    combined_mask = torch.cat([image_mask, text_mask], dim=1)
    flow = compute_attention_flow(weights, attention_mask=combined_mask)
    pool_size = int(config["model"]["image_encoder"].get("pool_size", 7))
    maps = split_image_attention_maps(
        flow,
        image_token_count=attention_output.image_token_count,
        pool_size=pool_size,
    )

    text_flow = flow.mean(dim=1)[:, attention_output.image_token_count :]
    token_labels = attention_output.text_features.token_labels
    if token_labels is None:
        token_labels = [[f"token_{index}" for index in range(text_flow.shape[1])] for _ in prompts]
    merged_tokens = [
        merge_wordpiece_scores(labels, scores)
        for labels, scores in zip(token_labels, text_flow, strict=True)
    ]

    gallery_embeddings, gallery_ids = encode_gallery(model, gallery_loader, device, amp=amp)
    similarities = attention_output.embedding.detach().cpu().float() @ gallery_embeddings.float().T
    gallery_index = {str(image_id): index for index, image_id in enumerate(gallery_ids)}
    source_position = gallery_index.get(query_id)
    if source_position is not None:
        similarities[:, source_position] = -torch.inf
    prompt_results: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        top_scores, top_indices = similarities[index].topk(min(args.top_k, len(gallery_ids) - 1))
        target_position = gallery_index.get(target_id)
        target_rank = None
        if target_position is not None:
            target_score = similarities[index, target_position]
            target_rank = int((similarities[index] > target_score).sum().item()) + 1
        prompt_results.append(
            {
                "prompt": prompt,
                "top_ids": [str(gallery_ids[position]) for position in top_indices.tolist()],
                "top_scores": [float(value) for value in top_scores.tolist()],
                "original_target_rank": target_rank,
                "text_tokens": merged_tokens[index],
            }
        )

    comparison = map_similarity(maps["average"][0], maps["average"][1])
    first_ids = set(prompt_results[0]["top_ids"])
    second_ids = set(prompt_results[1]["top_ids"])
    comparison["top_k_overlap"] = len(first_ids & second_ids)
    comparison["top_k_jaccard"] = len(first_ids & second_ids) / len(first_ids | second_ids)

    output_dir = Path(args.output_root) / args.category / probe_id
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _model_input_image(sample["query_image"])
    source.save(output_dir / "source_model_input.png")
    for index, prompt in enumerate(prompts, start=1):
        for map_name in ("stage3", "stage4", "average"):
            _overlay(source, maps[map_name][index - 1]).save(
                output_dir / f"prompt_{index:02d}_{map_name}.png"
            )
        if args.save_heads:
            for stage in ("stage3", "stage4"):
                _head_grid(
                    source,
                    maps[f"{stage}_per_head"][index - 1],
                    output_dir / f"prompt_{index:02d}_{stage}_heads.png",
                    f"{stage.upper()} per-head flow | prompt {index}",
                )

    raw_payload = {
        "attention_weights": weights,
        "attention_flow_per_head": flow,
        "combined_attention_mask": combined_mask,
        "stage3_per_head": maps["stage3_per_head"],
        "stage4_per_head": maps["stage4_per_head"],
        "stage3": maps["stage3"],
        "stage4": maps["stage4"],
        "average": maps["average"],
        "query_embedding": attention_output.embedding.detach().cpu().float(),
    }
    torch.save(raw_payload, output_dir / "attention_raw.pt")
    np.savez_compressed(
        output_dir / "attention_raw.npz",
        **{key: value.numpy() for key, value in raw_payload.items()},
    )

    metadata = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "category": args.category,
        "probe_id": probe_id,
        "query_index": query_index,
        "query_id": query_id,
        "target_id": target_id,
        "device": str(device),
        "amp": amp,
        "attention_shape": list(weights.shape),
        "image_token_count": attention_output.image_token_count,
        "stage_layout": f"Stage 3: first {pool_size * pool_size} tokens; Stage 4: next {pool_size * pool_size} tokens.",
        "flow_definition": "Multiply per-token alpha across composition blocks in log-space, min-max per head, then average heads.",
        "prompts": prompt_results,
        "comparison_first_two_prompts": comparison,
        "interpretation_caveat": "Attention variation is evidence of text-conditioned selection, not a causal explanation by itself.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    panel_size = source.width
    left_width = 310
    gap = 10
    title_height = 92
    row_height = panel_size + 108
    width = left_width + 4 * (panel_size + gap) + gap
    height = title_height + len(prompts) * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "AACL attention flow under counterfactual text", fill=(15, 15, 15), font=_font(24, True))
    draw.text(
        (12, 45),
        (
            f"category={args.category} | probe={probe_id} | query={query_id} | "
            f"map Pearson={comparison['pearson']:.3f} | JS={comparison['jensen_shannon']:.3f} | "
            f"top-{args.top_k} overlap={comparison['top_k_overlap']}"
        ),
        fill=(55, 55, 55),
        font=_font(14),
    )
    labels = ("Model input", "Stage 3", "Stage 4", "Average")
    for prompt_index, result in enumerate(prompt_results):
        y = title_height + prompt_index * row_height
        draw.rectangle((0, y, width, y + 30), fill=(232, 238, 247))
        draw.text((12, y + 6), f"PROMPT {prompt_index + 1}", fill=(20, 65, 120), font=_font(15, True))
        prompt_lines = _wrap(draw, result["prompt"], left_width - 24, _font(15))
        draw.multiline_text((12, y + 42), "\n".join(prompt_lines), fill=(25, 25, 25), font=_font(15), spacing=3)
        draw.multiline_text(
            (12, y + 112),
            "Top-5:\n" + "\n".join(result["top_ids"]),
            fill=(70, 70, 70),
            font=_font(12),
            spacing=2,
        )
        panels = [
            source,
            _overlay(source, maps["stage3"][prompt_index]),
            _overlay(source, maps["stage4"][prompt_index]),
            _overlay(source, maps["average"][prompt_index]),
        ]
        for panel_index, (label, panel) in enumerate(zip(labels, panels, strict=True)):
            x = left_width + gap + panel_index * (panel_size + gap)
            draw.text((x, y + 38), label, fill=(25, 25, 25), font=_font(14, True))
            canvas.paste(panel, (x, y + 60))
        _draw_token_flow(
            draw,
            result["text_tokens"],
            x=left_width + gap,
            y=y + panel_size + 70,
            max_width=width - left_width - 2 * gap,
        )

    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(report_output, optimize=True)
    print(f"Device: {device}; checkpoint: {checkpoint_path}")
    print(f"Attention shape: {list(weights.shape)}")
    print(f"Wrote raw attention and overlays: {output_dir}")
    print(f"Wrote counterfactual figure: {report_output}")
    print(f"Comparison: {comparison}")


if __name__ == "__main__":
    main()
