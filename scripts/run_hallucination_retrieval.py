from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from aacl_fashion.config import load_config
from aacl_fashion.data.builders import build_eval_loaders
from aacl_fashion.evaluation import encode_gallery
from aacl_fashion.models import build_model
from aacl_fashion.utils.checkpoint import load_checkpoint
from aacl_fashion.utils.device import resolve_device
from aacl_fashion.utils.seed import set_seed


RESULT_FIELDS = (
    "prompt_id",
    "probe_id",
    "scenario",
    "category",
    "query_index",
    "query_id",
    "target_id",
    "original_caption",
    "prompt",
    "expected_behavior",
    "rank",
    "image_id",
    "score",
    "image_path",
    "is_original_target",
    "top1_score",
    "top1_top2_margin",
    "contact_sheet",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-k retrieval and create qualitative contact sheets for fixed probes."
    )
    parser.add_argument("--config", default="configs/fashioniq_l40_shared.yaml")
    parser.add_argument("--prompts", default="outputs/hallucination/prompts.json")
    parser.add_argument("--category", required=True, choices=["dress", "shirt", "toptee"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-root", default="outputs/fashioniq_improved/l40_shared_seed42"
    )
    parser.add_argument("--output-root", default="outputs/hallucination")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-contact-sheets", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _image_tile(path: str | Path, size: int) -> Image.Image:
    source = Image.open(path).convert("RGB")
    source.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), (242, 242, 242))
    tile.paste(source, ((size - source.width) // 2, (size - source.height) // 2))
    return tile


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
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


def _make_contact_sheet(
    probe: dict[str, Any],
    scenario_rows: dict[str, list[dict[str, Any]]],
    query_path: Path,
    output_path: Path,
) -> None:
    scenarios = [
        "in_domain",
        "paraphrase",
        "identity",
        "contradiction",
        "ood",
        "unsatisfiable",
    ]
    thumb = 150
    gap = 10
    text_width = 360
    header_height = 42
    label_height = 44
    row_height = thumb + label_height + gap
    width = text_width + 6 * (thumb + gap) + gap
    height = header_height + len(scenarios) * row_height + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (gap, 12),
        f"Probe {probe['probe_id']} | category={probe['category']} | query={probe['query_id']}",
        fill=(15, 15, 15),
        font=font,
    )

    for scenario_index, scenario in enumerate(scenarios):
        rows = scenario_rows[scenario]
        y = header_height + scenario_index * row_height
        prompt = str(rows[0]["prompt"])
        draw.text((gap, y + 4), scenario.upper(), fill=(20, 70, 130), font=font)
        wrapped = _wrap_text(draw, prompt, text_width - 2 * gap, font)[:8]
        draw.multiline_text((gap, y + 22), "\n".join(wrapped), fill=(30, 30, 30), font=font, spacing=2)

        query_x = text_width
        canvas.paste(_image_tile(query_path, thumb), (query_x, y))
        draw.rectangle((query_x, y, query_x + thumb - 1, y + thumb - 1), outline=(30, 90, 210), width=4)
        draw.text((query_x, y + thumb + 5), "SOURCE", fill=(30, 90, 210), font=font)

        for result_index, row in enumerate(rows):
            x = text_width + (result_index + 1) * (thumb + gap)
            canvas.paste(_image_tile(row["image_path"], thumb), (x, y))
            draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline=(150, 150, 150), width=3)
            label = f"#{row['rank']} {row['image_id']}"
            draw.multiline_text((x, y + thumb + 5), label, fill=(20, 20, 20), font=font, spacing=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2 so the top-1/top-2 margin is defined.")
    if args.query_batch_size < 1:
        raise ValueError("--query-batch-size must be positive.")

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(args.device)
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    prompt_payload = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    prompt_manifest_sha256 = hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest()
    prompts = [item for item in prompt_payload["prompts"] if item["category"] == args.category]
    if not prompts:
        raise ValueError(f"No prompts found for category {args.category}.")

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(args.checkpoint_root) / args.category / "best.pt"
    )
    output_dir = Path(args.output_root) / args.category
    contact_dir = output_dir / "contact_sheets"

    model = build_model(config["model"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    query_loader, gallery_loader = build_eval_loaders(config, args.category)
    query_dataset = query_loader.dataset
    gallery_dataset = gallery_loader.dataset
    gallery_embeddings, gallery_ids = encode_gallery(model, gallery_loader, device, amp=amp)
    gallery_embeddings = gallery_embeddings.float()
    gallery_index = {str(image_id): index for index, image_id in enumerate(gallery_ids)}
    if args.top_k > len(gallery_ids) - 1:
        raise ValueError("--top-k exceeds the gallery size after excluding the source image.")

    all_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), args.query_batch_size):
        batch_prompts = prompts[start : start + args.query_batch_size]
        image_tensors = []
        captions = []
        for prompt in batch_prompts:
            sample = query_dataset[int(prompt["query_index"])]
            if str(sample["query_id"]) != str(prompt["query_id"]):
                raise ValueError(
                    f"Probe manifest no longer matches the dataset at index {prompt['query_index']}: "
                    f"expected {prompt['query_id']}, found {sample['query_id']}."
                )
            image_tensors.append(sample["query_image"])
            captions.append(str(prompt["prompt"]))
        images = torch.stack(image_tensors).to(device, non_blocking=True)
        autocast = torch.amp.autocast("cuda") if amp else nullcontext()
        with autocast:
            query_embeddings = model.encode_query(images, captions)
        similarities = query_embeddings.detach().cpu().float() @ gallery_embeddings.T

        for row_index, prompt in enumerate(batch_prompts):
            query_id = str(prompt["query_id"])
            source_position = gallery_index.get(query_id)
            if source_position is not None:
                similarities[row_index, source_position] = -torch.inf
            top_scores, top_indices = similarities[row_index].topk(args.top_k)
            margin = float(top_scores[0].item() - top_scores[1].item())
            contact_sheet = str(contact_dir / f"{prompt['probe_id']}.jpg")
            score_rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "probe_id": prompt["probe_id"],
                    "scenario": prompt["scenario"],
                    "category": args.category,
                    "query_id": query_id,
                    "target_id": prompt["target_id"],
                    "max_similarity": float(top_scores[0].item()),
                    "top1_top2_margin": margin,
                    "top1_id": gallery_ids[int(top_indices[0].item())],
                }
            )
            for rank, (score, gallery_position) in enumerate(
                zip(top_scores.tolist(), top_indices.tolist(), strict=True), start=1
            ):
                image_id = str(gallery_ids[gallery_position])
                image_path = Path(gallery_dataset._resolve_image(image_id))
                all_rows.append(
                    {
                        **{field: prompt.get(field, "") for field in RESULT_FIELDS},
                        "category": args.category,
                        "rank": rank,
                        "image_id": image_id,
                        "score": float(score),
                        "image_path": str(image_path),
                        "is_original_target": int(image_id == str(prompt["target_id"])),
                        "top1_score": float(top_scores[0].item()),
                        "top1_top2_margin": margin,
                        "contact_sheet": contact_sheet,
                    }
                )

    _write_csv(output_dir / "results.csv", all_rows, RESULT_FIELDS)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    score_fields = tuple(score_rows[0].keys())
    _write_csv(output_dir / "prompt_scores.csv", score_rows, score_fields)

    if not args.no_contact_sheets:
        by_probe: dict[str, dict[str, list[dict[str, Any]]]] = {}
        probe_metadata: dict[str, dict[str, Any]] = {}
        for row in all_rows:
            by_probe.setdefault(str(row["probe_id"]), {}).setdefault(str(row["scenario"]), []).append(row)
            probe_metadata[str(row["probe_id"])] = row
        for probe_id, scenario_rows in by_probe.items():
            probe = probe_metadata[probe_id]
            query_path = Path(query_dataset._resolve_image(str(probe["query_id"])))
            _make_contact_sheet(probe, scenario_rows, query_path, contact_dir / f"{probe_id}.jpg")

    metadata = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "category": args.category,
        "device": str(device),
        "amp": amp,
        "top_k": args.top_k,
        "num_prompts": len(prompts),
        "num_ranked_rows": len(all_rows),
        "prompts_file": args.prompts,
        "prompts_sha256": prompt_manifest_sha256,
        "mixed_gallery": False,
        "mixed_gallery_reason": (
            "Each category uses a separately trained checkpoint; a mixed gallery would confound "
            "prompt robustness with category routing."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "NEEDS_RERUN.md").unlink(missing_ok=True)
    print(f"Device: {device}; checkpoint: {checkpoint_path}")
    print(f"Wrote {len(all_rows)} ranked rows and {len(prompts)} prompt scores to {output_dir}")
    if not args.no_contact_sheets:
        print(f"Wrote {len(set(row['probe_id'] for row in all_rows))} contact sheets to {contact_dir}")


if __name__ == "__main__":
    main()
