from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image

from aacl_fashion.config import load_config
from aacl_fashion.data.builders import build_eval_loaders
from aacl_fashion.data.transforms import build_transforms
from aacl_fashion.evaluation import encode_gallery
from aacl_fashion.models import build_model
from aacl_fashion.utils.checkpoint import load_checkpoint
from aacl_fashion.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a visual FashionIQ retrieval demo from a checkpoint.")
    parser.add_argument("--config", default="configs/fashioniq.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or another checkpoint.")
    parser.add_argument("--category", default="dress", help="FashionIQ category: dress, shirt, or toptee.")
    parser.add_argument("--query-index", type=int, default=0, help="Validation query index to use.")
    parser.add_argument("--query-image", default=None, help="Optional custom query image path.")
    parser.add_argument("--text", default=None, help="Modification text for --query-image.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrieved gallery images.")
    parser.add_argument("--max-gallery", type=int, default=None, help="Optional limit for a faster CPU preview.")
    parser.add_argument("--output", default=None, help="Output contact-sheet image path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON path for ranked results.")
    return parser.parse_args()


def _resolve_dataset_image(dataset: Any, image_id: str) -> Path:
    resolver = getattr(dataset, "_resolve_image", None)
    if resolver is None:
        raise TypeError("The selected dataset does not expose image paths for visualization.")
    return Path(resolver(image_id))


def _preview_from_tensor(tensor: torch.Tensor, path: Path) -> Path:
    image = to_pil_image(tensor.detach().cpu().clamp(0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _load_custom_query(path: str, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    dataset_config = config["dataset"]
    transform = build_transforms(
        train=False,
        image_size=int(dataset_config.get("image_size", 224)),
        resize_size=int(dataset_config.get("resize_size", 256)),
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def _query_preview_path(query_dataset: Any, sample: dict[str, Any], output_path: Path) -> Path:
    query_id = str(sample["query_id"])
    try:
        return _resolve_dataset_image(query_dataset, query_id)
    except TypeError:
        return _preview_from_tensor(sample["query_image"], output_path.parent / "assets" / f"query_{query_id}.jpg")


def _gallery_preview_path(gallery_dataset: Any, image_id: str, index: int, output_path: Path) -> Path:
    if isinstance(gallery_dataset, Subset):
        mapped_index = gallery_dataset.indices[index]
        return _gallery_preview_path(gallery_dataset.dataset, image_id, mapped_index, output_path)
    try:
        return _resolve_dataset_image(gallery_dataset, image_id)
    except TypeError:
        sample = gallery_dataset[index]
        return _preview_from_tensor(sample["image"], output_path.parent / "assets" / f"gallery_{image_id}.jpg")


def _make_contact_sheet(
    query_path: Path,
    caption: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    thumb_size: int = 180,
) -> None:
    title_h = 42
    label_h = 48
    gap = 12
    columns = len(rows) + 1
    width = columns * thumb_size + (columns + 1) * gap
    height = title_h + thumb_size + label_h + gap * 3

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), f"Query: {caption[:120]}", fill=(20, 20, 20))

    tiles = [{"label": "query", "path": query_path, "score": None}] + rows
    for index, tile in enumerate(tiles):
        x = gap + index * (thumb_size + gap)
        y = title_h + gap
        image = Image.open(tile["path"]).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        tile_bg = Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
        paste_x = x + (thumb_size - image.width) // 2
        paste_y = y + (thumb_size - image.height) // 2
        canvas.paste(tile_bg, (x, y))
        canvas.paste(image, (paste_x, paste_y))

        if tile["score"] is None:
            label = tile["label"]
        else:
            label = f"#{index} {tile['image_id']}\nscore={tile['score']:.4f}"
        draw.multiline_text((x, y + thumb_size + 6), label, fill=(20, 20, 20), spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    if args.query_image and not args.text:
        raise ValueError("--text is required when using --query-image.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    query_loader, gallery_loader = build_eval_loaders(config, args.category)
    gallery_dataset = gallery_loader.dataset
    if args.max_gallery is not None:
        max_gallery = min(int(args.max_gallery), len(gallery_dataset))
        gallery_dataset = Subset(gallery_dataset, range(max_gallery))
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=gallery_loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    gallery_embeddings, gallery_ids = encode_gallery(
        model=model,
        gallery_loader=gallery_loader,
        device=device,
        amp=bool(config["training"].get("amp", True)) and device.type == "cuda",
    )

    output_path = Path(args.output) if args.output else Path("outputs/retrieval_demo") / args.category / "result.jpg"
    query_id = None
    if args.query_image:
        query_path = Path(args.query_image)
        caption = str(args.text)
        query_tensor = _load_custom_query(args.query_image, config, device)
    else:
        query_dataset = query_loader.dataset
        if args.query_index < 0 or args.query_index >= len(query_dataset):
            raise IndexError(f"--query-index must be in [0, {len(query_dataset) - 1}]")
        sample = query_dataset[args.query_index]
        query_id = str(sample["query_id"])
        query_path = _query_preview_path(query_dataset, sample, output_path)
        caption = str(sample["captions"])
        query_tensor = sample["query_image"].unsqueeze(0).to(device)

    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    autocast = torch.amp.autocast("cuda") if amp else nullcontext()
    with autocast:
        query_embedding = model.encode_query(query_tensor, [caption]).detach().cpu()

    scores = (query_embedding @ gallery_embeddings.T).squeeze(0)
    if query_id is not None:
        for index, image_id in enumerate(gallery_ids):
            if image_id == query_id:
                scores[index] = float("-inf")

    top_k = min(int(args.top_k), len(gallery_ids))
    top_scores, top_indices = torch.topk(scores, k=top_k)
    rows: list[dict[str, Any]] = []
    for rank, (score, index) in enumerate(zip(top_scores.tolist(), top_indices.tolist(), strict=True), start=1):
        image_id = gallery_ids[index]
        image_path = _gallery_preview_path(gallery_dataset, image_id, index, output_path)
        rows.append(
            {
                "rank": rank,
                "image_id": image_id,
                "score": float(score),
                "path": str(image_path),
            }
        )

    _make_contact_sheet(query_path=query_path, caption=caption, rows=rows, output_path=output_path)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "category": args.category,
            "query_image": str(query_path),
            "query_id": query_id,
            "caption": caption,
            "results": rows,
            "contact_sheet": str(output_path),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Query image: {query_path}")
    print(f"Text: {caption}")
    print(f"Contact sheet: {output_path}")
    for row in rows:
        print(f"#{row['rank']:02d} score={row['score']:.4f} image_id={row['image_id']} path={row['path']}")


if __name__ == "__main__":
    main()
