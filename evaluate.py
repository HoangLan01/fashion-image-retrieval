from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from aacl_fashion.config import load_config, resolve_categories
from aacl_fashion.data.builders import build_eval_loaders
from aacl_fashion.evaluation import evaluate_model_detailed
from aacl_fashion.models import build_model
from aacl_fashion.utils.checkpoint import load_checkpoint
from aacl_fashion.utils.device import resolve_device
from aacl_fashion.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AACL retrieval metrics.")
    parser.add_argument("--config", default="configs/fashioniq.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--category", default="dress", help="FashionIQ category or 'all'.")
    parser.add_argument("--json-output", default=None, help="Optional path for machine-readable metrics.")
    parser.add_argument(
        "--per-query-output",
        default=None,
        help="Optional CSV path containing target ranks, scores, top-1 IDs, and margins.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda, cuda:0, or cpu. CUDA_VISIBLE_DEVICES is also supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(args.device)
    model = build_model(config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config and checkpoint_config.get("model") != config.get("model"):
        raise ValueError("Checkpoint model config does not match the evaluation config.")
    model.load_state_dict(checkpoint["model"])

    evaluation_config = config["evaluation"]
    recall_ks = tuple(int(k) for k in evaluation_config.get("recall_ks", [10, 50]))
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"

    all_metrics = {}
    all_details: list[dict[str, object]] = []
    for category in resolve_categories(config, args.category):
        query_loader, gallery_loader = build_eval_loaders(config, category)
        metrics, details = evaluate_model_detailed(
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            recall_ks=recall_ks,
            exclude_query=bool(evaluation_config.get("exclude_query", True)),
            amp=amp,
        )
        all_metrics[category] = metrics
        all_details.extend({"category": category, **row} for row in details)
        print(f"[{category}] {metrics}")

    if len(all_metrics) > 1:
        metric_names = sorted({name for metrics in all_metrics.values() for name in metrics})
        average = {
            name: sum(metrics.get(name, 0.0) for metrics in all_metrics.values()) / len(all_metrics)
            for name in metric_names
        }
        print(f"[average] {average}")
    else:
        average = None

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_category": checkpoint.get("category"),
            "device": str(device),
            "categories": all_metrics,
            "average": average,
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Metrics JSON: {output_path}")

    if args.per_query_output:
        detail_path = Path(args.per_query_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "category",
            "query_id",
            "target_id",
            "target_rank",
            "target_score",
            "top1_id",
            "top1_score",
            "top1_top2_margin",
        ]
        with detail_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_details)
        print(f"Per-query CSV: {detail_path}")


if __name__ == "__main__":
    main()
