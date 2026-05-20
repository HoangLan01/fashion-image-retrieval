from __future__ import annotations

import argparse

import torch

from aacl_fashion.config import load_config, resolve_categories
from aacl_fashion.data.builders import build_eval_loaders
from aacl_fashion.evaluation import evaluate_model
from aacl_fashion.models import build_model
from aacl_fashion.utils.checkpoint import load_checkpoint
from aacl_fashion.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AACL retrieval metrics.")
    parser.add_argument("--config", default="configs/fashioniq.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--category", default="dress", help="FashionIQ category or 'all'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    evaluation_config = config["evaluation"]
    recall_ks = tuple(int(k) for k in evaluation_config.get("recall_ks", [10, 50]))
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"

    all_metrics = {}
    for category in resolve_categories(config, args.category):
        query_loader, gallery_loader = build_eval_loaders(config, category)
        metrics = evaluate_model(
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            recall_ks=recall_ks,
            exclude_query=bool(evaluation_config.get("exclude_query", True)),
            amp=amp,
        )
        all_metrics[category] = metrics
        print(f"[{category}] {metrics}")

    if len(all_metrics) > 1:
        metric_names = sorted({name for metrics in all_metrics.values() for name in metrics})
        average = {
            name: sum(metrics.get(name, 0.0) for metrics in all_metrics.values()) / len(all_metrics)
            for name in metric_names
        }
        print(f"[average] {average}")


if __name__ == "__main__":
    main()
