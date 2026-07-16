from __future__ import annotations

import argparse

from aacl_fashion.config import load_config, resolve_categories
from aacl_fashion.training import train_one_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AACL on FashionIQ.")
    parser.add_argument("--config", default="configs/fashioniq.yaml", help="Path to YAML config.")
    parser.add_argument("--category", default="dress", help="FashionIQ category or 'all'.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path to resume, or 'auto' for the run's latest.pt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start again and replace known artifacts in the selected run directory.",
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
    categories = resolve_categories(config, args.category)
    if args.resume not in (None, "auto") and len(categories) != 1:
        raise ValueError("An explicit --resume checkpoint can only be used with one category.")
    if args.resume is not None and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")

    all_metrics = {}
    for category in categories:
        metrics = train_one_category(
            config,
            category,
            resume=args.resume,
            overwrite=args.overwrite,
            device_name=args.device,
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
