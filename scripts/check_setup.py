from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

import yaml


CORE_PACKAGES = ["torch", "torchvision", "yaml", "PIL", "tqdm"]
CAPTION_PATTERNS = (
    "cap.{category}.{split}.json",
    "{category}.{split}.json",
    "{split}.{category}.json",
)
SPLIT_PATTERNS = (
    "split.{category}.{split}.json",
    "{category}.{split}.json",
    "{split}.{category}.json",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def status(ok: bool, message: str) -> None:
    prefix = "[OK]" if ok else "[WARN]"
    print(f"{prefix} {message}")


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 9)
    status(ok, f"Python {platform.python_version()} at {sys.executable}")
    return ok


def required_packages_for_config(config: dict[str, Any]) -> list[str]:
    packages = list(CORE_PACKAGES)
    model = config.get("model", {})
    image_type = model.get("image_encoder", {}).get("type", "swin")
    text_type = model.get("text_encoder", {}).get("type", "distilbert")
    if image_type == "swin":
        packages.append("timm")
    if text_type in {"distilbert", "clip"}:
        packages.append("transformers")
    return packages


def check_packages(config: dict[str, Any]) -> bool:
    ok = True
    for package in required_packages_for_config(config):
        found = has_module(package)
        ok = ok and found
        status(found, f"Import package: {package}")
    return ok


def check_cuda() -> bool:
    if not has_module("torch"):
        status(False, "Cannot check CUDA because torch is not installed.")
        return False

    import torch

    cuda_available = torch.cuda.is_available()
    status(cuda_available, f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"     CUDA version: {torch.version.cuda}")
        print(f"     GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_gb = props.total_memory / 1024**3
            print(f"     GPU {index}: {props.name} ({total_gb:.1f} GB)")
    else:
        print("     Synthetic smoke test can run on CPU, but FashionIQ training should use CUDA.")
    return cuda_available


def find_pattern(base_dir: Path, patterns: tuple[str, ...], category: str, split: str) -> Path | None:
    for pattern in patterns:
        candidate = base_dir / pattern.format(category=category, split=split)
        if candidate.exists():
            return candidate
    return None


def check_fashioniq(config: dict[str, Any]) -> bool:
    dataset = config.get("dataset", {})
    if dataset.get("name") != "fashioniq":
        status(True, f"Dataset is '{dataset.get('name')}', FashionIQ file checks skipped.")
        return True

    root = Path(dataset.get("root", "data/fashioniq"))
    categories = dataset.get("categories", ["dress", "shirt", "toptee"])
    train_split = dataset.get("train_split", "train")
    val_split = dataset.get("val_split", "val")
    ok = True

    for subdir in ("images", "captions", "image_splits"):
        exists = (root / subdir).exists()
        ok = ok and exists
        status(exists, f"FashionIQ directory: {root / subdir}")

    for category in categories:
        for split in (train_split, val_split):
            caption = find_pattern(root / "captions", CAPTION_PATTERNS, category, split)
            split_file = find_pattern(root / "image_splits", SPLIT_PATTERNS, category, split)
            ok = ok and caption is not None and split_file is not None
            status(caption is not None, f"Caption file for {category}/{split}")
            status(split_file is not None, f"Image split file for {category}/{split}")

    return ok


def print_next_steps(config_path: Path, config: dict[str, Any], cuda_ok: bool, data_ok: bool) -> None:
    dataset_name = config.get("dataset", {}).get("name", "fashioniq")
    categories = config.get("dataset", {}).get("categories", ["dress"])
    category = categories[0] if categories else "dress"

    print("\nNext commands:")
    print("  python train.py --config configs/synthetic_smoke.yaml --category dress")
    if dataset_name == "fashioniq" and data_ok:
        print(f"  python train.py --config {config_path.as_posix()} --category {category}")
    elif dataset_name == "fashioniq":
        print("  Prepare data/fashioniq before running real FashionIQ training.")
    if not cuda_ok:
        print("  Install a CUDA-enabled PyTorch build before production training.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AACL environment and dataset setup.")
    parser.add_argument("--config", default="configs/fashioniq.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    python_ok = check_python()
    packages_ok = check_packages(config)
    cuda_ok = check_cuda()
    data_ok = check_fashioniq(config)

    print("\nSummary:")
    status(python_ok and packages_ok, "Python and dependencies")
    status(cuda_ok, "CUDA readiness for real training")
    status(data_ok, "Dataset readiness for selected config")
    print_next_steps(config_path, config, cuda_ok=cuda_ok, data_ok=data_ok)


if __name__ == "__main__":
    main()
