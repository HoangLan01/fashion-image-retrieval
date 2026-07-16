from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three evaluations required for one lexical-holdout category."
    )
    parser.add_argument("--category", required=True, choices=["shirt", "toptee"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--baseline-root",
        default="outputs/fashioniq_improved/l40_shared_seed42",
    )
    parser.add_argument(
        "--holdout-root",
        default="outputs/fashioniq_lexical_holdout/seed42",
    )
    return parser.parse_args()


def _evaluate_command(
    config: str,
    checkpoint: Path,
    category: str,
    output_dir: Path,
    suffix: str,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "evaluate.py"),
        "--config",
        config,
        "--checkpoint",
        str(checkpoint),
        "--category",
        category,
        "--device",
        device,
        "--json-output",
        str(output_dir / f"evaluation_{suffix}.json"),
        "--per-query-output",
        str(output_dir / f"per_query_{suffix}.csv"),
    ]


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_root) / args.category
    holdout_dir = Path(args.holdout_root) / args.category
    required_checkpoints = [baseline_dir / "best.pt", holdout_dir / "best.pt"]
    missing = [path for path in required_checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n  " + "\n  ".join(map(str, missing)))

    jobs = [
        (
            "baseline model on lexical validation",
            _evaluate_command(
                "configs/fashioniq_lexical_holdout_targeted.yaml",
                baseline_dir / "best.pt",
                args.category,
                baseline_dir,
                "lexical_val",
                args.device,
            ),
        ),
        (
            "holdout model on full validation",
            _evaluate_command(
                "configs/fashioniq_lexical_holdout.yaml",
                holdout_dir / "best.pt",
                args.category,
                holdout_dir,
                "full_val",
                args.device,
            ),
        ),
        (
            "holdout model on lexical validation",
            _evaluate_command(
                "configs/fashioniq_lexical_holdout_targeted.yaml",
                holdout_dir / "best.pt",
                args.category,
                holdout_dir,
                "lexical_val",
                args.device,
            ),
        ),
    ]
    for label, command in jobs:
        print(f"\n=== {label} ===", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print(f"\nCompleted lexical evaluations for {args.category}.")


if __name__ == "__main__":
    main()
