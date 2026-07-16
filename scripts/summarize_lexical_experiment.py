from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aacl_fashion.data.lexical_holdout import SURFACE_PATTERNS, captions_from_record, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full-vs-holdout lexical evaluations.")
    parser.add_argument("--categories", nargs="+", default=["shirt", "toptee"])
    parser.add_argument(
        "--baseline-root",
        default="outputs/fashioniq_improved/l40_shared_seed42",
    )
    parser.add_argument(
        "--holdout-root",
        default="outputs/fashioniq_lexical_holdout/seed42",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/lexical_holdout_comparison/comparison.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="outputs/report_assets/table_lexical_holdout_comparison.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-data-root", default="data/fashioniq_lexical_holdout")
    return parser.parse_args()


def _load_metrics(path: Path, category: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["categories"][category].items()}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rank_summary(rows: list[dict[str, str]]) -> dict[str, float | int]:
    ranks = [int(row["target_rank"]) for row in rows]
    return {
        "queries": len(ranks),
        "median_target_rank": float(statistics.median(ranks)),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _paired_comparison(
    baseline_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline_rows) != len(holdout_rows):
        raise ValueError("Baseline and holdout targeted query counts do not match.")
    rank_pairs = []
    for baseline_row, holdout_row in zip(baseline_rows, holdout_rows, strict=True):
        baseline_key = (baseline_row["query_id"], baseline_row["target_id"])
        holdout_key = (holdout_row["query_id"], holdout_row["target_id"])
        if baseline_key != holdout_key:
            raise ValueError(
                f"Targeted query order mismatch: baseline={baseline_key}, holdout={holdout_key}."
            )
        rank_pairs.append((int(baseline_row["target_rank"]), int(holdout_row["target_rank"])))
    result: dict[str, Any] = {
        "queries": len(rank_pairs),
        "rank_improved": sum(holdout_rank < baseline_rank for baseline_rank, holdout_rank in rank_pairs),
        "rank_equal": sum(holdout_rank == baseline_rank for baseline_rank, holdout_rank in rank_pairs),
        "rank_worsened": sum(holdout_rank > baseline_rank for baseline_rank, holdout_rank in rank_pairs),
    }
    rng = random.Random(seed)
    for k in (10, 50):
        paired_deltas = [
            float(holdout_rank <= k) - float(baseline_rank <= k)
            for baseline_rank, holdout_rank in rank_pairs
        ]
        observed = sum(paired_deltas) / len(paired_deltas) * 100.0
        bootstrap = []
        for _ in range(bootstrap_samples):
            sample = [paired_deltas[rng.randrange(len(paired_deltas))] for _ in paired_deltas]
            bootstrap.append(sum(sample) / len(sample) * 100.0)
        result[f"delta_R@{k}_percentage_points"] = observed
        result[f"delta_R@{k}_bootstrap_95ci"] = [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ]
    return result


def _surface_breakdown(
    category: str,
    caption_root: Path,
    baseline_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    records = load_records(caption_root / f"cap.{category}.lexical_val.json")
    if not (len(records) == len(baseline_rows) == len(holdout_rows)):
        raise ValueError(f"Surface breakdown row count mismatch for {category}.")

    result: dict[str, Any] = {}
    for surface, pattern in SURFACE_PATTERNS.items():
        indices = [
            index
            for index, record in enumerate(records)
            if any(pattern.search(caption) is not None for caption in captions_from_record(record))
        ]
        if not indices:
            continue
        baseline_subset = [baseline_rows[index] for index in indices]
        holdout_subset = [holdout_rows[index] for index in indices]
        paired = _paired_comparison(
            baseline_subset,
            holdout_subset,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        baseline_ranks = [int(row["target_rank"]) for row in baseline_subset]
        holdout_ranks = [int(row["target_rank"]) for row in holdout_subset]
        result[surface] = {
            "queries": len(indices),
            "baseline_R@10": sum(rank <= 10 for rank in baseline_ranks) / len(indices) * 100.0,
            "holdout_R@10": sum(rank <= 10 for rank in holdout_ranks) / len(indices) * 100.0,
            "baseline_R@50": sum(rank <= 50 for rank in baseline_ranks) / len(indices) * 100.0,
            "holdout_R@50": sum(rank <= 50 for rank in holdout_ranks) / len(indices) * 100.0,
            "paired": paired,
        }
    return result


def _format_markdown(results: dict[str, Any]) -> str:
    lines = [
        "| Category | Model | Validation | N | R@10 | R@50 | Median target rank | MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = [
        ("baseline_full", "Full train", "Full val"),
        ("holdout_full", "Lexical holdout", "Full val"),
        ("baseline_targeted", "Full train", "Lexical val"),
        ("holdout_targeted", "Lexical holdout", "Lexical val"),
    ]
    for category, result in results.items():
        for key, model, validation in labels:
            row = result[key]
            lines.append(
                f"| {category} | {model} | {validation} | {row['queries']} | "
                f"{row['R@10']:.4f} | {row['R@50']:.4f} | "
                f"{row['median_target_rank']:.1f} | {row['mrr']:.4f} |"
            )

    lines.extend(
        [
            "",
            "| Category | Δ targeted R@10 | 95% CI | Δ targeted R@50 | 95% CI | Rank improved/equal/worsened |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for category, result in results.items():
        paired = result["paired_targeted"]
        ci10 = paired["delta_R@10_bootstrap_95ci"]
        ci50 = paired["delta_R@50_bootstrap_95ci"]
        lines.append(
            f"| {category} | {paired['delta_R@10_percentage_points']:.4f} | "
            f"[{ci10[0]:.4f}, {ci10[1]:.4f}] | "
            f"{paired['delta_R@50_percentage_points']:.4f} | "
            f"[{ci50[0]:.4f}, {ci50[1]:.4f}] | "
            f"{paired['rank_improved']}/{paired['rank_equal']}/{paired['rank_worsened']} |"
        )
    lines.extend(
        [
            "",
            "| Category | Surface form | N | Full-train R@10 | Holdout R@10 | ΔR@10 | Full-train R@50 | Holdout R@50 | ΔR@50 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for category, result in results.items():
        for surface, row in result["surface_breakdown"].items():
            lines.append(
                f"| {category} | `{surface}` | {row['queries']} | "
                f"{row['baseline_R@10']:.4f} | {row['holdout_R@10']:.4f} | "
                f"{row['paired']['delta_R@10_percentage_points']:.4f} | "
                f"{row['baseline_R@50']:.4f} | {row['holdout_R@50']:.4f} | "
                f"{row['paired']['delta_R@50_percentage_points']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    holdout_root = Path(args.holdout_root)
    caption_root = Path(args.holdout_data_root) / "captions"
    results: dict[str, Any] = {}
    for category in args.categories:
        baseline_dir = baseline_root / category
        holdout_dir = holdout_root / category
        paths = {
            "baseline_full_metrics": baseline_dir / "evaluation.json",
            "baseline_full_rows": baseline_dir / "per_query.csv",
            "baseline_targeted_metrics": baseline_dir / "evaluation_lexical_val.json",
            "baseline_targeted_rows": baseline_dir / "per_query_lexical_val.csv",
            "holdout_full_metrics": holdout_dir / "evaluation_full_val.json",
            "holdout_full_rows": holdout_dir / "per_query_full_val.csv",
            "holdout_targeted_metrics": holdout_dir / "evaluation_lexical_val.json",
            "holdout_targeted_rows": holdout_dir / "per_query_lexical_val.csv",
        }
        missing = [path for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing evaluation artifacts:\n  " + "\n  ".join(map(str, missing)))

        category_result: dict[str, Any] = {}
        for prefix in ("baseline_full", "baseline_targeted", "holdout_full", "holdout_targeted"):
            metrics = _load_metrics(paths[f"{prefix}_metrics"], category)
            rows = _load_rows(paths[f"{prefix}_rows"])
            category_result[prefix] = {**metrics, **_rank_summary(rows)}
        baseline_targeted_rows = _load_rows(paths["baseline_targeted_rows"])
        holdout_targeted_rows = _load_rows(paths["holdout_targeted_rows"])
        category_result["paired_targeted"] = _paired_comparison(
            baseline_targeted_rows,
            holdout_targeted_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        category_result["surface_breakdown"] = _surface_breakdown(
            category,
            caption_root,
            baseline_targeted_rows,
            holdout_targeted_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        results[category] = category_result

    payload = {
        "baseline_root": str(baseline_root),
        "holdout_root": str(holdout_root),
        "bootstrap_samples": args.bootstrap_samples,
        "results": results,
    }
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _format_markdown(results)
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
