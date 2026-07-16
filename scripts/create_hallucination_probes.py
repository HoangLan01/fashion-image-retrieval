from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aacl_fashion.config import load_config
from aacl_fashion.data.fashioniq_dataset import FashionIQDataset


CATEGORIES = ("dress", "shirt", "toptee")
SCENARIOS = (
    {
        "scenario": "in_domain",
        "expected_behavior": "Retrieve garments that satisfy the original FashionIQ modification.",
    },
    {
        "scenario": "paraphrase",
        "expected_behavior": "Remain close to the in-domain result under a meaning-preserving template change.",
    },
    {
        "scenario": "identity",
        "prompt": "Keep the garment exactly as it is; do not change its color, shape, material, or details.",
        "expected_behavior": "Preserve the source garment; returning a visually close neighbour is preferable.",
    },
    {
        "scenario": "contradiction",
        "prompt": "Make it both sleeveless and long-sleeved, and make it both shorter and longer.",
        "expected_behavior": "No gallery item can satisfy every mutually contradictory requirement.",
    },
    {
        "scenario": "ood",
        "prompt": "Make it play jazz music, increase its battery life, and make it smell like rain.",
        "expected_behavior": "The request is outside the fashion-image domain; high-confidence visual compliance is unsupported.",
    },
    {
        "scenario": "unsatisfiable",
        "prompt": "Turn it into a transparent glass garment with animated flames and invisible fabric.",
        "expected_behavior": "The fixed FashionIQ gallery is not expected to contain a full match.",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a fixed, non-cherry-picked FashionIQ hallucination/OOD probe set."
    )
    parser.add_argument("--config", default="configs/fashioniq_l40_shared.yaml")
    parser.add_argument("--output", default="outputs/hallucination/prompts.json")
    parser.add_argument("--review-output", default="outputs/hallucination/prompt_review.md")
    parser.add_argument(
        "--exclusions", default="configs/hallucination_probe_exclusions.json"
    )
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _paraphrase(captions: list[str]) -> str:
    clauses = [caption.strip().rstrip(". ") for caption in captions]
    return "Apply these requested changes: " + "; also, ".join(clauses) + "."


def _select_unique_queries(
    records: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    excluded_query_ids: set[str] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    excluded_query_ids = excluded_query_ids or set()
    indices = list(range(len(records)))
    rng.shuffle(indices)
    selected: list[tuple[int, dict[str, Any]]] = []
    seen_query_ids: set[str] = set()
    for index in indices:
        record = records[index]
        query_id = str(record["query_id"])
        if query_id in excluded_query_ids or query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        selected.append((index, record))
        if len(selected) == count:
            return selected
    raise ValueError(f"Only found {len(selected)} unique query images; requested {count}.")


def build_prompts(
    config: dict[str, Any],
    per_category: int,
    seed: int,
    exclusions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config["dataset"].get("name", "fashioniq") != "fashioniq":
        raise ValueError("Hallucination probes require the FashionIQ dataset.")

    root = config["dataset"]["root"]
    split = config["dataset"].get("val_split", "val")
    prompts: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    exclusion_records = list((exclusions or {}).get("records", []))

    for category_offset, category in enumerate(CATEGORIES):
        dataset = FashionIQDataset(root=root, category=category, split=split, transform=None)
        # Separate deterministic stream per category prevents changes in one category
        # from changing the selected probes in another.
        rng = random.Random(seed + 10_000 * category_offset)
        excluded_query_ids = {
            str(item["query_id"])
            for item in exclusion_records
            if str(item["category"]) == category
        }
        selected = _select_unique_queries(
            dataset.records, per_category, rng, excluded_query_ids=excluded_query_ids
        )
        for local_index, (query_index, record) in enumerate(selected, start=1):
            probe_id = f"{category}_q{local_index:02d}"
            original_caption = " [SEP] ".join(record["captions"])
            probes.append(
                {
                    "probe_id": probe_id,
                    "category": category,
                    "query_index": query_index,
                    "query_id": record["query_id"],
                    "target_id": record["target_id"],
                    "query_image_path": str(dataset._resolve_image(record["query_id"])),
                    "original_caption": original_caption,
                    "paraphrase": _paraphrase(record["captions"]),
                }
            )
            for scenario_index, scenario_spec in enumerate(SCENARIOS, start=1):
                scenario = str(scenario_spec["scenario"])
                if scenario == "in_domain":
                    prompt = original_caption
                elif scenario == "paraphrase":
                    prompt = _paraphrase(record["captions"])
                else:
                    prompt = str(scenario_spec["prompt"])
                prompts.append(
                    {
                        "prompt_id": f"{probe_id}_s{scenario_index:02d}",
                        "probe_id": probe_id,
                        "scenario": scenario,
                        "category": category,
                        "query_index": query_index,
                        "query_id": record["query_id"],
                        "target_id": record["target_id"],
                        "source_captions": list(record["captions"]),
                        "original_caption": original_caption,
                        "prompt": prompt,
                        "expected_behavior": scenario_spec["expected_behavior"],
                        "selection_seed": seed,
                    }
                )
                category_counts[category] += 1

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config["_meta"]["config_path"],
        "dataset_root": str(root),
        "split": split,
        "seed": seed,
        "selection_rule": (
            "For each category, shuffle validation record indices with a fixed category-specific "
            "random stream and take the first distinct, non-excluded query IDs. Pre-rating exclusions "
            "are limited to documented annotation/category errors. Selection never uses retrieval scores."
        ),
        "exclusions": exclusions or {"records": []},
        "per_category": per_category,
        "num_probe_images": per_category * len(CATEGORIES),
        "num_scenarios": len(SCENARIOS),
        "num_prompts": len(prompts),
        "category_prompt_counts": dict(category_counts),
        "scenarios": [item["scenario"] for item in SCENARIOS],
        "probes": probes,
        "prompts": prompts,
    }


def _write_review(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Review bộ probe hallucination/OOD",
        "",
        (
            f"Bộ probe có {payload['num_probe_images']} ảnh, được chọn trước khi retrieval bằng seed "
            f"{payload['seed']}. Không bỏ ảnh chỉ vì caption hoặc retrieval không đẹp; nếu annotation "
            "gốc nhiễu, đánh dấu `revise` và ghi lý do trước khi chạy model."
        ),
        "",
        "Quyết định chung: `[ ] accept nguyên bộ` / `[ ] revise manifest và ghi lại quy tắc`.",
        "",
        "| Category | Probe | Query image | Original caption | Review |",
        "|---|---|---|---|---|",
    ]
    review_parent = path.parent.resolve()
    for probe in payload["probes"]:
        image_path = Path(probe["query_image_path"]).resolve()
        display_path = Path(os.path.relpath(image_path, review_parent))
        caption = str(probe["original_caption"]).replace("|", "\\|")
        lines.append(
            f"| {probe['category']} | {probe['probe_id']} | "
            f"<img src=\"{display_path}\" width=\"110\"> | {caption} | `[ ] accept` / `[ ] revise` |"
        )
    lines.extend(
        [
            "",
            "## Ghi chú review",
            "",
            "- `<PLACEHOLDER_REVIEWER>`:",
            "- `<PLACEHOLDER_DATE>`:",
            "- `<PLACEHOLDER_AMBIGUOUS_PROMPTS_AND_DECISION>`:",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    exclusion_path = Path(args.exclusions)
    exclusions = (
        json.loads(exclusion_path.read_text(encoding="utf-8"))
        if exclusion_path.exists()
        else {"records": []}
    )
    payload = build_prompts(
        config, per_category=args.per_category, seed=args.seed, exclusions=exclusions
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review_output = Path(args.review_output)
    _write_review(review_output, payload)
    print(f"Wrote {payload['num_prompts']} prompts for {payload['num_probe_images']} probe images: {output}")
    print(f"Wrote prompt review sheet: {review_output}")
    print(payload["selection_rule"])


if __name__ == "__main__":
    main()
