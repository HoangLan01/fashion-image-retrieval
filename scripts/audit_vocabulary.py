from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aacl_fashion.data.lexical_holdout import audit_records, load_records, vocabulary_audit_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit lexical holdout terms in FashionIQ captions.")
    parser.add_argument("--root", default="data/fashioniq", help="FashionIQ root directory.")
    parser.add_argument("--categories", nargs="+", default=["dress", "shirt", "toptee"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--json-output",
        default="outputs/vocabulary_audit/vocabulary_audit.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--markdown-output",
        default="outputs/report_assets/table_vocabulary_audit.md",
        help="Output Markdown table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for category in args.categories:
        for split in args.splits:
            caption_path = root / "captions" / f"cap.{category}.{split}.json"
            result = audit_records(load_records(caption_path))
            rows.append({"category": category, "split": split, "path": str(caption_path), **result})

    payload = {
        "root": str(root),
        "categories": args.categories,
        "splits": args.splits,
        "holdout_concept": "T-shirt surface forms: t-shirt, t shirt, tshirt, tee; singular/plural",
        "rows": rows,
    }
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(vocabulary_audit_markdown(rows), encoding="utf-8")

    print(vocabulary_audit_markdown(rows), end="")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
