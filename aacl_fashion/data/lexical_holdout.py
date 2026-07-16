from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SURFACE_PATTERNS: dict[str, re.Pattern[str]] = {
    "t-shirt": re.compile(r"(?<![a-z0-9])t-shirts?(?![a-z0-9])", re.IGNORECASE),
    "t shirt": re.compile(r"(?<![a-z0-9])t\s+shirts?(?![a-z0-9])", re.IGNORECASE),
    "tshirt": re.compile(r"(?<![a-z0-9])tshirts?(?![a-z0-9])", re.IGNORECASE),
    "tee": re.compile(r"(?<![a-z0-9])tees?(?![a-z0-9])", re.IGNORECASE),
}

HOLDOUT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:t(?:-|\s*)shirts?|tees?)(?![a-z0-9])",
    re.IGNORECASE,
)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def captions_from_record(record: dict[str, Any]) -> list[str]:
    captions = record.get("captions") or record.get("caption") or record.get("text")
    if captions is None:
        raise ValueError("Caption record is missing captions/caption/text.")
    if isinstance(captions, str):
        return [captions]
    return [str(caption) for caption in captions]


def record_matches_holdout(record: dict[str, Any]) -> bool:
    return any(HOLDOUT_PATTERN.search(caption) is not None for caption in captions_from_record(record))


def filter_holdout_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in records:
        (removed if record_matches_holdout(record) else retained).append(record)
    return retained, removed


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    captions = [caption for record in records for caption in captions_from_record(record)]
    joined_records = ["\n".join(captions_from_record(record)) for record in records]
    return {
        "records": len(records),
        "captions": len(captions),
        "matching_records": sum(HOLDOUT_PATTERN.search(text) is not None for text in joined_records),
        "matching_captions": sum(HOLDOUT_PATTERN.search(caption) is not None for caption in captions),
        "surface_occurrences": {
            name: sum(len(pattern.findall(caption)) for caption in captions)
            for name, pattern in SURFACE_PATTERNS.items()
        },
    }


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def vocabulary_audit_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Category | Split | Records | Matching records | `t-shirt` | `t shirt` | `tshirt` | `tee` |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        counts = row["surface_occurrences"]
        lines.append(
            "| {category} | {split} | {records} | {matching_records} | {t_hyphen} | "
            "{t_space} | {tshirt} | {tee} |".format(
                category=row["category"],
                split=row["split"],
                records=row["records"],
                matching_records=row["matching_records"],
                t_hyphen=counts["t-shirt"],
                t_space=counts["t shirt"],
                tshirt=counts["tshirt"],
                tee=counts["tee"],
            )
        )
    lines.extend(
        [
            "",
            "Ghi chú: pattern không phân biệt hoa/thường, chấp nhận dạng số nhiều và chỉ khớp "
            "các token/cụm từ độc lập. `Matching records` là số cặp query-target có ít nhất một "
            "caption chứa một dạng thuộc khái niệm holdout.",
            "",
        ]
    )
    return "\n".join(lines)
