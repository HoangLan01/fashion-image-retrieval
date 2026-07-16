from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aacl_fashion.data.lexical_holdout import (
    audit_records,
    filter_holdout_records,
    load_records,
    record_matches_holdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a FashionIQ train split with T-shirt surface forms held out."
    )
    parser.add_argument("--source-root", default="data/fashioniq")
    parser.add_argument("--output-root", default="data/fashioniq_lexical_holdout")
    parser.add_argument("--categories", nargs="+", default=["dress", "shirt", "toptee"])
    return parser.parse_args()


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing holdout data: {output} or {temporary}."
        )
    if not (source / "images").is_dir():
        raise FileNotFoundError(f"Missing source images directory: {source / 'images'}")

    manifest: dict[str, object] = {
        "source_root": str(source),
        "output_root": str(output),
        "policy": "Remove an entire train record if any caption contains an independent T-shirt/T shirt/tshirt/tee surface form, including plurals.",
        "categories": {},
    }
    try:
        temporary.mkdir(parents=True)
        image_target = os.path.relpath(source / "images", start=temporary)
        (temporary / "images").symlink_to(image_target, target_is_directory=True)
        shutil.copytree(source / "image_splits", temporary / "image_splits")
        (temporary / "captions").mkdir()

        for caption_file in sorted((source / "captions").glob("*.json")):
            shutil.copy2(caption_file, temporary / "captions" / caption_file.name)

        for category in args.categories:
            train_source = source / "captions" / f"cap.{category}.train.json"
            train_records = load_records(train_source)
            retained, removed = filter_holdout_records(train_records)
            train_output = temporary / "captions" / train_source.name
            _write_records(train_output, retained)

            val_source = source / "captions" / f"cap.{category}.val.json"
            val_records = load_records(val_source)
            targeted_val = [record for record in val_records if record_matches_holdout(record)]
            targeted_caption = temporary / "captions" / f"cap.{category}.lexical_val.json"
            _write_records(targeted_caption, targeted_val)

            val_gallery = source / "image_splits" / f"split.{category}.val.json"
            targeted_gallery = temporary / "image_splits" / f"split.{category}.lexical_val.json"
            shutil.copy2(val_gallery, targeted_gallery)

            output_audit = audit_records(retained)
            if output_audit["matching_records"] != 0:
                raise RuntimeError(f"Lexical leakage remains in filtered {category} train split.")
            manifest["categories"][category] = {
                "source_train_records": len(train_records),
                "retained_train_records": len(retained),
                "removed_train_records": len(removed),
                "targeted_validation_records": len(targeted_val),
                "filtered_train_sha256": _sha256(train_output),
                "filtered_train_audit": output_audit,
            }

        (temporary / "holdout_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Created lexical holdout dataset: {output}")


if __name__ == "__main__":
    main()
