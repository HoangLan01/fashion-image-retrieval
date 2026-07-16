from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CASES = (
    ("dress", "dress_q01"),
    ("shirt", "shirt_q01"),
    ("toptee", "toptee_q01"),
)
PROMPT = "Turn it into a transparent glass garment with animated flames and invisible fabric."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible qualitative hallucination figure from fixed q01 probes."
    )
    parser.add_argument("--input-root", default="outputs/hallucination")
    parser.add_argument(
        "--output", default="outputs/report_assets/fig_hallucination_cases.png"
    )
    parser.add_argument(
        "--analysis-output",
        default="outputs/report_assets/analysis_hallucination_qualitative.md",
    )
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    panels: list[tuple[str, str, Image.Image]] = []
    case_rows: list[dict[str, Any]] = []

    for category, probe_id in CASES:
        sheet_path = root / category / "contact_sheets" / f"{probe_id}.jpg"
        sheet = Image.open(sheet_path).convert("RGB")
        # Contact-sheet layout is fixed by run_hallucination_retrieval.py:
        # header 42 px followed by six rows of 204 px. Unsatisfiable is row 6.
        row_top = 42 + 5 * 204
        panel = sheet.crop((0, row_top, sheet.width, min(sheet.height, row_top + 204)))

        score_rows = _read_csv(root / category / "prompt_scores.csv")
        score = next(
            row
            for row in score_rows
            if row["probe_id"] == probe_id and row["scenario"] == "unsatisfiable"
        )
        result_rows = [
            row
            for row in _read_csv(root / category / "results.csv")
            if row["probe_id"] == probe_id and row["scenario"] == "unsatisfiable"
        ]
        result_rows.sort(key=lambda row: int(row["rank"]))
        panels.append((category, str(score["query_id"]), panel))
        case_rows.append(
            {
                "category": category,
                "probe_id": probe_id,
                "query_id": score["query_id"],
                "top5": ", ".join(row["image_id"] for row in result_rows),
                "max_similarity": float(score["max_similarity"]),
                "margin": float(score["top1_top2_margin"]),
            }
        )

    width = max(panel.width for _, _, panel in panels)
    title_height = 104
    label_height = 34
    gap = 12
    height = title_height + sum(label_height + panel.height + gap for _, _, panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        "Qualitative AACL failure cases for an unsatisfiable prompt",
        fill=(15, 15, 15),
        font=_font(24, bold=True),
    )
    draw.text((12, 45), PROMPT, fill=(25, 25, 25), font=_font(16))
    draw.text(
        (12, 72),
        "Fixed q01 probe from each category; SOURCE is followed by the model's top-5.",
        fill=(70, 70, 70),
        font=_font(14),
    )

    y = title_height
    for category, query_id, panel in panels:
        draw.rectangle((0, y, width, y + label_height), fill=(232, 238, 247))
        draw.text(
            (12, y + 7),
            f"{category.upper()} | query_id={query_id}",
            fill=(20, 65, 120),
            font=_font(16, bold=True),
        )
        y += label_height
        canvas.paste(panel, (0, y))
        y += panel.height + gap

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)

    analysis_lines = [
        "# Phân tích định tính hallucination/OOD",
        "",
        "## Thiết kế ca minh họa",
        "",
        (
            "Nhóm sử dụng cùng probe `q01` đã được chọn trước retrieval cho từng category, thay vì "
            "tìm hậu nghiệm ca có kết quả xấu nhất. Cả ba probe nhận cùng một yêu cầu không thể đáp "
            "ứng trong gallery:"
        ),
        "",
        f"> {PROMPT}",
        "",
        "| Category | Probe | Query ID | MaxSim | Top-1/Top-2 margin |",
        "|---|---|---|---:|---:|",
    ]
    for row in case_rows:
        analysis_lines.append(
            f"| {row['category']} | {row['probe_id']} | `{row['query_id']}` | "
            f"{row['max_similarity']:.4f} | {row['margin']:.4f} |"
        )
    analysis_lines.extend(
        [
            "",
            "![Ba ca thất bại định tính của AACL](fig_hallucination_cases.png)",
            "",
            (
                "**Nhận xét.** Trong cả ba category, hệ thống vẫn trả về năm ảnh thời trang có vẻ "
                "hợp lệ theo phân phối gallery, nhưng không ảnh nào đồng thời là trang phục thủy tinh "
                "trong suốt, có ngọn lửa động và vải vô hình. Kết quả cho thấy mô hình không có cơ chế "
                "phát hiện yêu cầu vô nghiệm hoặc từ chối trả lời; embedding văn bản vẫn bị ánh xạ đến "
                "các láng giềng gần nhất dù yêu cầu không được grounding trong gallery."
            ),
            "",
            (
                "Cụ thể, checkpoint `dress` chủ yếu trả các váy hoa thông thường; checkpoint `shirt` "
                "trả cả tất, bao bì áo, ảnh chữ và kính; checkpoint `toptee` trả váy, áo hai dây, mũ "
                "và vali. Các kết quả ngoài loại trang phục mong đợi làm biểu hiện false grounding "
                "trực quan hơn, nhưng không được dùng để suy ra tần suất lỗi trên toàn bộ tập dữ liệu."
            ),
            "",
            (
                "**Cách diễn đạt thận trọng.** Đây là bằng chứng định tính về sự tồn tại của false "
                "grounding/hallucination-like retrieval, không phải ước lượng tỷ lệ hallucination trên "
                "toàn bộ dữ liệu. Do không thực hiện chấm relevance đầy đủ, báo cáo không trình bày "
                "TextMatch@5, FullMatch@5, FAR hay Cohen's kappa cho thí nghiệm này."
            ),
            "",
            (
                "**Caption đề xuất.** Hình X. Kết quả top-5 của ba checkpoint AACL trước cùng một "
                "prompt không thể thỏa mãn. Mỗi hàng dùng probe `q01` cố định của một category. Mô hình "
                "luôn trả về láng giềng trong gallery dù không kết quả nào đáp ứng đầy đủ yêu cầu, cho "
                "thấy hạn chế về phát hiện truy vấn vô nghiệm và khả năng abstention."
            ),
        ]
    )
    analysis_output = Path(args.analysis_output)
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis_output.write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    print(f"Wrote qualitative case figure: {output}")
    print(f"Wrote qualitative analysis: {analysis_output}")


if __name__ == "__main__":
    main()
