# Hướng dẫn Giai đoạn 6 — Context intervention

## 1. Mục tiêu

Giai đoạn này kiểm tra checkpoint đã huấn luyện có phụ thuộc vào global context đúng của từng
query hay không. Đây là inference-only intervention trên toàn bộ validation set `shirt`; không
cần train checkpoint mới.

Ba chế độ dùng cùng checkpoint, query order và gallery:

- `full`: AACL không thay đổi;
- `shuffled`: tại mỗi composition block, context vector `c` được dịch vòng sang mẫu kế tiếp trong
  fixed evaluation batch; ảnh và caption của query không bị tráo;
- `uniform`: learned attention `alpha` được thay bằng trọng số đều trên mọi image/text token hợp
  lệ, còn toàn bộ trọng số checkpoint được giữ nguyên.

`shuffled` kiểm tra context có phụ thuộc đúng query hay không. `uniform` kiểm tra learned token
weighting có tạo khác biệt so với phép lấy trung bình đều hay không. Vì model không được retrain,
hai chế độ này không được gọi là so sánh công bằng giữa các kiến trúc đã huấn luyện độc lập.

## 2. Metric

Script xuất:

- R@10 và R@50 trên cùng FashionIQ validation set;
- ΔRecall so với full AACL;
- MRR và median target rank;
- paired target-rank change với bootstrap 95% CI;
- cosine giữa intervention embedding và full embedding;
- top-5 overlap và tỷ lệ top-1 không đổi so với full AACL.

Không báo `TextMatch@5`, `Preservation@5` hoặc `FullMatch@5` vì các metric này cần rubric và chấm
tay. Tự suy ra chúng từ FashionIQ target ID sẽ tạo số liệu không hợp lệ.

## 3. Lệnh cần chạy

Kiểm tra GPU trống bằng `nvtop`, không dừng tiến trình của user khác. Job chỉ inference và dùng
evaluation batch 16, phù hợp phần VRAM còn lại sau các job trước.

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=<GPU_TRONG> python scripts/run_context_intervention.py \
  --category shirt \
  --device cuda
```

Chỉ cần chạy `shirt` trước vì đây là category đã dùng cho attention và occlusion. Không chạy cả
ba category cho đến khi Codex đọc kết quả `shirt` và xác định có cần mở rộng hay không.

## 4. Output mong đợi

```text
outputs/context_intervention/shirt/seed42/
  summary.json
  per_query.csv

outputs/report_assets/
  table_global_context_metrics.md
  fig_context_intervention.png
```

Trong log cần có bốn lượt encode: gallery, query full, query shuffled và query uniform. Baseline
R@10/R@50 phải gần kết quả evaluate checkpoint shirt đã có; nếu khác đáng kể thì dừng trước khi
diễn giải intervention.

## 5. Điểm dừng và capture

Đã kiểm tra xong:

- Full baseline khớp chính xác evaluation trước;
- CSV có 6.114 dòng dữ liệu, bằng `3 × 2.038` record;
- mọi rank/cosine/top-5 value đều hữu hạn và mỗi variant có đủ record;
- paired rank CI của shuffled và uniform đều cách xa 0;
- hình hiển thị đúng Recall và không dùng trục gây hiểu nhầm.

Hình cần capture/chèn: `outputs/report_assets/fig_context_intervention.png`.

Bảng số liệu: `outputs/report_assets/table_global_context_metrics.md`.

Nội dung và caption hoàn chỉnh:
`documents/noi_dung_bao_cao_giai_doan_6_context_intervention.md`.
