# Hướng dẫn Giai đoạn 5 — Occlusion faithfulness

## 1. Thiết kế

Thí nghiệm dùng 10 probe `shirt_q01…shirt_q10` đã cố định trước khi xem attention. Mỗi probe dùng
caption FashionIQ in-domain gốc. Average-stage attention map 7x7 được dùng để chọn patch:

- `high`: các patch có attention flow cao nhất;
- `low`: các patch có attention flow thấp nhất;
- tỷ lệ: 10%, 20%, 30%, tương ứng cùng số patch cho high và low;
- pixel bị che được đặt về ImageNet mean, tức giá trị `0` sau normalization.

Primary endpoint cố định ảnh top-1 của truy vấn không che làm reference, sau đó đo similarity drop
và rank increase của chính ảnh đó dưới occlusion. FashionIQ target similarity/rank được lưu như
secondary endpoint. Cách này kiểm tra attention có hỗ trợ quyết định hiện tại của model hay không,
đồng thời không phụ thuộc hoàn toàn vào target rank vốn có thể rất thấp.

## 2. Thống kê

Với từng probe và mask ratio, script tính chênh lệch ghép cặp `high - low`. Báo cáo gồm:

- mean high và mean low;
- paired high-minus-low;
- bootstrap 95% confidence interval, 5.000 resample, seed 42;
- exact two-sided sign-flip p-value trên 10 probe.

Nếu high-attention masking làm similarity giảm và rank tăng nhiều hơn low-attention masking, với
CI high-minus-low chủ yếu lớn hơn 0, attention map có bằng chứng faithfulness tốt hơn. Nếu CI cắt
0 hoặc hiệu ứng đảo chiều, không nên diễn giải heatmap như explanation đáng tin cậy.

## 3. Lệnh chạy

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=<GPU_TRONG> python scripts/run_occlusion_faithfulness.py \
  --category shirt \
  --num-probes 10 \
  --device cuda
```

Đây là inference-only: attention được trích một lần cho 10 ảnh, gallery được encode một lần, sau
đó 60 query occluded được chạy theo batch 12. Không cần train checkpoint mới.

## 4. Output

```text
outputs/occlusion/shirt/shirt_probe10/
  results.csv
  summary.json
  occlusion_masks_and_attention.npz
  preview_source.png
  preview_high_mask.png
  preview_low_mask.png

outputs/report_assets/
  fig_occlusion_comparison.png
  table_occlusion_faithfulness.md
```

## 5. Điểm capture

Đã kiểm tra xong:

- đủ 60 kết quả ghép cặp cho 10 probe, ba tỷ lệ và hai loại mask;
- mask 10/20/30% có đúng 5/10/15 patch; high và low bằng diện tích, không chồng nhau;
- ảnh preview dùng ImageNet-mean fill đúng vị trí và không lỗi normalization;
- biểu đồ hiển thị đúng cả Δsim âm và dương;
- kết luận được giới hạn ở faithfulness một phần vì CI của primary endpoint cắt 0 tại 20%/30%.

Hình cần capture/chèn vào báo cáo:
`outputs/report_assets/fig_occlusion_comparison.png`.

Bảng số liệu có thể copy trực tiếp từ:
`outputs/report_assets/table_occlusion_faithfulness.md`.

Nội dung và caption hoàn chỉnh nằm tại:
`documents/noi_dung_bao_cao_giai_doan_5_occlusion.md`.
