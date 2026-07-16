# Hướng dẫn Giai đoạn 4 — Attention extraction và visualization

## 1. Mục tiêu

Giai đoạn này tái hiện phép trực quan hóa trong bài AACL: lấy trọng số `alpha` của từng token ở
ba composition block, nhân qua các block trong log-space, sau đó ánh xạ 98 image token thành hai
heatmap 7x7 riêng biệt:

- 49 token đầu: Swin Stage 3;
- 49 token tiếp theo: Swin Stage 4;
- average-stage: chuẩn hóa từng stage rồi lấy trung bình.

Tensor attention gốc có dạng `[batch, block, head, token]`. Với checkpoint hiện tại và hai prompt,
shape kỳ vọng là `[2, 3, 8, 98 + số text token]`.

## 2. Ca counterfactual cố định

Thí nghiệm chính dùng `shirt_q01`, được chọn trước retrieval bằng manifest seed 42. Cùng một ảnh
nguồn nhận hai prompt gần với Figure 8 của bài báo:

1. `Make the shirt have longer sleeves.`
2. `Make the shirt have a different graphic.`

Thiết kế này kiểm tra xem thay đổi duy nhất ở text có làm attention map và top-5 thay đổi hay không.

## 3. Lệnh chạy

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=<GPU_TRONG> python scripts/visualize_attention.py \
  --category shirt \
  --probe-id shirt_q01 \
  --device cuda \
  --save-heads
```

Đây là inference `model.eval()` và không tính gradient. Gallery chỉ được encode một lần; mức dùng
VRAM dự kiến thấp hơn training. Không cần train lại checkpoint.

## 4. Output

Raw và hình chi tiết nằm tại:

```text
outputs/attention/shirt/shirt_q01/
  attention_raw.pt
  attention_raw.npz
  metadata.json
  source_model_input.png
  prompt_01_stage3.png
  prompt_01_stage4.png
  prompt_01_average.png
  prompt_01_stage3_heads.png
  prompt_01_stage4_heads.png
  prompt_02_stage3.png
  prompt_02_stage4.png
  prompt_02_average.png
  prompt_02_stage3_heads.png
  prompt_02_stage4_heads.png
```

Hình ghép dành cho báo cáo:

```text
outputs/report_assets/fig_attention_counterfactual.png
```

`metadata.json` lưu Pearson correlation, Jensen-Shannon divergence giữa hai average-stage map,
top-5 overlap, target rank và text-token flow. Overlay dùng chính tensor 224x224 sau resize và
center crop của model, vì vậy không bị lệch do đặt heatmap lên ảnh gốc chưa crop.

## 5. Cách diễn giải

- Heatmap khác nhau và ranking thay đổi: bằng chứng mô hình thực hiện text-conditioned selection.
- Heatmap khác nhưng ranking giữ nguyên: attention có điều kiện theo text nhưng ảnh hưởng retrieval
  chưa rõ.
- Heatmap gần như giống nhau: không nên tuyên bố mô hình hiểu global context chỉ dựa trên hình.
- Attention visualization là mô tả cơ chế, chưa phải bằng chứng nhân quả. Giai đoạn 5 occlusion mới
  kiểm tra vùng attention cao có thực sự quan trọng hơn vùng attention thấp hay không.

## 6. Điểm capture

Đã kiểm tra output thực: raw tensor hợp lệ, orientation đúng, top-5 overlap bằng 0 nhưng hai map
vẫn chủ yếu tập trung vào logo ngực. Hình chính cần chèn là
`outputs/report_assets/fig_attention_counterfactual.png`. Không cần chụp lại từ terminal; dùng
trực tiếp PNG để giữ nguyên độ phân giải.
