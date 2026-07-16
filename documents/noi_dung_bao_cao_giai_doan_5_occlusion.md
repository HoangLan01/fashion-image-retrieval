# Nội dung báo cáo — Occlusion faithfulness

## Thiết kế kiểm thử

Attention heatmap chỉ mô tả nơi mô hình đặt trọng số, chưa cho biết vùng đó có thực sự ảnh hưởng
đến quyết định retrieval hay không. Vì vậy, nhóm thực hiện kiểm thử occlusion trên checkpoint tốt
nhất của category `shirt` tại epoch 45. Thí nghiệm dùng 10 probe `shirt_q01…shirt_q10` đã được
cố định trước khi xem kết quả và caption FashionIQ in-domain gốc của từng probe.

Average-stage attention map 7x7 được dùng để chọn hai mask có cùng diện tích:

- `high`: các patch có attention flow cao nhất;
- `low`: các patch có attention flow thấp nhất.

Ba tỷ lệ che 10%, 20% và 30% tương ứng với 5, 10 và 15 patch trên lưới 7x7. Hai mask trong mỗi
cặp không chồng nhau. Pixel bị che được đặt về ImageNet mean, tức giá trị 0 sau normalization.
Đây là phép kiểm thử inference-only; checkpoint không được huấn luyện lại.

Primary endpoint dùng ảnh top-1 của truy vấn không che làm reference cố định. Sau khi che ảnh
nguồn, nhóm đo độ giảm cosine similarity và mức tăng rank của chính reference này. FashionIQ
ground-truth target được giữ làm secondary endpoint. Với từng tỷ lệ, hiệu ứng `high−low` được
tính theo cặp trên cùng probe, kèm bootstrap 95% confidence interval từ 5.000 resample và exact
two-sided sign-flip p-value. Mọi phép lấy mẫu dùng seed 42.

![So sánh occlusion vùng attention cao và thấp](../outputs/report_assets/fig_occlusion_comparison.png)

**Caption đề xuất:** Hình X. Occlusion faithfulness trên 10 probe `shirt` cố định. Hàng trên minh
họa ảnh nguồn và hai mask high/low có cùng diện tích 20%; màu xám là ImageNet mean. Hai biểu đồ
cho thấy mức giảm similarity và tăng rank của top-1 reference khi che vùng attention cao so với
vùng attention thấp.

## Kết quả primary endpoint

| Tỷ lệ che | Δsim high | Δsim low | high−low [bootstrap 95% CI] | Δrank high | Δrank low | p (Δsim) |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0,0015 | −0,0018 | 0,0033 [0,0006; 0,0057] | 0,30 | 0,00 | 0,0391 |
| 20% | 0,0041 | −0,0006 | 0,0047 [−0,0004; 0,0105] | 0,70 | 0,00 | 0,1289 |
| 30% | 0,0053 | −0,0000 | 0,0053 [−0,0025; 0,0134] | 1,00 | 0,20 | 0,2520 |

Giá trị Δsim dương nghĩa là similarity với top-1 reference bị giảm sau khi che; giá trị âm nghĩa
là similarity tăng nhẹ. Ở cả ba tỷ lệ, che vùng high-attention làm giảm similarity và tăng rank
nhiều hơn trung bình so với vùng low-attention. Hiệu ứng rõ nhất ở tỷ lệ 10%: chênh lệch Δsim là
0,0033, bootstrap CI không cắt 0 và sign-flip `p=0,0391`. Có 9/10 probe cho hiệu ứng đúng chiều
ở tỷ lệ này. Tại 20% và 30%, hiệu ứng trung bình vẫn đúng chiều nhưng CI cắt 0 và `p>0,05`; do đó
không thể kết luận hiệu ứng ổn định khi vùng che được mở rộng.

## Độ thay đổi của query embedding

| Tỷ lệ che | Cosine-drop high | Cosine-drop low | high−low [bootstrap 95% CI] | p |
|---:|---:|---:|---:|---:|
| 10% | 0,0112 | 0,0012 | 0,0100 [0,0023; 0,0202] | 0,0078 |
| 20% | 0,0171 | 0,0023 | 0,0148 [0,0026; 0,0281] | 0,0195 |
| 30% | 0,0247 | 0,0035 | 0,0212 [0,0078; 0,0363] | 0,0020 |

Che vùng high-attention làm query embedding lệch khỏi embedding gốc nhiều hơn vùng low-attention
ở cả ba tỷ lệ; cả ba CI đều không cắt 0. Đây là bằng chứng nhất quán nhất rằng attention map đã
đánh dấu các vùng quan trọng đối với representation nội bộ của mô hình.

## Failure case và secondary endpoint

Primary endpoint không đúng chiều trên mọi probe. Ở tỷ lệ 10%, `shirt_q04` có chênh lệch Δsim
`high−low = −0,0059`, dù cosine-drop của query embedding vẫn lớn hơn khi che high-attention
(`high−low = 0,0118`). Ảnh này là áo đen có chữ trắng ở ngực; kết quả cho thấy thay đổi embedding
không nhất thiết làm similarity của một reference cụ thể giảm theo quan hệ tuyến tính.

Kết quả trên FashionIQ ground-truth target cũng không ổn định. Chênh lệch high−low của target
similarity lần lượt là 0,0006, −0,0006 và −0,0036 ở ba tỷ lệ; tất cả bootstrap CI đều cắt 0 và
sign-flip `p>0,6`. Target-rank có phương sai lớn và cũng không cho hiệu ứng có ý nghĩa. Vì vậy,
occlusion chưa chứng minh các vùng attention cao là nguyên nhân trực tiếp giúp mô hình tiến gần
đúng FashionIQ target.

## Kết luận

Occlusion cung cấp **bằng chứng faithfulness một phần**. Vùng high-attention quan trọng hơn rõ
rệt đối với query representation, và ở mask 10% cũng ảnh hưởng có ý nghĩa đến top-1 retrieval
decision hiện tại. Tuy nhiên, bằng chứng yếu đi ở mask lớn hơn, có probe đảo chiều, và secondary
endpoint trên ground-truth target không ủng hộ một hiệu ứng ổn định.

Kết hợp với attention visualization ở giai đoạn trước, kết luận phù hợp là AACL có sử dụng một số
vùng global image context để hình thành representation và ranking. Các kết quả này không đủ để
khẳng định mô hình “hiểu” đúng global context theo nghĩa ngữ nghĩa của con người, cũng không cho
phép xem mọi attention heatmap là một explanation nhân quả hoàn chỉnh.
