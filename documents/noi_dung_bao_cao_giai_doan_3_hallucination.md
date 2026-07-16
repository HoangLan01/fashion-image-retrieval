# Nội dung báo cáo — Kiểm thử hallucination/OOD

## Thiết kế kiểm thử định tính

Để kiểm tra phản ứng của AACL trước yêu cầu không thể thỏa mãn, nhóm sử dụng cùng một prompt
cho ba checkpoint `dress`, `shirt` và `toptee`:

> Turn it into a transparent glass garment with animated flames and invisible fabric.

Nhóm lấy probe `q01` đã được chọn trước khi chạy retrieval của mỗi category. Cách chọn này tránh
việc tìm hậu nghiệm đúng ba trường hợp có kết quả xấu nhất. Một record FashionIQ sai category
(ảnh nguồn là váy nhưng target và caption là giày) đã bị loại trong bước kiểm tra dữ liệu trước
khi tạo hình; việc loại này được lưu trong `configs/hallucination_probe_exclusions.json`.

| Category | Probe | Query ID | MaxSim | Top-1/Top-2 margin |
|---|---|---|---:|---:|
| dress | dress_q01 | `B00ANK6ND0` | 0,3041 | 0,0157 |
| shirt | shirt_q01 | `B003JY6WY2` | 0,2787 | 0,0036 |
| toptee | toptee_q01 | `B008BT599E` | 0,2547 | 0,0032 |

![Ba ca thất bại định tính của AACL](../outputs/report_assets/fig_hallucination_cases.png)

**Caption đề xuất:** Hình X. Kết quả top-5 của ba checkpoint AACL trước cùng một prompt không
thể thỏa mãn. Mỗi hàng dùng probe `q01` cố định của một category. Mô hình luôn trả về láng giềng
trong gallery dù không kết quả nào đáp ứng đầy đủ yêu cầu, cho thấy hạn chế về phát hiện truy vấn
vô nghiệm và khả năng abstention.

## Nhận xét

Trong cả ba category, hệ thống vẫn trả về năm ảnh gần nhất mặc dù không ảnh nào đồng thời là
trang phục thủy tinh trong suốt, có ngọn lửa động và vải vô hình. Checkpoint `dress` chủ yếu trả
các váy hoa thông thường. Checkpoint `shirt` còn trả cả tất, bao bì áo, ảnh chữ và kính; checkpoint
`toptee` trả váy, áo hai dây, mũ và vali. Điều này cho thấy embedding của prompt vô nghiệm vẫn bị
ép ánh xạ đến các láng giềng trong gallery và mô hình không có cơ chế phát hiện yêu cầu không được
grounding hoặc từ chối trả kết quả.

Trong ngữ cảnh này, nhóm gọi hiện tượng trên là **false grounding/hallucination-like retrieval**.
Cách gọi này chính xác hơn việc khẳng định AACL “bịa” nội dung như một mô hình sinh, vì AACL là
hệ thống retrieval và bản thân kiến trúc luôn phải xếp hạng một gallery cố định.

## Giới hạn kết luận

Ba ca trên chứng minh sự tồn tại của hành vi thất bại đối với prompt vô nghiệm, nhưng không ước
lượng tần suất hallucination trên toàn bộ tập dữ liệu. Do giới hạn thời gian, nhóm không thực hiện
chấm relevance toàn bộ top-5 bởi hai người đánh giá. Vì vậy báo cáo không trình bày
`TextMatch@5`, `FullMatch@5`, `FAR` hoặc Cohen's kappa cho phần này và không thay các metric đó
bằng nhãn suy đoán.
