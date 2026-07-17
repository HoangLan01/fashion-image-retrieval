# Phân tích định tính hallucination/OOD

## Thiết kế ca minh họa

Nhóm sử dụng cùng probe `q01` đã được chọn trước retrieval cho từng category, thay vì tìm hậu nghiệm ca có kết quả xấu nhất. Cả ba probe nhận cùng một yêu cầu không thể đáp ứng trong gallery:

> Turn it into a transparent glass garment with animated flames and invisible fabric.

| Category | Probe | Query ID | MaxSim | Top-1/Top-2 margin |
|---|---|---|---:|---:|
| dress | dress_q01 | `B00ANK6ND0` | 0.3041 | 0.0157 |
| shirt | shirt_q01 | `B003JY6WY2` | 0.2787 | 0.0036 |
| toptee | toptee_q01 | `B008BT599E` | 0.2547 | 0.0032 |

![Ba ca thất bại định tính của AACL](fig_hallucination_cases.png)

**Nhận xét.** Trong cả ba category, hệ thống vẫn trả về năm ảnh thời trang có vẻ hợp lệ theo phân phối gallery, nhưng không ảnh nào đồng thời là trang phục thủy tinh trong suốt, có ngọn lửa động và vải vô hình. Kết quả cho thấy mô hình không có cơ chế phát hiện yêu cầu vô nghiệm hoặc từ chối trả lời; embedding văn bản vẫn bị ánh xạ đến các láng giềng gần nhất dù yêu cầu không được grounding trong gallery.

Cụ thể, checkpoint `dress` chủ yếu trả các váy hoa thông thường; checkpoint `shirt` trả cả tất, bao bì áo, ảnh chữ và kính; checkpoint `toptee` trả váy, áo hai dây, mũ và vali. Các kết quả ngoài loại trang phục mong đợi làm biểu hiện false grounding trực quan hơn, nhưng không được dùng để suy ra tần suất lỗi trên toàn bộ tập dữ liệu.

**Cách diễn đạt thận trọng.** Đây là bằng chứng định tính về sự tồn tại của false grounding/hallucination-like retrieval, không phải ước lượng tỷ lệ hallucination trên toàn bộ dữ liệu. Do không thực hiện chấm relevance đầy đủ, báo cáo không trình bày TextMatch@5, FullMatch@5, FAR hay Cohen's kappa cho thí nghiệm này.

**Caption đề xuất.** Hình X. Kết quả top-5 của ba checkpoint AACL trước cùng một prompt không thể thỏa mãn. Mỗi hàng dùng probe `q01` cố định của một category. Mô hình luôn trả về láng giềng trong gallery dù không kết quả nào đáp ứng đầy đủ yêu cầu, cho thấy hạn chế về phát hiện truy vấn vô nghiệm và khả năng abstention.
