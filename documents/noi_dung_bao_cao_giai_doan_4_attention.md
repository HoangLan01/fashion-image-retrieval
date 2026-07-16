# Nội dung báo cáo — Global context qua attention visualization

## Thiết kế kiểm thử

Nhóm tái hiện cách trực quan hóa attention flow trong bài AACL. Với mỗi token, trọng số
`alpha` của ba additive-attention composition block được nhân trong log-space. Tensor thu được
có shape `[batch, block, head, token] = [2, 3, 8, 108]`, gồm 98 image token và 10 text token
sau padding. Padding token có attention bằng 0 và tổng `alpha` của mỗi head bằng 1.

Nhóm dùng checkpoint tốt nhất của category `shirt`, probe cố định `shirt_q01` với query ID
`B003JY6WY2`, và hai prompt gần với Figure 8 của bài báo:

1. `Make the shirt have longer sleeves.`
2. `Make the shirt have a different graphic.`

Probe được chọn trước khi xem heatmap, không thay bằng một ví dụ thuận mắt hơn sau khi có kết quả.
Mỗi Swin stage được xử lý riêng: 49 token đầu tạo Stage 3 map 7x7, 49 token tiếp theo tạo Stage 4
map 7x7; hai map được chuẩn hóa trước khi lấy trung bình.

![Attention flow dưới hai counterfactual prompt](../outputs/report_assets/fig_attention_counterfactual.png)

**Caption đề xuất:** Hình X. Attention flow của cùng ảnh nguồn `B003JY6WY2` dưới hai prompt
counterfactual. Stage 3 và Stage 4 được ánh xạ riêng về lưới 7x7; Average là trung bình của hai
stage sau chuẩn hóa. Hai prompt tạo top-5 không trùng nhau, nhưng vùng attention mạnh của cả hai
vẫn chủ yếu nằm trên logo ở ngực.

## Kết quả định lượng phụ trợ

| Chỉ báo so sánh hai prompt | Giá trị |
|---|---:|
| Cosine giữa hai query embedding | 0,1954 |
| Pearson giữa hai average-stage map | 0,8780 |
| Jensen-Shannon divergence giữa hai map | 0,0402 |
| Mean absolute map difference | 0,0754 |
| Top-5 overlap | 0/5 |
| Top-5 Jaccard | 0,0000 |

Attention mass trung bình qua tám head dịch dần về image token ở các block sau:

| Prompt | Block 1 image/text mass | Block 2 image/text mass | Block 3 image/text mass |
|---|---:|---:|---:|
| Longer sleeves | 0,5863 / 0,4137 | 0,7212 / 0,2788 | 0,9377 / 0,0623 |
| Different graphic | 0,5116 / 0,4884 | 0,6431 / 0,3569 | 0,9314 / 0,0686 |

Hai query embedding có cosine thấp và hai top-5 hoàn toàn khác nhau. Điều này cho thấy text đã
điều kiện hóa representation và ranking, thay vì bị mô hình bỏ qua hoàn toàn. Attention mass ở
block cuối chủ yếu thuộc về image token, phù hợp với mục tiêu tạo representation ảnh đã được sửa.

## Kiểm tra semantic alignment

Kết quả không đủ mạnh để khẳng định mô hình “hiểu” đúng vùng ngữ nghĩa như cách con người hiểu.
Cả prompt `longer sleeves` và `different graphic` đều tập trung mạnh vào logo giữa ngực ở Stage 3,
Stage 4 và average-stage. Prompt yêu cầu tay áo dài không làm attention dịch rõ sang hai tay áo.
Ở prompt thứ hai, một số head tăng attention ở cánh tay, góc dưới và background, nhưng đây chưa
phải localization rõ ràng của “different graphic”.

Text-token flow cũng không ưu tiên ổn định các từ khóa nội dung: `longer` và `graphic` nhận điểm
thấp nhất sau chuẩn hóa trong hai prompt tương ứng, trong khi các từ chức năng như `make` hoặc
`have` có thể nhận điểm cao. Do đó, màu token chỉ phản ánh attention nội bộ của checkpoint này,
không nên được diễn giải thành tầm quan trọng ngôn ngữ theo nghĩa nhân quả.

## Kết luận

Thí nghiệm cung cấp bằng chứng rằng AACL sử dụng text để thay đổi global representation và kết quả
retrieval: embedding thay đổi mạnh và top-5 không trùng nhau. Tuy nhiên, heatmap không cho thấy
semantic localization thuyết phục đối với thuộc tính `longer sleeves`. Vì vậy, kết luận phù hợp là
AACL có **text-conditioned global context ở mức hành vi**, nhưng visualization hiện tại **chưa
chứng minh mô hình hiểu global context đúng theo vùng ngữ nghĩa**.

Attention visualization chỉ là bằng chứng mô tả. Nếu cần kiểm chứng nhân quả, bước tiếp theo là
occlusion faithfulness: che vùng attention cao và thấp với cùng diện tích rồi so sánh độ giảm
similarity/rank. Nếu không chạy occlusion do giới hạn thời gian, báo cáo phải giữ nguyên giới hạn
kết luận trên.
