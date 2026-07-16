# Nội dung báo cáo — Global-context intervention

## Thiết kế kiểm thử

Để kiểm tra checkpoint có thực sự phụ thuộc vào global context đúng của từng query hay không, nhóm
thực hiện hai intervention tại vector context `c` trong từng additive-attention composition block.
Thí nghiệm dùng checkpoint tốt nhất của category `shirt` tại epoch 45, toàn bộ 2.038 query-caption
record validation (1.541 source image ID phân biệt), cùng candidate gallery và evaluation batch 16.

Ba chế độ được so sánh:

- **Full AACL:** inference không thay đổi;
- **Shuffled context:** dịch vòng context vector `c` một vị trí giữa các mẫu trong từng fixed
  evaluation batch; ảnh và caption của query vẫn giữ nguyên;
- **Uniform context:** thay learned attention `alpha` bằng trọng số đều trên tất cả image/text token
  hợp lệ, trong khi giữ nguyên toàn bộ trọng số checkpoint.

Gallery chỉ được encode một lần. Mỗi chế độ sau đó encode cùng query order và được đánh giá bằng
R@10, R@50, target rank, cosine với Full embedding và top-5 overlap. Với target-rank change, nhóm
dùng paired bootstrap 95% CI với 5.000 resample và paired Monte Carlo sign-flip test 5.000 lần,
seed 42. Đây là inference-only intervention, không phải so sánh các kiến trúc đã retrain độc lập.

![Kết quả global-context intervention](../outputs/report_assets/fig_context_intervention.png)

**Caption đề xuất:** Hình X. Recall trên 2.038 validation query-caption record của category
`shirt` khi giữ nguyên AACL, tráo context vector giữa các mẫu, hoặc thay learned attention bằng
uniform context. Cả hai intervention đều dùng cùng checkpoint và gallery với Full AACL.

## Kết quả

| Variant | R@10 | R@50 | ΔR@10 | ΔR@50 | Median target rank | Cosine→Full | Top-5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full AACL | 16,8302 | 40,7753 | 0,0000 | 0,0000 | 86,0 | 1,0000 | 5,00/5 |
| Shuffled context | 1,2758 | 3,4347 | −15,5545 | −37,3405 | 2.080,0 | 0,2223 | 0,09/5 |
| Uniform context | 4,3670 | 13,0520 | −12,4632 | −27,7233 | 951,5 | 0,4819 | 0,35/5 |

Full AACL tái lập chính xác evaluation trước đó: R@10 = 16,8302 và R@50 = 40,7753. Vì vậy,
chênh lệch không đến từ checkpoint, gallery hoặc preprocessing khác nhau.

Shuffled context làm R@10 giảm 15,55 điểm và R@50 giảm 37,34 điểm. Median target rank tăng từ 86
lên 2.080; mean paired rank increase là 1.885,79 với bootstrap 95% CI
`[1.797,97; 1.972,41]` và Monte Carlo sign-flip `p≈0,0002`. Target rank xấu đi trên 1.828/2.038
record (89,7%). Mean embedding cosine với Full chỉ còn 0,2223, top-5 overlap còn 0,09/5 và top-1
không đổi ở 1,03% record. Kết quả cho thấy context của một query không thể được thay bằng context
của query khác mà vẫn giữ hành vi retrieval.

Uniform context tốt hơn shuffled nhưng vẫn giảm mạnh: R@10 giảm 12,46 điểm, R@50 giảm 27,72
điểm và median target rank tăng lên 951,5. Mean paired rank increase là 1.202,71 với bootstrap
95% CI `[1.126,42; 1.279,82]`, `p≈0,0002`; 1.708/2.038 record (83,8%) có rank xấu đi. Embedding
cosine còn 0,4819 và top-5 overlap còn 0,35/5. Do đó, learned token weighting mang thông tin hữu
ích hơn phép gán trọng số đều trên cùng tập token.

## Diễn giải và giới hạn

Kết quả cung cấp bằng chứng mạnh về **query-specific context dependence**: representation và
retrieval performance phụ thuộc vào context được tính từ đúng cặp ảnh–caption, đồng thời phụ thuộc
vào learned attention distribution thay vì chỉ trung bình đều token. Hiệu ứng xuất hiện trên phần
lớn validation record và paired CI cách xa 0, nên không bị chi phối bởi một vài outlier.

Tuy nhiên, shuffled context tạo một trạng thái ngoài phân phối huấn luyện và kết quả phụ thuộc vào
fixed batch/permutation. Uniform context cũng không được retrain, vì vậy thí nghiệm không chứng
minh kiến trúc learned attention ưu việt hơn mọi kiến trúc uniform-context được tối ưu từ đầu.
Ngoài ra, Recall giảm chỉ chứng minh context được sử dụng hữu ích, không cho biết mô hình gắn đúng
từng từ với đúng vùng ảnh theo cách con người diễn giải.

## Kết luận

Kết hợp ba nguồn bằng chứng cho thấy: counterfactual text làm embedding/ranking thay đổi; occlusion
cho faithfulness một phần đối với vùng high-attention; và context intervention làm Recall suy giảm
rất mạnh khi context bị tráo hoặc làm đều. Vì vậy, có thể kết luận AACL học và sử dụng
**global context phụ thuộc query** trong cơ chế retrieval. Cách viết phù hợp không phải “AACL đã
hiểu hoàn toàn global context”, mà là “AACL sử dụng global context có ích về mặt hành vi; semantic
localization và explanation nhân quả vẫn còn giới hạn”.
