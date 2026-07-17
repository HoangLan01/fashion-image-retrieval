## Phân tích thí nghiệm lexical holdout

Thí nghiệm loại khỏi tập fine-tuning mọi record có ít nhất một caption chứa các surface form
`t-shirt`, `t shirt`, `tshirt`, `tee` hoặc dạng số nhiều tương ứng. Tập validation chuẩn được
giữ nguyên; đồng thời, nhóm tạo lexical validation gồm 167 query shirt và 94 query toptee có
chứa ít nhất một surface form bị giữ lại.

Trên validation đầy đủ, mô hình lexical-holdout không suy giảm nhất quán. R@10 tăng 1,4230
điểm phần trăm ở shirt và 1,4278 điểm ở toptee, trong khi R@50 giảm lần lượt 0,9323 và 0,6119
điểm. Điều này cho thấy việc giảm số mẫu train không làm hỏng chất lượng retrieval tổng quát.

Trên lexical validation, hiệu năng giảm rõ hơn. Với shirt, R@10 giảm từ 10,1796 xuống 7,1856
và R@50 giảm từ 31,1377 xuống 19,7605. Paired bootstrap 95% confidence interval của thay đổi
R@50 là [−17,9641; −4,7904], không chứa 0. Đồng thời, target rank xấu đi ở 104/167 query,
cải thiện ở 61 query và giữ nguyên ở 2 query. Đây là bằng chứng mạnh nhất rằng việc không quan
sát các surface form trong fine-tuning làm suy giảm grounding theo nhiệm vụ đối với nhóm shirt.

Với toptee, R@10 giảm từ 13,8298 xuống 12,7660 và R@50 giảm từ 44,6809 xuống 37,2340. Tuy
nhiên, khoảng tin cậy của cả hai thay đổi đều chứa 0; chỉ có 94 query nên chưa đủ cơ sở để khẳng
định mức suy giảm ổn định về thống kê. Target rank cải thiện ở 42 query, giữ nguyên ở 5 và xấu
đi ở 47 query. MRR tăng nhẹ dù median rank và Recall giảm, cho thấy một số ít query được đẩy lên
vị trí rất cao trong khi nhiều query khác bị đẩy xuống khỏi top-50.

Breakdown theo cách viết cho thấy `T shirt` ở shirt bị ảnh hưởng mạnh nhất trong các nhóm có
cỡ mẫu tương đối: R@50 giảm 16,0714 điểm. `tee` ở toptee giảm 12,2449 điểm R@50. Một số nhóm
nhỏ như `tshirt` toptee chỉ có 8 query, nên không nên diễn giải riêng như bằng chứng chắc chắn.

Kết quả không có nghĩa DistilBERT hoàn toàn không hiểu từ chưa fine-tune. Mô hình holdout vẫn
truy hồi đúng một phần query và một số target rank còn cải thiện, phù hợp với khả năng chuyển
giao từ pretraining và các từ liên quan như `shirt` vẫn còn trong dữ liệu. Kết luận phù hợp là:

> AACL có khả năng khái quát từ vựng một phần nhờ text encoder tiền huấn luyện, nhưng grounding
> theo nhiệm vụ retrieval đối với surface form không thấy trong fine-tuning suy giảm, rõ nhất ở
> R@50 của nhóm shirt. Đây là lỗi generalization/grounding, chưa phải hallucination theo nghĩa
> sinh nội dung.

Một giới hạn của thiết kế là việc loại toàn bộ record đồng thời loại cả cặp ảnh và các thuộc tính
đi kèm, không chỉ loại từ. Validation đầy đủ gần như ổn định giúp giảm bớt lo ngại về suy giảm
chung do thiếu dữ liệu, nhưng chưa loại bỏ hoàn toàn confound này. Nếu có thêm ngân sách GPU,
có thể huấn luyện matched-size control bằng cách loại ngẫu nhiên cùng số record không chứa term.
