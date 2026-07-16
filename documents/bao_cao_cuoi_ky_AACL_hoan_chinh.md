# HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG

## KHOA ĐÀO TẠO SAU ĐẠI HỌC

# BÁO CÁO KẾT THÚC HỌC PHẦN

**Học phần:** Tìm kiếm và Truy xuất Thông tin  
**Đề tài:** Truy xuất ảnh thời trang với phản hồi văn bản bằng học kết hợp chú ý cộng (AACL)  
**Nhóm thực hiện:** Nhóm 04  
**Lớp:** M25CQKH01-B  
**Địa điểm, thời gian:** Hà Nội, 2026

| Mã học viên | Họ và tên |
|---|---|
| B25CHKH006 | Hứa Sỹ Đạo |
| B25CHKH010 | Nguyễn Hoàng Dương |
| B25CHKH034 | Nguyễn Hoàng Lân |
| B25CHKH041 | Vũ Sơn |

---

## Phân công nhiệm vụ

| TT | Công việc | Học viên thực hiện |
|---:|---|---|
| 1 | Nghiên cứu cơ sở lý thuyết về truy xuất đa phương thức, phương pháp liên quan và bộ dữ liệu | Hứa Sỹ Đạo |
| 2 | Cài đặt Image Encoder, Text Encoder, pipeline dữ liệu, môi trường huấn luyện và checkpoint | Nguyễn Hoàng Dương |
| 3 | Cài đặt Additive Attention Composition Module, thiết kế kiểm thử hallucination và biên tập báo cáo | Nguyễn Hoàng Lân |
| 4 | Huấn luyện, đo Recall@K, trực quan hóa attention flow và phân tích sai số | Vũ Sơn |

## Tóm tắt

Truy xuất ảnh với phản hồi văn bản nhận một ảnh nguồn và một mô tả sửa đổi, sau đó tìm trong kho
ảnh các sản phẩm vừa bảo toàn nội dung cần giữ của ảnh nguồn, vừa thỏa mãn yêu cầu mới trong văn
bản. Báo cáo nghiên cứu và cài đặt lại Additive Attention Compositional Learning (AACL), mô hình
được Tian, Newsam và Boakye công bố tại WACV 2023 [1]. AACL dùng Swin Transformer mã hóa ảnh,
DistilBERT mã hóa văn bản và additive attention để học vector ngữ cảnh toàn cục dùng chung khi
điều chỉnh các token ảnh.

Nhóm huấn luyện độc lập trên ba category FashionIQ gồm `dress`, `shirt` và `toptee`. Với một seed,
mô hình đạt macro R@10 = 19,8376 và R@50 = 45,2071. Ngoài Recall, báo cáo tập trung vào hai câu hỏi
thực nghiệm. Thứ nhất, khi khái niệm `T-shirt` và các cách viết tương đương không xuất hiện trong
dữ liệu fine-tuning, mô hình còn grounding được truy vấn hay tạo hành vi giống hallucination hay
không? Lexical holdout cho thấy khả năng chuyển giao một phần từ text encoder tiền huấn luyện,
nhưng R@50 trên tập mục tiêu `shirt` giảm 11,3772 điểm với bootstrap 95% CI không chứa 0. Với yêu
cầu vô nghiệm, mô hình vẫn cưỡng bức trả top-K, tạo false grounding dù không sinh ảnh mới.

Thứ hai, AACL có thực sự sử dụng global context hay chỉ tạo heatmap có vẻ hợp lý? Counterfactual
text làm embedding và top-5 thay đổi mạnh nhưng attention map vẫn tương quan cao và chưa định vị
đúng tay áo. Occlusion cho faithfulness một phần: che 10% vùng attention cao gây ảnh hưởng lớn hơn
vùng thấp, nhưng hiệu ứng không ổn định ở mask lớn và trên ground-truth target. Bằng chứng mạnh
nhất đến từ context intervention: trên 2.038 validation record `shirt`, tráo context làm R@50 giảm
từ 40,7753 xuống 3,4347; uniform context chỉ còn 13,0520. Kết luận phù hợp là AACL học và sử dụng
global context phụ thuộc query có ích cho retrieval, nhưng chưa đủ bằng chứng để nói mô hình hiểu
ngữ cảnh theo nghĩa ngữ nghĩa và nhân quả của con người.

**Từ khóa:** compositional image retrieval, FashionIQ, AACL, additive attention, global context,
lexical holdout, hallucination, occlusion faithfulness.

---

# 1. Giới thiệu

## 1.1. Bối cảnh

Trong tìm kiếm sản phẩm thời trang, một truy vấn ảnh đơn lẻ biểu đạt tốt kiểu dáng tổng thể nhưng
khó chỉ rõ phần nào người dùng muốn thay đổi. Ngược lại, văn bản có thể mô tả “màu xanh hơn”, “tay
dài hơn” hoặc “đổi họa tiết”, nhưng thiếu thông tin về phom, chất liệu và các chi tiết cần giữ của
sản phẩm nguồn. Truy xuất ảnh với phản hồi văn bản kết hợp hai tín hiệu này thành truy vấn
đa phương thức `(x,t)`, trong đó `x` là ảnh nguồn và `t` là yêu cầu sửa đổi.

Mục tiêu của hệ thống là xếp hạng ảnh đích `y` sao cho ảnh vừa giống nguồn ở các thuộc tính không
được nhắc tới, vừa thỏa mãn thay đổi trong văn bản. Đây là bài toán fine-grained: sai khác nhỏ ở
cổ áo, tay áo, chiều dài, màu hoặc họa tiết có thể quyết định tính liên quan. Đồng thời, một ảnh
trong kho có thể hợp lý với người dùng nhưng không phải target duy nhất do benchmark gán, khiến
Recall@K có thể đánh giá thấp relevance cảm nhận.

## 1.2. Mục tiêu và câu hỏi nghiên cứu

Báo cáo có bốn mục tiêu:

1. Phân tích nguyên lý của AACL và mối liên hệ với truy xuất thông tin đa phương thức.
2. Cài đặt, huấn luyện và đánh giá lại mô hình trên FashionIQ bằng một pipeline có thể truy vết.
3. Kiểm tra khả năng khái quát từ vựng và hành vi false grounding trước truy vấn ngoài miền/vô nghiệm.
4. Kiểm chứng tuyên bố global context bằng attention visualization, occlusion và context intervention.

Hai câu hỏi trọng tâm là:

- **RQ1 — Generalization và hallucination:** nếu fine-tuning không chứa `T-shirt`, `T shirt`,
  `tshirt` hoặc `tee`, mô hình còn truy xuất đúng khi gặp các cách viết này không? Với yêu cầu
  không thể thỏa mãn, mô hình có biết từ chối hay vẫn tạo false grounding?
- **RQ2 — Global context:** attention map có phản ánh vùng ngữ nghĩa đúng và context vector có
  thực sự ảnh hưởng đến representation/ranking hay không?

## 1.3. Đóng góp thực nghiệm của nhóm

Ngoài baseline FashionIQ, nhóm bổ sung bốn lớp kiểm thử:

- vocabulary audit và lexical holdout không làm thay đổi dữ liệu gốc;
- bộ prompt có cấu trúc cho in-domain, paraphrase, identity, contradiction, OOD và unsatisfiable;
- attention extraction theo block/head/token và counterfactual visualization;
- high/low-attention occlusion cùng diện tích và shuffled/uniform-context intervention.

Mọi kết quả của nhóm được lấy từ checkpoint, JSON hoặc CSV trong project. Số liệu của bài báo gốc
được ghi rõ là tham chiếu và không được trình bày như kết quả tái hiện.

---

# 2. Cơ sở lý thuyết và phương pháp liên quan

## 2.1. Phát biểu bài toán

Gọi `f(x,t)` là hàm hợp thành ảnh nguồn và phản hồi, `g(y)` là bộ mã hóa ảnh gallery. Hệ thống xếp
hạng theo cosine hoặc tích vô hướng giữa embedding đã chuẩn hóa:

$$
s(x,t,y) = f(x,t)^\top g(y).
$$

Trong một batch có `B` cặp đúng, mục tiêu phân loại theo batch đẩy truy vấn gần target tương ứng và
xa các target còn lại:

$$
\mathcal{L}_{q\rightarrow y}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp(s(q_i,y_i)/\tau)}
{\sum_{j=1}^{B}\exp(s(q_i,y_j)/\tau)}.
$$

Implementation của nhóm dùng loss đối xứng, tức trung bình thêm chiều `y→q`, temperature 0,07 và
label smoothing 0,1. Đây là mở rộng triển khai so với công thức một chiều được trình bày trong bài
báo, vì vậy kết quả được xem là tái hiện có điều chỉnh chứ không phải sao chép tuyệt đối protocol.

## 2.2. Các hướng hợp thành ảnh–văn bản

Các phương pháp trước AACL gồm MRN với residual và phép nhân phần tử [6], FiLM với affine
conditioning [7], TIRG với gating và residual [2], ComposeAE với không gian embedding phức [5],
MAAF với attention fusion [3], và RTIC với residual cùng graph regularization [4]. Điểm khác biệt
của AACL là tạo một context vector từ toàn bộ chuỗi ảnh–văn bản rồi dùng lại vector này để điều
chỉnh từng token, thay vì xây ma trận quan hệ cặp–cặp đầy đủ.

Dot-product self-attention chuẩn có ma trận `N×N`; additive attention trong AACL tính một trọng số
cho mỗi token nên phần tổng hợp context tăng tuyến tính theo số token. Đổi lại, quan hệ chi tiết có
thể bị nén vào một vector duy nhất. Vì vậy, “global context” chỉ nên được hiểu ban đầu là phạm vi
tổng hợp toàn chuỗi, chưa đồng nghĩa với hiểu ngữ nghĩa toàn cục.

## 2.3. Recall@K và giới hạn đánh giá

Với mỗi query, Recall@K bằng 1 nếu target chính thức xuất hiện trong K kết quả đầu, ngược lại bằng
0. Kết quả cuối là tỷ lệ phần trăm trên toàn bộ query. Báo cáo dùng R@10 và R@50 theo FashionIQ,
đồng thời loại ảnh nguồn khỏi candidate ranking nếu ID đó có trong gallery.

Recall có ưu điểm rõ ràng và tái lập được, nhưng FashionIQ chỉ gán một target cho mỗi query. Nhiều
ảnh khác có thể vẫn đúng màu, kiểu và sửa đổi mong muốn. Do đó, báo cáo kết hợp Recall với target
rank, MRR, retrieval visualization, embedding similarity, top-K overlap và perturbation test.

## 2.4. Hallucination trong hệ truy xuất

AACL không sinh ảnh; đầu ra luôn là một phần tử của gallery. Vì vậy, “hallucination” ở đây không
nên được dùng theo nghĩa mô hình sinh bịa pixel hoặc sản phẩm. Hai khái niệm phù hợp hơn là:

- **false grounding:** mô hình ánh xạ yêu cầu không được hỗ trợ sang ảnh có vẻ gần nhưng không
  thỏa yêu cầu;
- **forced retrieval:** hệ thống luôn trả top-K dù query mâu thuẫn, ngoài miền hoặc không có đáp án.

Hai hành vi trên vẫn quan trọng trong giao diện tìm kiếm vì người dùng có thể hiểu top-1 là một
kết quả hợp lệ nếu hệ thống không có confidence threshold hoặc cơ chế abstention.

---

# 3. Kiến trúc AACL và cài đặt

## 3.1. Image encoder

Ảnh được resize về 256, center crop 224 ở evaluation và chuẩn hóa ImageNet. Backbone là
`swin_base_patch4_window7_224.ms_in22k_ft_in1k`, tiền huấn luyện ImageNet-22K rồi fine-tune
ImageNet-1K. Nhóm lấy feature từ Stage 3 và Stage 4, chiếu về 768 chiều và adaptive-pool mỗi stage
thành lưới 7×7. Sau khi nối hai stage, mỗi ảnh có 98 token.

Việc dùng Stage 3+4 theo thiết kế bài báo giúp kết hợp thông tin trung gian và ngữ nghĩa cao. Trong
ablation Shopping100k của tác giả, Stage 3+4 đạt R@10/R@50 = 49,20/81,29, cao hơn nhẹ so với chỉ
Stage 4 hoặc ghép thêm Stage 2 [1]. Nhóm không huấn luyện lại ablation stage riêng và chỉ dùng số
này như bằng chứng tham chiếu.

## 3.2. Text encoder

Phản hồi được mã hóa bằng `distilbert-base-uncased`, embedding 768 chiều, max length 128. Hai câu
caption của FashionIQ được nối bằng token phân cách theo `caption_mode: concat`. Padding mask được
truyền vào composition module để padding token không nhận attention mass. Text encoder được
fine-tune cùng mô hình, không freeze.

## 3.3. Additive Attention Composition

Gọi chuỗi joint image–text sau phép chiếu là
`Φ = [φ_x; φ_t] ∈ R^{N×d}`. Với mỗi attention head, block tính hidden state:

$$
h_i = F_h(\phi_i),
$$

trọng số token:

$$
\alpha_i =
\frac{\exp(w_h^\top h_i/\sqrt{d_h})}
{\sum_j\exp(w_h^\top h_j/\sqrt{d_h})},
$$

và context vector:

$$
c = \sum_i \alpha_i h_i.
$$

Context được nhân Hadamard với từng token rồi qua output projection:

$$
o_i = h_i + F_o(c\odot h_i).
$$

Implementation gồm ba composition block, tám head mỗi block, residual connection, LayerNorm và
feed-forward network. Sau block cuối, chỉ 98 image token được lấy ra, chuẩn hóa, mean-pool và
L2-normalize thành query embedding. Gallery image embedding được tạo từ image token mà không dùng
text composition.

## 3.4. Khác biệt so với protocol bài báo

Bài báo huấn luyện bằng SGD, learning rate 0,035, bốn GPU, batch 32 trên mỗi GPU và báo trung bình
năm trial [1]. Project dùng AdamW, learning rate riêng cho encoder/composition, một seed và
micro-batch 8 trên một process GPU. Vì batch-softmax chỉ thấy tám mẫu trong mỗi forward, mỗi query
có bảy in-batch negative; gradient accumulation 4 chỉ làm effective optimizer batch thành 32,
không tăng negative set của loss. Các khác biệt này là nguyên nhân chính khiến không nên so sánh
số tuyệt đối như một exact reproduction.

---

# 4. Thiết lập thực nghiệm

## 4.1. FashionIQ

FashionIQ gồm ba category `dress`, `shirt`, `toptee`, mỗi query có ảnh nguồn, target và trung bình
hai câu sửa đổi tự nhiên [12]. Project dùng split train/val chuẩn:

| Category | Train record | Validation query | Gallery image |
|---|---:|---:|---:|
| Dress | 5.985 | 2.017 | 3.817 |
| Shirt | 5.988 | 2.038 | 6.346 |
| Toptee | 6.027 | 1.961 | 5.373 |

Trong quá trình QA probe định tính, nhóm phát hiện record `B008LTJG3E` có ảnh nguồn là váy nhưng
target/caption liên quan đến giày. Record này chỉ bị loại khỏi manifest minh họa hallucination và
lý do được lưu trong `configs/hallucination_probe_exclusions.json`; dữ liệu benchmark gốc không bị
sửa.

## 4.2. Cấu hình huấn luyện

| Thành phần | Cấu hình |
|---|---|
| Seed | 42 |
| Image encoder | Swin Base, Stage 3+4, 98 token, 768 chiều |
| Text encoder | DistilBERT uncased, max length 128 |
| Composition | 3 block, 8 head, FFN multiplier 4, dropout 0,1 |
| Optimizer | AdamW, weight decay 0,01 |
| Learning rate | image/text `1e-5`; composition `3e-4` |
| Scheduler | cosine, `eta_min=1e-6` |
| Epoch | 60 |
| Batch | micro-batch 8, accumulation 4 |
| Loss | symmetric batch-softmax, temperature 0,07, smoothing 0,1 |
| Evaluation | batch 16, R@10/R@50, loại query image |
| AMP | Có |

Mô hình có 173.780.920 tham số trainable. Các job chạy trên NVIDIA L40 khoảng 44,4 GiB, trong bối
cảnh GPU dùng chung; peak allocated của model khoảng 5,4 GiB. Mỗi category có `best.pt`,
`latest.pt`, resolved config, log theo epoch và `run_summary.json`. Best checkpoint được chọn theo
trung bình R@10 và R@50 trên các epoch evaluation.

## 4.3. Nguyên tắc thống kê và chống chọn mẫu thuận lợi

Các probe attention, occlusion và hallucination được cố định trước khi xem kết quả hoặc được lấy
từ manifest seed 42. Paired comparison dùng cùng query ở hai điều kiện. Occlusion dùng bootstrap
95% CI 5.000 resample và exact sign-flip do chỉ có 10 probe. Context intervention dùng 2.038 record,
bootstrap 5.000 resample và Monte Carlo sign-flip 5.000 lần. Báo cáo giữ cả kết quả thuận, kết quả
một phần và failure case, không chỉ chọn heatmap đẹp.

---

# 5. Kết quả và phân tích

## 5.1. Baseline FashionIQ của nhóm

| Category | Best epoch | R@10 | R@50 |
|---|---:|---:|---:|
| Dress | 60 | 19,6331 | 44,7695 |
| Shirt | 45 | 16,8302 | 40,7753 |
| Toptee | 55 | 23,0495 | 50,0765 |
| **Macro average** | — | **19,8376** | **45,2071** |

Toptee đạt kết quả cao nhất, shirt thấp nhất. Điều này có thể liên quan đến kích thước gallery,
độ đa dạng fine-grained và phân bố caption khác nhau giữa category. Cả ba checkpoint đã được
evaluate lại độc lập và kết quả khớp `run_summary.json`.

### Đối chiếu với bài báo gốc

| Nguồn kết quả | Shirt R@10/R@50 | Dress R@10/R@50 | Toptee R@10/R@50 |
|---|---:|---:|---:|
| AACL trong bài báo, trung bình 5 trial [1] | 24,82 / 48,85 | 29,89 / 55,85 | 30,88 / 56,85 |
| Nhóm tái hiện, seed 42 | 16,83 / 40,78 | 19,63 / 44,77 | 23,05 / 50,08 |

Kết quả nhóm thấp hơn bài báo ở cả ba category, nhưng đây không phải so sánh cùng protocol. Bài
báo dùng 4×32 sample mỗi step và SGD, trong khi project có tám sample trong batch-softmax, AdamW
và một seed. Negative set nhỏ hơn làm bài toán phân biệt trong loss dễ hơn nhưng cung cấp ít hard
negative, thường làm chất lượng retrieval toàn gallery kém hơn. Do chưa chạy cùng năm seed và cấu
hình của tác giả, báo cáo không kết luận khoảng cách này xuất phát riêng từ composition module.

## 5.2. RQ1a — Vocabulary audit và lexical holdout `T-shirt`

Audit không phân biệt hoa/thường, chấp nhận số nhiều và chỉ khớp token/cụm độc lập:

| Category | Split | Record | Record chứa term | `t-shirt` | `t shirt` | `tshirt` | `tee` |
|---|---|---:|---:|---:|---:|---:|---:|
| Dress | train | 5.985 | 3 | 2 | 0 | 0 | 1 |
| Dress | val | 2.017 | 4 | 0 | 3 | 0 | 1 |
| Shirt | train | 5.988 | 483 | 147 | 150 | 49 | 159 |
| Shirt | val | 2.038 | 167 | 55 | 58 | 23 | 42 |
| Toptee | train | 6.027 | 261 | 79 | 34 | 45 | 115 |
| Toptee | val | 1.961 | 94 | 29 | 10 | 8 | 50 |

Lexical-holdout loại toàn bộ train record nếu một trong hai caption chứa surface form mục tiêu.
Cách này ngăn caption còn lại làm rò rỉ term khi hai caption được nối. Sau lọc, train còn 5.505
record shirt và 5.766 record toptee; dress chỉ có bốn validation query mục tiêu nên không được dùng
cho kết luận riêng.

| Category | Model | Validation | N | R@10 | R@50 | Median rank | MRR |
|---|---|---|---:|---:|---:|---:|---:|
| Shirt | Full train | Full val | 2.038 | 16,8302 | 40,7753 | 86,0 | 0,0864 |
| Shirt | Lexical holdout | Full val | 2.038 | 18,2532 | 39,8430 | 91,0 | 0,0884 |
| Shirt | Full train | Lexical val | 167 | 10,1796 | 31,1377 | 144,0 | 0,0583 |
| Shirt | Lexical holdout | Lexical val | 167 | 7,1856 | 19,7605 | 186,0 | 0,0384 |
| Toptee | Full train | Full val | 1.961 | 23,0495 | 50,0765 | 50,0 | 0,1144 |
| Toptee | Lexical holdout | Full val | 1.961 | 24,4773 | 49,4646 | 53,0 | 0,1201 |
| Toptee | Full train | Lexical val | 94 | 13,8298 | 44,6809 | 66,5 | 0,0626 |
| Toptee | Lexical holdout | Lexical val | 94 | 12,7660 | 37,2340 | 86,5 | 0,0686 |

Trên full validation, holdout model gần như không suy giảm, nên không có dấu hiệu mô hình bị hỏng
chung chỉ vì mất một phần dữ liệu. Trên 167 lexical query shirt, R@50 giảm 11,3772 điểm với paired
bootstrap 95% CI `[−17,9641; −4,7904]`; target rank xấu đi ở 104 query, bằng ở 2 và tốt hơn ở 61.
Đây là bằng chứng rõ nhất về suy giảm task-specific grounding khi term không xuất hiện trong
fine-tuning. Với toptee, R@50 giảm 7,4468 điểm nhưng CI `[−18,0851; 3,1915]` chứa 0 do chỉ có 94
query, vì vậy chưa thể khẳng định hiệu ứng ổn định.

Kết quả không có nghĩa DistilBERT hoàn toàn không hiểu `T-shirt`: holdout model vẫn trả đúng một
phần query nhờ pretraining, subword và các khái niệm liên quan còn trong dữ liệu. Kết luận là mô
hình có **khả năng khái quát từ vựng một phần**, nhưng grounding theo nhiệm vụ suy giảm, đặc biệt
trên shirt. Đây là generalization error, chưa phải hallucination sinh nội dung.

## 5.3. RQ1b — Prompt vô nghiệm và forced retrieval

Nhóm dùng cùng prompt ngoài khả năng của gallery cho probe `q01` cố định ở ba category:

> Turn it into a transparent glass garment with animated flames and invisible fabric.

| Category | Probe | Query ID | Top-1 similarity | Top-1/Top-2 margin |
|---|---|---|---:|---:|
| Dress | `dress_q01` | `B00ANK6ND0` | 0,3041 | 0,0157 |
| Shirt | `shirt_q01` | `B003JY6WY2` | 0,2787 | 0,0036 |
| Toptee | `toptee_q01` | `B008BT599E` | 0,2547 | 0,0032 |

![Kết quả top-5 trước prompt vô nghiệm](../outputs/report_assets/fig_hallucination_cases.png)

**Hình 1.** Top-5 của ba checkpoint trước cùng prompt không thể thỏa mãn. Mỗi hàng dùng probe
được cố định trước retrieval. Mô hình vẫn trả láng giềng trong gallery dù không ảnh nào đáp ứng
đầy đủ yêu cầu.

Dress chủ yếu trả váy hoa thông thường; shirt và toptee còn có kết quả lệch loại rõ. Similarity và
margin vẫn được tạo như với query bình thường, không có nhánh “không tìm thấy”. Kết quả chứng minh
sự tồn tại của false grounding/forced retrieval nhưng không ước lượng tỷ lệ lỗi trên toàn bộ dữ
liệu. Do không chấm relevance toàn bộ top-5 bởi nhiều người đánh giá, nhóm không tạo các metric
`TextMatch@5`, `FullMatch@5`, false-acceptance rate hoặc Cohen's kappa từ nhãn suy đoán.

## 5.4. RQ2a — Attention visualization dưới counterfactual text

Nhóm dùng checkpoint shirt, ảnh nguồn `B003JY6WY2` và hai prompt:

1. `Make the shirt have longer sleeves.`
2. `Make the shirt have a different graphic.`

Attention tensor có shape `[2,3,8,108]`: hai prompt, ba block, tám head và 108 token gồm 98 image
token cùng 10 text token sau padding. Padding mass bằng 0; attention mỗi head có tổng bằng 1. Stage
3 và Stage 4 được map riêng về 7×7, chuẩn hóa rồi mới lấy trung bình.

![Attention flow dưới hai counterfactual prompt](../outputs/report_assets/fig_attention_counterfactual.png)

**Hình 2.** Attention flow của cùng ảnh nguồn dưới hai prompt. Hai top-5 không trùng nhau, nhưng
average-stage map vẫn cùng tập trung vào logo giữa ngực; prompt “longer sleeves” không dịch chuyển
attention rõ đến tay áo.

| Chỉ báo so sánh hai prompt | Giá trị |
|---|---:|
| Cosine giữa query embedding | 0,1954 |
| Pearson giữa average-stage map | 0,8780 |
| Jensen–Shannon divergence | 0,0402 |
| Mean absolute map difference | 0,0754 |
| Top-5 overlap | 0/5 |

Embedding cosine thấp và top-5 thay đổi hoàn toàn cho thấy text không bị bỏ qua. Tuy nhiên,
attention map có Pearson 0,8780 và cùng tập trung vào logo. Text-token flow cũng không ưu tiên ổn
định từ khóa `longer` hoặc `graphic`. Do đó, visualization chỉ hỗ trợ text-conditioned behavior,
chưa chứng minh semantic localization đúng như cách con người hiểu yêu cầu.

## 5.5. RQ2b — Occlusion faithfulness

Trên 10 probe shirt cố định, nhóm chọn các patch attention cao nhất và thấp nhất trên lưới 7×7 ở
ba tỷ lệ. Mask high/low có cùng 5/10/15 patch, không chồng nhau; pixel che được đặt về ImageNet
mean. Primary endpoint cố định top-1 của query không che làm reference và đo similarity/rank của
chính ảnh đó sau occlusion.

![Occlusion vùng attention cao và thấp](../outputs/report_assets/fig_occlusion_comparison.png)

**Hình 3.** Ảnh nguồn, mask high/low 20% cùng diện tích và mức thay đổi top-1 reference khi che
10%, 20%, 30% patch.

| Tỷ lệ | Δsim high | Δsim low | high−low [bootstrap 95% CI] | Δrank high | Δrank low | p |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0,0015 | −0,0018 | 0,0033 [0,0006; 0,0057] | 0,30 | 0,00 | 0,0391 |
| 20% | 0,0041 | −0,0006 | 0,0047 [−0,0004; 0,0105] | 0,70 | 0,00 | 0,1289 |
| 30% | 0,0053 | −0,0000 | 0,0053 [−0,0025; 0,0134] | 1,00 | 0,20 | 0,2520 |

Ở 10%, high-attention mask gây giảm similarity lớn hơn low-attention trên 9/10 probe; CI không cắt
0. Query-embedding cosine-drop high−low cũng dương có ý nghĩa ở cả ba tỷ lệ: 0,0100, 0,0148 và
0,0212, với ba CI đều không cắt 0. Đây là bằng chứng rằng vùng attention cao quan trọng hơn đối
với representation nội bộ.

Tuy nhiên, primary endpoint 20%/30% có CI cắt 0, `shirt_q04` đảo chiều ở 10%, và secondary endpoint
trên FashionIQ target không ổn định (`p>0,6`). Attention vì vậy có **faithfulness một phần**, không
phải explanation nhân quả hoàn chỉnh.

## 5.6. RQ2c — Context intervention trên toàn validation set

Nhóm can thiệp trực tiếp vào `c` trong từng composition block của checkpoint shirt:

- **shuffled:** dịch vòng context giữa các mẫu trong fixed batch 16, giữ nguyên ảnh và caption;
- **uniform:** thay learned `alpha` bằng trọng số đều trên token hợp lệ.

Thí nghiệm dùng toàn bộ 2.038 query-caption record, 1.541 source ID phân biệt và cùng gallery.

![Recall dưới context intervention](../outputs/report_assets/fig_context_intervention.png)

**Hình 4.** Recall của Full AACL, shuffled context và uniform context trên cùng checkpoint/gallery.

| Variant | R@10 | R@50 | ΔR@10 | ΔR@50 | Median rank | Cosine→Full | Top-5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full AACL | 16,8302 | 40,7753 | 0,0000 | 0,0000 | 86,0 | 1,0000 | 5,00/5 |
| Shuffled context | 1,2758 | 3,4347 | −15,5545 | −37,3405 | 2.080,0 | 0,2223 | 0,09/5 |
| Uniform context | 4,3670 | 13,0520 | −12,4632 | −27,7233 | 951,5 | 0,4819 | 0,35/5 |

Full AACL khớp tuyệt đối evaluation độc lập trước đó. Shuffled làm target rank xấu đi trên 89,7%
record; mean increase 1.885,79, bootstrap CI `[1.797,97; 1.972,41]`, Monte Carlo sign-flip
`p≈0,0002`. Uniform làm rank xấu đi trên 83,8%; mean increase 1.202,71, CI
`[1.126,42; 1.279,82]`, `p≈0,0002`.

Đây là bằng chứng mạnh về **query-specific context dependence**: context của query khác không thể
thay context đúng, và learned weighting hữu ích hơn trung bình đều. Tuy nhiên, shuffled tạo trạng
thái ngoài phân phối và phụ thuộc fixed permutation; uniform không được retrain. Vì vậy, kết quả
không thay thế một architectural ablation được tối ưu từ đầu.

---

# 6. Thảo luận

## 6.1. Trả lời RQ1: mô hình có hallucinate không?

Câu trả lời phụ thuộc định nghĩa:

- **Không theo nghĩa sinh nội dung:** AACL chỉ xếp hạng ảnh có sẵn, không tạo trang phục hoặc pixel.
- **Có hành vi hallucination-like/false grounding:** với prompt vô nghiệm, mô hình vẫn trả top-K
  và không biểu thị rằng yêu cầu không có trong gallery.
- **Có giới hạn lexical generalization:** khi loại các cách viết `T-shirt` khỏi fine-tuning, mô hình
  vẫn hiểu một phần nhờ DistilBERT nhưng targeted retrieval suy giảm rõ trên shirt.

Do đó, ví dụ “train trên shirt/dress/toptee nhưng prompt T-shirt” nên được trình bày như kiểm thử
OOD lexical grounding. Nếu `T-shirt` chưa thấy trong fine-tuning nhưng text encoder đã thấy trong
pretraining, truy vấn không hoàn toàn “chưa từng biết”. Lexical holdout đo đúng phần đóng góp của
task fine-tuning và cho kết luận thận trọng hơn một contact sheet đơn lẻ.

Về sản phẩm thực tế, hệ thống cần một nhánh abstention: nếu top-1 similarity thấp, margin nhỏ hoặc
query OOD, giao diện nên báo “không tìm thấy sản phẩm đáp ứng đầy đủ” thay vì ngầm xác nhận top-1.
Threshold cần được hiệu chỉnh trên một tập query satisfiable/unsatisfiable có nhãn, không thể chọn
từ ba ví dụ minh họa.

## 6.2. Trả lời RQ2: AACL có hiểu global context không?

Ba lớp bằng chứng cho các kết luận khác nhau:

1. **Counterfactual behavior:** text làm embedding và ranking thay đổi mạnh — context có điều kiện
   hóa theo văn bản.
2. **Occlusion:** vùng attention cao ảnh hưởng representation nhiều hơn vùng thấp — attention có
   faithfulness một phần.
3. **Context intervention:** Recall sụt rất mạnh khi `c` bị tráo hoặc làm đều — checkpoint thực sự
   phụ thuộc context đúng của query và learned token weighting.

Như vậy, nhóm có thể khẳng định AACL **học và sử dụng global context hữu ích về mặt hành vi**. Nhóm
không nên viết “mô hình hiểu hoàn toàn global context”, bởi heatmap không định vị rõ `longer
sleeves`, target-occlusion không ổn định, và intervention chỉ cho sensitivity chứ không giải thích
logic ngữ nghĩa bên trong.

## 6.3. Quan hệ giữa các kết quả

Lexical holdout và context intervention bổ sung cho nhau. Text encoder tiền huấn luyện giúp mô hình
không sụp đổ hoàn toàn với term holdout, nhưng task grounding vẫn cần fine-tuning. Khi context đúng
bị tráo, retrieval gần như sụp đổ, chứng tỏ composition không đơn thuần bỏ qua text. Attention
visualization lại cho thấy sử dụng text không đồng nghĩa với localization đúng. Đây là lý do báo
cáo cần kết hợp metric hành vi, perturbation và hình ảnh thay vì dùng riêng một heatmap.

## 6.4. Các nguồn sai số và đe dọa tính hợp lệ

- Chỉ có một seed cho baseline và holdout; bootstrap trên query không thay thế biến thiên giữa lần train.
- Batch-softmax chỉ có bảy negative/query, thấp hơn protocol bài báo.
- Lexical holdout loại cả record và các thuộc tính đi kèm, không chỉ loại token; chưa có matched-size random control.
- Ba ca hallucination chứng minh sự tồn tại của lỗi, không cung cấp tỷ lệ lỗi toàn hệ thống.
- Occlusion chỉ có 10 probe và mean-fill có thể tạo distribution shift.
- Attention lưới 7×7 có độ phân giải thấp; nhân flow qua block không phải gradient attribution.
- Shuffled context phụ thuộc fixed batch/permutation; uniform context không được retrain.
- FashionIQ chỉ có một official target dù nhiều ảnh có thể hợp lý.
- Không chạy Fashion200k/Shopping100k trong project; mọi số liệu hai bộ này chỉ là tham chiếu [1].

---

# 7. Kết luận và hướng phát triển

## 7.1. Kết luận

Nhóm đã cài đặt một pipeline AACL hoàn chỉnh trên FashionIQ, gồm loader, Swin/DistilBERT encoder,
multi-head additive-attention composition, training có AMP/checkpoint/resume, evaluation Recall và
các công cụ hậu kiểm. Ba best checkpoint đạt macro R@10 = 19,8376 và R@50 = 45,2071 với seed 42.

Đối với RQ1, mô hình khái quát một phần tới các surface form `T-shirt` không thấy khi fine-tune,
nhưng grounding suy giảm rõ trên targeted shirt validation. Với prompt vô nghiệm, mô hình không
sinh ảnh giả nhưng vẫn cưỡng bức trả kết quả không thỏa yêu cầu. Vì vậy, hạn chế đúng là lexical
generalization, false grounding và thiếu abstention.

Đối với RQ2, attention visualization một mình không chứng minh hiểu ngữ nghĩa. Occlusion cung cấp
faithfulness một phần, còn context intervention cho bằng chứng mạnh rằng context vector đúng và
learned token weighting là thiết yếu đối với checkpoint. Kết luận cuối cùng là AACL sử dụng
query-specific global context hữu ích để xây dựng representation và ranking; semantic
localization và explanation nhân quả vẫn chưa hoàn chỉnh.

## 7.2. Hướng phát triển

1. Chạy ít nhất ba seed, báo trung bình và độ lệch chuẩn giữa lần train.
2. Tăng negative set bằng distributed training, memory bank hoặc hard-negative mining.
3. Huấn luyện matched-size random-removal control cho lexical holdout.
4. Xây dựng tập OOD/unsatisfiable có nhãn để hiệu chỉnh confidence và ngưỡng abstention.
5. Retrain uniform, image-only, text-only, dot-product và addition variants với cùng protocol.
6. Kết hợp attention với gradient attribution, blur/inpainting occlusion và nhiều probe hơn.
7. Dùng relevance đa mức hoặc nhiều target để phản ánh chất lượng cảm nhận tốt hơn Recall nhãn đơn.

---

# Phụ lục A. Khả năng tái lập

## A.1. Artifact chính

| Nội dung | Đường dẫn trong project |
|---|---|
| Cấu hình baseline | `configs/fashioniq_l40_shared.yaml` |
| Checkpoint/log | `outputs/fashioniq_improved/l40_shared_seed42/<category>/` |
| Lexical holdout | `outputs/lexical_holdout_comparison/` |
| Hallucination retrieval | `outputs/hallucination/<category>/` |
| Attention raw/figure | `outputs/attention/shirt/shirt_q01/` |
| Occlusion | `outputs/occlusion/shirt/shirt_probe10/` |
| Context intervention | `outputs/context_intervention/shirt/seed42/` |
| Bảng/hình báo cáo | `outputs/report_assets/` |

## A.2. Các lệnh chính

```bash
source .venv/bin/activate

# Baseline
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/fashioniq_l40_shared.yaml --category dress

# Đánh giá độc lập
python evaluate.py \
  --config configs/fashioniq_l40_shared.yaml \
  --checkpoint outputs/fashioniq_improved/l40_shared_seed42/dress/best.pt \
  --category dress

# Attention counterfactual
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_attention.py \
  --category shirt --probe-id shirt_q01 --device cuda --save-heads

# Occlusion faithfulness
CUDA_VISIBLE_DEVICES=0 python scripts/run_occlusion_faithfulness.py \
  --category shirt --num-probes 10 --device cuda

# Context intervention
CUDA_VISIBLE_DEVICES=0 python scripts/run_context_intervention.py \
  --category shirt --device cuda

# Unit test
python -m unittest discover -s tests
```

---

# Tài liệu tham khảo

[1] Y. Tian, S. Newsam, K. Boakye, “Fashion Image Retrieval with Text Feedback by Additive
Attention Compositional Learning,” *Proceedings of WACV*, 2023.

[2] N. Vo, L. Jiang, C. Sun, K. Murphy, L.-J. Li, L. Fei-Fei, J. Hays, “Composing Text and Image
for Image Retrieval — An Empirical Odyssey,” *CVPR*, 2019.

[3] E. Dodds, J. Culpepper, S. Herdade, Y. Zhang, K. Boakye, “Modality-Agnostic Attention Fusion
for Visual Search with Text Feedback,” 2020.

[4] M. Shin, Y. Cho, B. Ko, G. Gu, “RTIC: Residual Learning for Text and Image Composition Using
Graph Convolutional Network,” 2021.

[5] M. U. Anwaar, E. Labintcev, M. Kleinsteuber, “Compositional Learning of Image-Text Query for
Image Retrieval,” *WACV*, 2021.

[6] J.-H. Kim et al., “Multimodal Residual Learning for Visual QA,” *NeurIPS*, 2016.

[7] E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville, “FiLM: Visual Reasoning with a
General Conditioning Layer,” *AAAI*, 2018.

[8] D. Bahdanau, K. Cho, Y. Bengio, “Neural Machine Translation by Jointly Learning to Align and
Translate,” 2014.

[9] A. Vaswani et al., “Attention Is All You Need,” *NeurIPS*, 2017.

[10] Z. Liu et al., “Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows,”
*ICCV*, 2021.

[11] V. Sanh, L. Debut, J. Chaumond, T. Wolf, “DistilBERT, a Distilled Version of BERT: Smaller,
Faster, Cheaper and Lighter,” 2019.

[12] X. Guo et al., “The Fashion IQ Dataset: Retrieving Images by Combining Side Information and
Relative Natural Language Feedback,” 2019.

[13] X. Han et al., “Automatic Spatially-Aware Fashion Concept Discovery,” *ICCV*, 2017.

[14] K. E. Ak, J. H. Lim, J. Y. Tham, A. A. Kassim, “Efficient Multi-Attribute Similarity Learning
Towards Attribute-Based Fashion Search,” *WACV*, 2018.

[15] S. Abnar, W. Zuidema, “Quantifying Attention Flow in Transformers,” *ACL*, 2020.

[16] H. Chefer, S. Gur, L. Wolf, “Transformer Interpretability Beyond Attention Visualization,”
*CVPR*, 2021.
