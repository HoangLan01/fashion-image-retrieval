# Hướng tiếp cận kiểm thử hallucination và global context của AACL

## 1. Mục tiêu

Tài liệu này đề xuất hai nhóm thực nghiệm bổ sung cho báo cáo cuối kỳ về Additive Attention Compositional Learning (AACL):

1. Kiểm tra hành vi của mô hình với từ ngữ chưa gặp trong quá trình fine-tuning, truy vấn ngoài miền, truy vấn mâu thuẫn và truy vấn không có đáp án trong kho ảnh.
2. Kiểm tra liệu global context của AACL có thực sự được mô hình sử dụng để điều kiện hóa phép truy xuất theo cả ảnh nguồn và phản hồi văn bản hay không.

Hai câu hỏi cần được phát biểu thận trọng. AACL là mô hình truy xuất, không phải mô hình sinh ảnh, nên không hallucinate theo đúng nghĩa thường dùng cho mô hình sinh. Tương tự, attention visualization chỉ mô tả sự phân bố trọng số, chưa tự nó chứng minh mô hình “hiểu” ngữ cảnh theo nghĩa nhận thức của con người.

## 2. Câu hỏi nghiên cứu và giả thuyết

### 2.1. Câu hỏi RQ1: AACL có biểu hiện tương tự hallucination không?

Trong phạm vi báo cáo, nhóm định nghĩa **semantic retrieval hallucination** là trường hợp hệ thống trả về kết quả có vẻ đáng tin cậy dù kết quả không được hỗ trợ đầy đủ bởi ảnh nguồn, phản hồi văn bản hoặc tập ứng viên. Biểu hiện cụ thể gồm:

- tự ý thay đổi thuộc tính không được yêu cầu;
- bỏ qua một phần truy vấn mâu thuẫn;
- ánh xạ từ ngoài miền sang một thuộc tính thời trang không có căn cứ;
- luôn trả về top-K dù không tồn tại ảnh thỏa mãn;
- tỏ ra “tự tin” với kết quả chỉ thỏa một phần yêu cầu.

Các giả thuyết cần kiểm tra:

- **H1.1 — Lexical generalization:** các cách diễn đạt đồng nghĩa tạo embedding và ranking tương đối ổn định.
- **H1.2 — OOD sensitivity:** truy vấn ngoài miền hoặc không khả thi có phân bố điểm khác truy vấn hợp lệ.
- **H1.3 — Forced retrieval:** AACL nguyên bản vẫn trả về top-K cho truy vấn không có đáp án vì không có cơ chế abstention.
- **H1.4 — Source preservation:** với truy vấn hợp lệ, mô hình thay đổi thuộc tính được yêu cầu nhưng hạn chế thay đổi các thuộc tính còn lại.

### 2.2. Câu hỏi RQ2: AACL có thực sự sử dụng global context không?

Theo kiến trúc AACL, vector ngữ cảnh được tính bởi:

$$
c=\sum_{i=1}^{N}\alpha_i h_i,
$$

trong đó $h_i$ gồm cả token ảnh và token văn bản. Việc vector $c$ tồn tại là hệ quả trực tiếp của kiến trúc. Điều cần kiểm nghiệm không phải “mô hình có tính $c$ hay không”, mà là:

- $c$ có thay đổi theo ảnh và phản hồi văn bản không;
- $c$ có điều khiển đúng vùng ảnh hoặc thuộc tính liên quan không;
- phá vỡ $c$ có làm chất lượng truy xuất giảm không;
- mô hình có đồng thời tuân thủ yêu cầu thay đổi và bảo toàn ngữ cảnh còn lại không.

Các giả thuyết cần kiểm tra:

- **H2.1 — Text-conditioned attention:** cùng một ảnh nguồn nhưng prompt khác nhau tạo attention map và ranking khác nhau theo hướng hợp lý.
- **H2.2 — Causal faithfulness:** che vùng attention cao làm similarity hoặc thứ hạng target giảm nhiều hơn che vùng attention thấp.
- **H2.3 — Context dependence:** làm nhiễu, tráo hoặc thay global context bằng context đều làm Recall giảm.
- **H2.4 — Global preservation:** full AACL cân bằng tốt hơn giữa tuân thủ text và bảo toàn thuộc tính nguồn so với các biến thể ablation.

## 3. Kiểm tra lại giả định về “T-shirt”

Ví dụ “train trên shirt, dress, toptee nhưng test prompt T-shirt” cần được điều chỉnh trước khi sử dụng làm minh chứng.

Kết quả rà soát captions train hiện có:

| Category train | `t-shirt` | `tshirt` | `tee` |
|---|---:|---:|---:|
| shirt | 147 | 48 | 159 |
| toptee | 79 | 45 | 115 |
| dress | 2 | 0 | 1 |

Đây là số lần xuất hiện của từ/cụm từ, không phải số caption duy nhất. Như vậy, “T-shirt” không phải từ chưa thấy trong FashionIQ train. Ngược lại, chuỗi kỹ thuật `toptee` không xuất hiện trong captions và chỉ đóng vai trò nhãn category của bộ dữ liệu.

Ngoài ra, text encoder là DistilBERT đã được tiền huấn luyện. Một từ không xuất hiện trong FashionIQ vẫn có thể đã được encoder học trước đó. Vì vậy, báo cáo cần phân biệt:

- chưa xuất hiện trong dữ liệu fine-tuning FashionIQ;
- chưa có trong vocabulary/tokenizer;
- chưa từng xuất hiện trong dữ liệu tiền huấn luyện của text encoder;
- khái niệm ngoài miền thời trang;
- khái niệm hợp lệ nhưng không có ảnh tương ứng trong gallery.

### 3.1. Thiết kế lexical holdout

Để tạo thí nghiệm đúng với giả thuyết của giảng viên:

1. Tạo một train split loại các record có caption chứa `t-shirt`, `tshirt` hoặc từ độc lập `tee`.
2. Huấn luyện một checkpoint trên split đã lọc, giữ nguyên cấu hình còn lại.
3. Test các prompt đồng nghĩa như:
   - “make it look like a T-shirt”;
   - “make it look like a tee”;
   - “make it a short-sleeved jersey top”.
4. So sánh với checkpoint huấn luyện đầy đủ.

Kết quả này nên được gọi là **khái quát từ vựng ngoài dữ liệu fine-tuning**. Không nên tuyên bố đây là một khái niệm hoàn toàn chưa từng được mô hình biết, vì DistilBERT có tri thức từ pretraining.

### 3.2. Phân biệt prompt noun-only và relative feedback

FashionIQ huấn luyện trên phản hồi mô tả phép thay đổi tương đối. Prompt chỉ gồm “T-shirt” khác cả về từ vựng lẫn định dạng nhiệm vụ. Nên so sánh ít nhất hai dạng:

- noun-only: “T-shirt”;
- relative feedback: “make it look more like a T-shirt with shorter sleeves”.

Nếu noun-only hoạt động kém hơn, chưa thể kết luận mô hình không hiểu T-shirt; nguyên nhân có thể là prompt format shift.

## 4. Giao thức kiểm thử semantic retrieval hallucination

### 4.1. Phạm vi định tính đã chọn

Do giới hạn thời gian, báo cáo không thực hiện chấm relevance toàn bộ top-5. Pipeline vẫn lưu
manifest 30 probe và raw retrieval để truy vết, nhưng minh họa chính sử dụng cùng probe `q01` đã
được chọn trước retrieval cho mỗi category. Cách này tránh tìm hậu nghiệm đúng các ca có kết quả
xấu nhất.

Ba checkpoint nhận cùng prompt vô nghiệm:

> Turn it into a transparent glass garment with animated flames and invisible fabric.

Hình kết quả cho thấy mô hình luôn trả về các láng giềng trong gallery dù không ảnh nào đáp ứng
đầy đủ yêu cầu. Báo cáo gọi đây là **false grounding/hallucination-like retrieval**, không phải
hallucination sinh nội dung. Không trình bày TextMatch@5, FullMatch@5, FAR hoặc độ đồng thuận khi
không có nhãn thủ công.

### 4.2. Gallery và giới hạn

Mỗi checkpoint vẫn dùng gallery riêng theo category, đúng giao thức training hiện tại. Vì vậy,
kết quả định tính chứng minh sự tồn tại của forced retrieval và thiếu cơ chế abstention, nhưng
không ước lượng tỷ lệ hallucination và không kiểm tra category routing trong mixed-gallery.

### 4.3. Kết luận có thể rút ra

- Nếu paraphrase tạo ranking gần nhau, mô hình có lexical robustness nhất định.
- Nếu lexical holdout vẫn hoạt động, đây là bằng chứng của transfer từ text encoder và/hoặc compositional generalization.
- Nếu OOD vẫn có MaxSim tương đương in-domain và top-K không thỏa truy vấn, mô hình có forced retrieval và thiếu cơ chế abstention.
- Nếu OOD có điểm thấp rõ rệt, có thể xây dựng lớp từ chối bên ngoài AACL; đây không phải chức năng có sẵn của AACL nguyên bản.

## 5. Trực quan hóa attention flow

### 5.1. Tái hiện phương pháp của bài báo

Bài báo lấy trọng số $\alpha_i$ trong từng composition block, nhân qua các block để tạo attention flow của từng token, rồi chuẩn hóa min-max. Image token được ánh xạ về lưới 7×7 và phóng to để overlay lên ảnh nguồn. Text token có điểm cao được tô màu để minh họa từ được nhấn mạnh.

Implementation hiện tính `weights` trong `MultiHeadAdditiveAttention.forward()` nhưng chưa trả chúng ra ngoài. Cần bổ sung một đường inference tùy chọn để thu được tensor dạng:

```text
[batch, block, head, token]
```

Đường training mặc định không nên thay đổi output hoặc tạo thêm chi phí lưu trữ khi không yêu cầu attention.

### 5.2. Xử lý 98 image tokens

Image encoder hiện trả 98 tokens: 49 token từ Stage 3 và 49 token từ Stage 4, mỗi stage đã được pool về 7×7. Vì vậy:

- tạo heatmap Stage 3 riêng;
- tạo heatmap Stage 4 riêng;
- chuẩn hóa và average hai stage nếu muốn tạo một heatmap tổng;
- không reshape trực tiếp 98 token thành 7×14 rồi diễn giải như một feature map không gian duy nhất.

Đối với multi-head attention, báo cáo nên trình bày heatmap average-head và có thể thêm một phụ lục về head diversity. Đối với phép nhân flow qua block, nên dùng log-space để tránh underflow:

$$
\log f_i=\sum_{l=1}^{L}\log(\alpha_i^{(l)}+\epsilon).
$$

Padding token phải được loại trước khi chuẩn hóa. WordPiece của DistilBERT cần được gộp lại khi tô màu từ.

### 5.3. Visualize sau training hay trong training?

Thực nghiệm chính nên chạy **sau khi huấn luyện xong**, sử dụng `best.pt`, `model.eval()` và tắt gradient. Attention là hàm của cả checkpoint và cặp ảnh–prompt, nên phải chạy inference trên một probe set cụ thể.

Visualize trong training chỉ nên dùng để theo dõi động học:

- cố định 5–10 ảnh probe và prompt;
- lưu checkpoint theo chu kỳ đánh giá, ví dụ mỗi 5 epoch;
- so sánh attention entropy, image/text attention mass và heatmap qua epoch;
- không cần capture ở mọi batch.

Nếu thời gian hạn chế, chỉ visualize `best.pt` là đủ cho báo cáo chính. Nếu còn thời gian, trajectory qua epoch là một phân tích bổ sung có giá trị và giảm nguy cơ chọn hình thuận mắt.

## 6. Kiểm chứng global context bằng nhiều nguồn bằng chứng

### 6.1. Counterfactual text trên cùng ảnh

Với cùng ảnh nguồn, dùng các prompt chỉ khác một thuộc tính, ví dụ:

- “has longer sleeves” và “has a different graphic”;
- “is red” và “is blue”;
- “dress is longer” và “dress has longer sleeves”;
- “has a higher neckline” và “has a lower neckline”.

Đo đồng thời:

- Jensen–Shannon divergence hoặc correlation giữa các attention map;
- cosine distance giữa query embeddings;
- top-K overlap;
- thay đổi rank của target phù hợp từng prompt;
- mức dịch chuyển attention đến vùng được kỳ vọng.

Heatmap thay đổi nhưng ranking không đổi chỉ cho thấy attention biến thiên; chưa chứng minh tín hiệu đó được downstream retrieval sử dụng hiệu quả.

### 6.2. Occlusion test

Quy trình:

1. Lấy top-p phần trăm pixel hoặc patch có attention cao nhất.
2. Che vùng đó bằng màu trung bình hoặc blur.
3. Tính lại similarity với target và rank của target.
4. Lặp lại với vùng attention thấp có cùng diện tích.
5. So sánh trên toàn bộ probe set.

Đại lượng chính:

$$
\Delta_{high}=s(q,y)-s(q_{occlude-high},y),
$$

$$
\Delta_{low}=s(q,y)-s(q_{occlude-low},y).
$$

Nếu $\Delta_{high}>\Delta_{low}$ ổn định và có khoảng tin cậy hợp lý, attention map có độ trung thực tốt hơn. Nên báo bootstrap 95% confidence interval hoặc kiểm định Wilcoxon vì các mẫu được ghép cặp.

### 6.3. Context intervention và ablation

Các biến thể ưu tiên:

| Biến thể | Cách thực hiện | Điều được kiểm tra |
|---|---|---|
| Full AACL | Learned additive attention | Mốc chính |
| Uniform context | Thay $\alpha_i$ bằng trọng số đều | Giá trị của learned attention |
| Image-only context | Context chỉ từ image tokens | Đóng góp của text vào context |
| Text-only context | Context chỉ từ text tokens | Đóng góp của ảnh vào context |
| Shuffled context | Tráo $c$ giữa các mẫu khi inference | Context có phụ thuộc đúng query không |
| Additive → dot-product | Huấn luyện variant tương ứng | So sánh cơ chế attention |
| Hadamard → addition | Huấn luyện variant tương ứng | Vai trò tương tác phi tuyến |

Các thay đổi kiến trúc lớn nên được huấn luyện lại với cùng encoder, optimizer, epoch và seed. `Shuffled context` là intervention có thể chạy sau training để đo mức phụ thuộc của checkpoint vào context đúng.

Khi tài nguyên có hạn, ưu tiên theo thứ tự:

1. Full AACL;
2. shuffled context inference;
3. uniform-context model;
4. image-only/text-only;
5. dot-product và các variant còn lại.

### 6.4. Preservation và compliance

Tuyên bố “global context” của AACL không chỉ có nghĩa mô hình nhìn toàn ảnh. Mục tiêu thực tế là thay đổi đúng phần được yêu cầu và bảo toàn phần không được nhắc tới. Vì vậy, bảng kết quả global context phải đặt cạnh nhau:

- Recall@10 và Recall@50;
- qualitative preservation/compliance trên các ca minh họa;
- occlusion faithfulness;
- context-ablation performance drop.

## 7. Cấu hình thực nghiệm với hai NVIDIA L40

Hai GPU L40 khoảng 45 GiB VRAM mỗi GPU là đủ mạnh để chạy hai category độc lập song song với AMP và batch size hiện tại. Code hiện chưa triển khai DistributedDataParallel, vì vậy phương án ít rủi ro nhất là:

- GPU vật lý 0 chạy một category;
- GPU vật lý 1 chạy category khác;
- sau khi một job xong, GPU đó chạy category còn lại hoặc ablation.

Không nên khởi chạy thêm job trước khi xác định khoảng 18–20 GiB đang được tiến trình nào giữ. GPU utilization 0% nhưng VRAM đã cấp phát có thể là tiến trình đang chờ, data loading, notebook giữ model hoặc job treo.

Đối với các bảng so sánh khoa học, giữ nguyên:

- train/validation split;
- pretrained backbone;
- seed;
- optimizer và learning-rate schedule;
- số epoch;
- tiêu chí chọn best checkpoint;
- candidate gallery và cách loại query image.

Nếu không đủ thời gian chạy ba seed, báo rõ giới hạn và dùng bootstrap confidence interval trên query validation thay vì tạo độ lệch chuẩn giả.

## 8. Bộ bằng chứng tối thiểu cho báo cáo

Một bộ thực nghiệm tối thiểu nhưng đủ thuyết phục gồm:

1. Vocabulary audit chứng minh “T-shirt” có hay không có trong train.
2. Một lexical-holdout experiment hoặc một từ thực sự chưa xuất hiện trong fine-tuning.
3. Raw retrieval cho bộ probe cố định và một hình hallucination định tính dùng cùng `q01` trên ba category.
4. Attention flow của ít nhất một ảnh với hai counterfactual prompts.
5. Occlusion high-attention so với low-attention nếu đủ thời gian.
6. Ít nhất một context intervention nếu đủ thời gian.

## 9. Cách diễn đạt kết luận

Nếu kết quả ủng hộ các giả thuyết, kết luận nên giới hạn ở mức sau:

> AACL không hallucinate theo nghĩa sinh ảnh mới, nhưng có thể biểu hiện forced retrieval và semantic overreach vì luôn xếp hạng một gallery đóng và không có cơ chế từ chối. Các thí nghiệm counterfactual, occlusion và context intervention cho thấy global context được sử dụng hữu ích để điều kiện hóa retrieval theo phản hồi văn bản. Tuy nhiên, attention visualization không đủ để khẳng định mô hình hiểu ngữ cảnh theo nghĩa con người; đây là bằng chứng về cơ chế điều kiện hóa và mức độ sử dụng context, không phải một explanation nhân quả hoàn chỉnh.

Nếu occlusion hoặc ablation không ủng hộ attention map, cần báo cáo trung thực:

> Attention map thay đổi theo prompt nhưng chưa cho thấy độ trung thực nhân quả ổn định. Vì vậy, visualization có giá trị minh họa hành vi, còn tuyên bố về global-context understanding cần được giới hạn dựa trên Recall, preservation và ablation.
