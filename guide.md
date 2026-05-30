# Prompt: Triển khai & Cải tiến AACL — Fashion Image Retrieval with Text Feedback

---

## 🎯 Mục tiêu tổng thể

Tái hiện và cải tiến mô hình **Additive Attention Compositional Learning (AACL)** từ bài báo WACV 2023 của Tian et al., áp dụng trên dataset **FashionIQ** (ưu tiên) hoặc **Fashion200k**. Mục tiêu cuối cùng là vượt qua kết quả R@10 và R@50 gốc của AACL trên ít nhất một trong ba category của FashionIQ thông qua các cải tiến có căn cứ.

---

## 📚 Bối cảnh & Kiến trúc gốc cần tái hiện

### Bài toán
- **Input**: ảnh nguồn `x` + câu phản hồi văn bản `t` (ví dụ: "Dress is blue with higher neckline")
- **Output**: danh sách ảnh đích `y` trong database thỏa mãn sửa đổi mô tả trong `t`
- **Metric đánh giá**: Recall@K (K=10, K=50) — chuẩn của FashionIQ

### Pipeline kiến trúc AACL gốc
```
Image x ──► Swin Transformer (Stage 3+4) ──► φ_x ∈ R^{98×768}
                                                        │
Text t  ──► DistilBERT                  ──► φ_t ∈ R^{m×768}
                                                        │
                            Concat ──► φ_xt ∈ R^{N×768}
                                                        │
                    Additive Attention Composition Module
                    (L=3 blocks, 8 heads mỗi block)
                                                        │
                            o_xt ──► Batch Classification Loss
                                         (so sánh với φ_y)
```

### Công thức cốt lõi — Additive Self-Attention
```
h = F_h(φ_xt)                          # Linear projection
α_i = softmax(w_h^T · h_i / √d)        # Scalar attention weight
c = Σ α_i · h_i                        # Global context vector
v_i = c ⊙ h_i                          # Hadamard product
o_i = h_i + F_o(c ⊙ h_i)              # Residual output
```

### Kết quả baseline cần đạt (FashionIQ, same encoder setting)
| Category | R@10    | R@50    |
|----------|---------|---------|
| Shirt    | 24.82   | 48.85   |
| Dress    | 29.89   | 55.85   |
| Toptee   | 30.88   | 56.85   |
| **Avg**  | **41.19** | —     |

---

## 🛠️ Yêu cầu triển khai

### Bước 1 — Chuẩn bị môi trường & dữ liệu

```
Ngôn ngữ: Python 3.9+
Framework: PyTorch 2.x + HuggingFace Transformers
```

**FashionIQ dataset setup:**
- Tải ảnh từ Amazon (script crawl công khai trên GitHub: `XiaoxiaoGuo/fashion-iq`)
- Cấu trúc thư mục: `data/fashioniq/{images/, captions/, image_splits/}`
- Ba category: `{dress, shirt, toptee}`
- Mỗi cặp ảnh đi kèm 2 câu mô tả → dùng cả hai câu (concat hoặc chọn ngẫu nhiên khi train)

**DataLoader requirements:**
- Augmentation train: RandomHorizontalFlip, ColorJitter(0.4, 0.4, 0.4), RandomResizedCrop(224)
- Augmentation val: CenterCrop(224), Normalize(ImageNet mean/std)
- Batch size: 32 per GPU (tổng 128 với 4 GPU), hoặc accumulate gradient nếu ít GPU hơn

---

### Bước 2 — Tái hiện AACL chính xác

**2.1. Image Encoder — Swin Transformer**
```python
# Dùng swin_base_patch4_window7_224 từ timm
# Pre-trained: ImageNet-22K → fine-tune ImageNet-1K
# Trích token từ Stage 3 (49 tokens) + Stage 4 (49 tokens)
# → concat → 98 tokens × 1024-dim
# → Linear projection → 98 × 768
```

**2.2. Text Encoder — DistilBERT**
```python
# Từ HuggingFace: distilbert-base-uncased
# Output: hidden states layer cuối → m tokens × 768-dim
# Freeze hoặc fine-tune với lr nhỏ hơn (1/10 lr chính)
```

**2.3. Additive Attention Composition Module**
```python
class AdditiveAttentionBlock(nn.Module):
    # Input: φ_xt [B, N, d]  (N = 98 + m tokens)
    # Components:
    #   - Linear F_h: d → d
    #   - Learnable vector w_h: d
    #   - Linear F_o: d → d
    #   - LayerNorm + Residual connection
    #   - FFN: d → 4d → d
    # Stack L=3 blocks, 8 heads (tách d=768 thành 8 × 96-dim subspaces)
```

**2.4. Loss — Batch-based Classification**
```python
# Softmax cross-entropy trong batch
# κ(φ_y, o_xt) = dot product (chuẩn hóa L2 trước)
# Batch size B=32, mỗi sample là 1 positive pair
# Không dùng triplet loss (paper xác nhận kém hơn)
```

**2.5. Optimizer & Schedule**
```python
optimizer = SGD(lr=0.035, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(step_size=10, gamma=0.1)  # FashionIQ: 60 epochs
```

---

### Bước 3 — Các cải tiến đề xuất (chọn ≥ 3 để implement)

#### 🔵 Cải tiến 1: Thay SGD bằng AdamW + Cosine Annealing
**Lý do**: AdamW ổn định hơn với transformer-based model, cosine schedule tránh learning rate collapse.
```python
optimizer = AdamW([
    {'params': image_encoder.parameters(), 'lr': 1e-5},
    {'params': text_encoder.parameters(), 'lr': 1e-5},
    {'params': composition_module.parameters(), 'lr': 3e-4}
], weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=60, eta_min=1e-6)
```

#### 🔵 Cải tiến 2: Symmetric Contrastive Loss (InfoNCE hai chiều)
**Lý do**: Học cả chiều `(x,t) → y` và implicit `y → (x,t)` giúp embedding space cân bằng hơn.
```python
loss = (cross_entropy(logits, targets) + cross_entropy(logits.T, targets)) / 2
# Temperature τ learnable hoặc set = 0.07
```

#### 🔵 Cải tiến 3: Sử dụng cả 2 câu caption (FashionIQ có 2 câu/cặp)
**Lý do**: Paper gốc dùng ngẫu nhiên 1 câu, dùng cả 2 câu (concat hoặc ensemble) cung cấp thêm supervision.
```python
# Option A: Concat với separator token [SEP]
# Option B: Forward 2 lần, average loss
# Option C: Cross-attention giữa 2 câu trước khi encode
```

#### 🔵 Cải tiến 4: Thay DistilBERT bằng CLIP Text Encoder
**Lý do**: CLIP text encoder được pre-train trên dữ liệu image-text quy mô lớn, phù hợp với fashion domain hơn.
```python
from transformers import CLIPTextModel, CLIPTokenizer
# Model: openai/clip-vit-base-patch32
# Output: pooled_output hoặc sequence output → project → 768-dim
```

#### 🔵 Cải tiến 5: Feature-level Dropout & Label Smoothing
**Lý do**: Tăng regularization, tránh overfit trên FashionIQ (dataset nhỏ ~18k pairs).
```python
# Dropout(p=0.1) sau mỗi Additive Attention Block
# Label smoothing = 0.1 trong cross-entropy loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

#### 🔵 Cải tiến 6: Query Expansion tại Inference
**Lý do**: Average embedding của top-K kết quả vòng 1 với query gốc để re-rank.
```python
# 1. Retrieve top-K với o_xt
# 2. Average φ_y của top-K → φ_expanded
# 3. o_xt_final = α*o_xt + (1-α)*φ_expanded
# 4. Re-retrieve
```

---

### Bước 4 — Cấu trúc project

```
aacl_fashion/
├── configs/
│   ├── fashioniq.yaml          # Hyperparameters, paths
│   └── fashion200k.yaml
├── data/
│   ├── fashioniq_dataset.py    # Dataset class, DataLoader
│   └── transforms.py           # Augmentation pipelines
├── models/
│   ├── image_encoder.py        # Swin Transformer wrapper
│   ├── text_encoder.py         # DistilBERT / CLIP wrapper
│   ├── composition_module.py   # AdditiveAttentionBlock + stack
│   └── aacl.py                 # Full model, forward pass
├── losses/
│   └── batch_softmax.py        # Batch classification + InfoNCE
├── utils/
│   ├── metrics.py              # Recall@K computation
│   ├── logger.py               # WandB / TensorBoard logging
│   └── checkpoint.py           # Save/load model
├── train.py                    # Main training loop
├── evaluate.py                 # Inference + metric computation
├── requirements.txt
└── README.md
```

---

### Bước 5 — Training loop chi tiết

```python
# train.py skeleton
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        query_images, target_images, texts = batch
        
        # Forward
        phi_x = image_encoder(query_images)       # [B, 98, 768]
        phi_t = text_encoder(texts)               # [B, m, 768]
        phi_y = image_encoder(target_images)      # [B, 1, 768] → pool
        
        o_xt = composition_module(phi_x, phi_t)   # [B, 768]
        phi_y_pooled = pool(phi_y)                 # [B, 768]
        
        # Loss
        loss = batch_softmax_loss(o_xt, phi_y_pooled)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    scheduler.step()
    
    # Eval mỗi 5 epoch
    if epoch % 5 == 0:
        recall_at_10, recall_at_50 = evaluate(model, val_loader, gallery_loader)
        log(recall_at_10, recall_at_50)
```

---

### Bước 6 — Evaluation protocol

```python
# evaluate.py
# 1. Build gallery: encode toàn bộ ảnh trong val set → matrix [N_gallery, 768]
# 2. Với mỗi query (img, text):
#    - Tính o_xt
#    - Cosine similarity với gallery
#    - Kiểm tra target_id có trong top-K không
# 3. Report R@10, R@50 cho từng category + average

def recall_at_k(query_embeds, gallery_embeds, targets, k):
    sims = query_embeds @ gallery_embeds.T   # [Q, G]
    topk_indices = sims.topk(k, dim=1).indices
    hits = [t in topk_indices[i] for i, t in enumerate(targets)]
    return sum(hits) / len(hits) * 100
```

---

## 📊 Experiment tracking

Chạy **5 lần** mỗi config (như paper gốc), báo cáo **mean ± std**.

| Experiment | Thay đổi so với baseline | R@10 (Avg) | R@50 (Avg) |
|---|---|---|---|
| AACL-Repro | Tái hiện gốc | — | — |
| +AdamW | Cải tiến 1 | — | — |
| +SymLoss | Cải tiến 2 | — | — |
| +DualCaption | Cải tiến 3 | — | — |
| +CLIP-Text | Cải tiến 4 | — | — |
| **AACL-Improved** | Tất cả cải tiến | — | — |

---

## ⚠️ Lưu ý quan trọng

1. **Reproducibility**: Set `seed=42` cho `torch`, `numpy`, `random` và `torch.backends.cudnn.deterministic = True`
2. **Memory**: Swin-Base + DistilBERT cần ~16GB VRAM. Dùng `torch.cuda.amp` (mixed precision FP16) nếu ít VRAM hơn
3. **Gallery leakage**: Đảm bảo ảnh query KHÔNG nằm trong gallery khi evaluate
4. **FashionIQ split**: Dùng đúng `val_split` từ file JSON gốc, không tự chia
5. **Text preprocessing**: Lowercase, giữ dấu câu, max_length=77 (CLIP) hoặc 128 (BERT)
6. **Pooling target image**: Dùng mean pooling qua các token (không phải [CLS] đơn thuần)

---

## 🚀 Thứ tự ưu tiên thực thi

```
Phase 1 (tuần 1): Tái hiện AACL gốc → đạt ~41 R@10 avg trên FashionIQ
Phase 2 (tuần 2): Áp dụng Cải tiến 1 + 2 + 5 (ít rủi ro nhất)
Phase 3 (tuần 3): Thêm Cải tiến 3 + 4 (thay encoder)
Phase 4 (tuần 4): Ablation study + Query Expansion + viết báo cáo
```

---

*Prompt này đủ để một kỹ sư ML bắt đầu implement từ đầu mà không cần đọc lại paper gốc.*
