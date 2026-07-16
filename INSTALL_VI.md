# Hướng dẫn cài đặt và khởi chạy AACL Fashion Retrieval

Tài liệu này dùng để đưa project sang một máy khác có GPU mạnh và chạy training/evaluation cho mô hình Additive Attention Compositional Learning (AACL) trên FashionIQ.

## 1. Yêu cầu hệ thống

- Python: khuyến nghị 3.10 hoặc 3.11. Project có thể chạy Python 3.9+, nhưng nên tránh dùng bản quá mới nếu server CUDA/PyTorch chưa hỗ trợ tốt.
- GPU: NVIDIA GPU có CUDA. Khuyến nghị từ 16 GB VRAM trở lên cho Swin-Base + DistilBERT.
- RAM: khuyến nghị 32 GB trở lên.
- Disk: cần đủ chỗ cho FashionIQ, pretrained weights cache và checkpoint trong `outputs/`.
- Internet lần đầu để tải pretrained weights từ HuggingFace và `timm`, trừ khi bạn đã copy sẵn cache.

Kiểm tra driver NVIDIA:

```bash
nvidia-smi
```

Nếu lệnh này không chạy hoặc không thấy GPU, cần cài NVIDIA driver trước khi train.

## 2. Tạo môi trường Python

### Linux GPU server

```bash
cd fashion-image-retrieval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows

```powershell
cd fashion-image-retrieval
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3. Cài PyTorch CUDA

Nên cài PyTorch theo đúng CUDA/driver của máy từ trang chính thức: <https://pytorch.org/get-started/locally/>.

Ví dụ với CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Sau đó cài các thư viện còn lại:

```bash
pip install -r requirements.txt
```

Kiểm tra CUDA trong PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Kết quả mong muốn là `True` và tên GPU.

## 4. Kiểm tra setup bằng script

Chạy kiểm tra nhanh môi trường:

```bash
python scripts/check_setup.py --config configs/fashioniq.yaml
```

Nếu chưa có FashionIQ local, script sẽ báo thiếu `data/fashioniq`. Điều này bình thường cho đến khi bạn chuẩn bị dataset.

Kiểm tra cấu hình synthetic không cần dataset thật:

```bash
python scripts/check_setup.py --config configs/synthetic_smoke.yaml
```

## 5. Chuẩn bị FashionIQ

Project kỳ vọng cấu trúc:

```text
data/fashioniq/
  images/
    <image_id>.jpg
  captions/
    cap.dress.train.json
    cap.dress.val.json
    cap.shirt.train.json
    cap.shirt.val.json
    cap.toptee.train.json
    cap.toptee.val.json
  image_splits/
    split.dress.train.json
    split.dress.val.json
    split.shirt.train.json
    split.shirt.val.json
    split.toptee.train.json
    split.toptee.val.json
```

Loader cũng chấp nhận một số biến thể tên file như `{category}.{split}.json` hoặc `{split}.{category}.json`, nhưng tên chính thức ở trên là lựa chọn nên dùng.

Project không tự tải FashionIQ vì ảnh thường cần crawl từ nguồn ngoài và phụ thuộc quyền truy cập. Hãy chuẩn bị dataset trước rồi đặt vào `data/fashioniq`.

## 6. Chạy smoke test

Smoke test dùng dữ liệu giả và mock encoder, không cần FashionIQ, `timm`, hoặc pretrained HuggingFace weights:

```bash
python train.py --config configs/synthetic_smoke.yaml --category dress
python evaluate.py --config configs/synthetic_smoke.yaml --checkpoint outputs/synthetic_smoke/dress/best.pt --category dress
python -m unittest discover -s tests
```

Nếu các lệnh này chạy được, train/eval loop cơ bản đang ổn.

## 7. Train baseline và improved

Train baseline AACL reproduction:

```bash
python train.py --config configs/fashioniq_baseline.yaml --category dress
```

Train improved config với AdamW + cosine schedule + symmetric InfoNCE + dropout + label smoothing:

```bash
python train.py --config configs/fashioniq.yaml --category dress
```

Nếu NVIDIA L40 đang được chia sẻ và chỉ còn khoảng 25 GiB VRAM, dùng cấu hình thận trọng:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/fashioniq_l40_probe.yaml --category dress
```

Probe trên chạy một epoch và ghi peak VRAM vào `outputs/fashioniq_probe/l40_probe_seed42/dress/metrics.csv`.
Nếu peak reserved memory an toàn so với phần VRAM còn trống, bắt đầu full run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/fashioniq_l40_shared.yaml --category dress
```

Cấu hình này dùng micro-batch 8 và gradient accumulation 4. Optimizer effective batch là 32,
nhưng batch-softmax loss vẫn chỉ thấy 8 mẫu trong từng forward, tức 7 in-batch negatives cho
mỗi query. Cần ghi rõ khác biệt này khi so sánh với một lần chạy batch 32 thực.

Artifact được lưu tại:

```text
outputs/fashioniq_improved/l40_shared_seed42/dress/
  best.pt
  latest.pt
  metrics.csv
  metrics.jsonl
  config.resolved.yaml
  run.json
  run_summary.json
```

Pretrained weights được cache trong `.cache/huggingface/hub`, đường dẫn này đã bị loại khỏi Git.

Nếu job bị ngắt sau khi đã lưu `latest.pt`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/fashioniq_l40_shared.yaml --category dress --resume auto
```

Nếu chạy lại từ đầu vào đúng run directory, chương trình sẽ từ chối ghi đè. Chỉ dùng
`--overwrite` khi chủ động muốn thay các artifact cũ.

Train cả ba category:

```bash
python train.py --config configs/fashioniq.yaml --category all
```

Checkpoint được lưu theo category:

```text
outputs/fashioniq_improved/dress/best.pt
outputs/fashioniq_improved/dress/latest.pt
```

## 8. Evaluate checkpoint

Evaluate một category:

```bash
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/dress/best.pt --category dress
```

Xuất kết quả có cấu trúc để dùng trong báo cáo:

```bash
python evaluate.py \
  --config configs/fashioniq_l40_shared.yaml \
  --checkpoint outputs/fashioniq_improved/l40_shared_seed42/dress/best.pt \
  --category dress \
  --json-output outputs/fashioniq_improved/l40_shared_seed42/dress/evaluation.json \
  --per-query-output outputs/fashioniq_improved/l40_shared_seed42/dress/per_query.csv
```

File `per_query.csv` chứa rank của target, target score, top-1 ID, top-1 score và chênh lệch
top-1/top-2 cho từng query.

Evaluate cả ba category với cùng checkpoint chỉ phù hợp nếu checkpoint đó được train theo cách bạn muốn dùng chung. Thực tế nên evaluate từng checkpoint theo từng category:

```bash
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/dress/best.pt --category dress
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/shirt/best.pt --category shirt
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/toptee/best.pt --category toptee
```

### 8.1. Xem kết quả trực quan từ `best.pt`

Nếu bạn chỉ muốn xem metric tổng quát, dùng `evaluate.py` như phần trên. Kết quả sẽ in ra dạng:

```text
[dress] {'recall@10': 0.1234, 'recall@50': 0.4567}
```

Trong đó `recall@10` là tỉ lệ query có ảnh đích nằm trong 10 ảnh model trả về đầu tiên; `recall@50` tương tự với 50 ảnh đầu tiên.

Nếu muốn nhìn trực tiếp các ảnh được retrieve, dùng script demo:

```bash
python scripts/retrieve_demo.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/dress/best.pt --category dress --query-index 0 --top-k 10 --output outputs/retrieval_demo/dress/query0.jpg --json-output outputs/retrieval_demo/dress/query0.json
```

Script này sẽ:

- Nạp model từ `best.pt`.
- Lấy một query trong tập validation theo `--query-index`.
- Encode toàn bộ gallery validation của category tương ứng.
- In ra top-K ảnh gần nhất theo score.
- Tạo contact sheet tại đường dẫn `--output` để bạn mở lên xem query và các ảnh top-K.
- Nếu truyền `--json-output`, lưu thêm file JSON chứa `image_id`, `score`, và đường dẫn ảnh.

Ví dụ nếu file checkpoint của bạn đang nằm ngay trong thư mục project với tên `best.pt`:

```bash
python evaluate.py --config configs/fashioniq.yaml --checkpoint best.pt --category dress
python scripts/retrieve_demo.py --config configs/fashioniq.yaml --checkpoint best.pt --category dress --query-index 0 --top-k 10 --output outputs/retrieval_demo/dress/query0.jpg
```

Xem nhiều query khác nhau:

```bash
python scripts/retrieve_demo.py --config configs/fashioniq.yaml --checkpoint best.pt --category dress --query-index 1 --top-k 10 --output outputs/retrieval_demo/dress/query1.jpg
python scripts/retrieve_demo.py --config configs/fashioniq.yaml --checkpoint best.pt --category dress --query-index 2 --top-k 10 --output outputs/retrieval_demo/dress/query2.jpg
```

Dùng ảnh query tự chọn và câu mô tả tự nhập:

```bash
python scripts/retrieve_demo.py --config configs/fashioniq.yaml --checkpoint best.pt --category dress --query-image data/fashioniq/images/<image_id>.jpg --text "make it sleeveless and darker" --top-k 10 --output outputs/retrieval_demo/dress/custom.jpg
```

Lưu ý: `--category` phải khớp với category mà checkpoint đã train. Ví dụ checkpoint train bằng `dress` thì evaluate/demo với `--category dress`.

## 9. Chỉnh cấu hình theo GPU

Các tham số chính nằm trong `configs/fashioniq.yaml`:

- `training.batch_size`: giảm nếu bị CUDA out of memory.
- `training.grad_accumulation`: tăng để giữ effective batch size khi giảm batch size.
- `training.amp`: để `true` trên GPU để tiết kiệm VRAM.
- `dataset.num_workers`: tăng trên Linux server mạnh, ví dụ `4` đến `8`; giảm về `0` nếu DataLoader lỗi trên Windows.
- `evaluation.batch_size`: giảm nếu evaluate gallery bị OOM.

Ví dụ nếu GPU không đủ cho batch size 32:

```yaml
training:
  batch_size: 8
  grad_accumulation: 4
```

Effective batch size vẫn xấp xỉ 32.

## 10. Pretrained weights và cache

Lần đầu chạy config thật, code sẽ tải:

- Swin từ `timm`.
- DistilBERT từ HuggingFace.

Nếu server không có internet, chuẩn bị cache trước trên máy có internet rồi copy sang server. Các cache thường nằm ở:

```text
~/.cache/huggingface/
~/.cache/torch/hub/
```

## 11. Lỗi thường gặp

### `ModuleNotFoundError: No module named 'timm'`

Chưa cài dependency:

```bash
pip install -r requirements.txt
```

### `torch.cuda.is_available()` trả về `False`

Kiểm tra:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Nếu `nvidia-smi` thấy GPU nhưng PyTorch không thấy CUDA, cài lại PyTorch CUDA đúng phiên bản.

### Không tải được HuggingFace hoặc Swin weights

Máy cần internet lần đầu hoặc cần copy cache pretrained weights sang server. Có thể đặt biến môi trường cache nếu cần:

```bash
export HF_HOME=/path/to/hf_cache
export TORCH_HOME=/path/to/torch_cache
```

### Thiếu FashionIQ file hoặc image

Chạy:

```bash
python scripts/check_setup.py --config configs/fashioniq.yaml
```

Sau đó kiểm tra lại `data/fashioniq/images`, `data/fashioniq/captions`, `data/fashioniq/image_splits`.

### CUDA out of memory

Giảm:

```yaml
training:
  batch_size: 8
  grad_accumulation: 4

evaluation:
  batch_size: 16
```

Giữ `training.amp: true`.

## 12. Quy trình khuyến nghị trên máy GPU mới

```bash
cd fashion-image-retrieval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_setup.py --config configs/synthetic_smoke.yaml
python train.py --config configs/synthetic_smoke.yaml --category dress
python scripts/check_setup.py --config configs/fashioniq.yaml
python train.py --config configs/fashioniq.yaml --category dress
```

Sau khi `dress` chạy ổn, train tiếp `shirt`, `toptee` hoặc dùng `--category all`.
