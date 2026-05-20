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

Evaluate cả ba category với cùng checkpoint chỉ phù hợp nếu checkpoint đó được train theo cách bạn muốn dùng chung. Thực tế nên evaluate từng checkpoint theo từng category:

```bash
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/dress/best.pt --category dress
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/shirt/best.pt --category shirt
python evaluate.py --config configs/fashioniq.yaml --checkpoint outputs/fashioniq_improved/toptee/best.pt --category toptee
```

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
