| Category | Best epoch | R@10 | R@50 |
|---|---:|---:|---:|
| Dress | 60 | 19.6331 | 44.7695 |
| Shirt | 45 | 16.8302 | 40.7753 |
| Toptee | 55 | 23.0495 | 50.0765 |
| **Macro average** | — | **19.8376** | **45.2071** |

Ghi chú: Kết quả được lấy từ best checkpoint riêng của từng category. Cấu hình dùng
micro-batch 8, gradient accumulation 4 và seed 42. Gradient accumulation tạo optimizer
effective batch 32 nhưng batch-softmax loss chỉ sử dụng 8 mẫu trong mỗi forward.
