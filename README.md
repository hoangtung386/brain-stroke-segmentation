# Brain Stroke Segmentation - LCNN Architecture

Dự án phân đoạn vùng đột quỵ não sử dụng kiến trúc LCNN kết hợp SEAN (Symmetry Enhanced Attention Network) và ResNeXt50.

## Cấu trúc dự án

```
brain-stroke-segmentation/
│
├── config.py                 # Cấu hình dự án
├── dataset.py                # Dataset và DataLoader
├── download_dataset.py       # Download dataset
├── trainer.py                # Training logic
├── train.py                  # Script chính để train
├── evaluate.py               # Script đánh giá model
├── setup.sh                  # Script setup
├── requirements.txt          # Dependencies
├── README.md                 # File này
│
├── models/
│   ├── __init__.py
│   ├── components.py         # Các thành phần của model
│   ├── sean.py              # SEAN architecture
│   ├── global_path.py       # ResNeXt global path
│   └── lcnn.py              # LCNN main architecture
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py     # Visualization utilities
│   └── metrics.py           # Metrics computation
│
├── data/                     # Thư mục chứa dữ liệu
│   ├── image/               # CT images
│   └── mask/                # Segmentation masks
│
├── checkpoints/             # Thư mục lưu checkpoints
└── outputs/                 # Thư mục lưu kết quả
```

## Yêu cầu hệ thống

- **GPU**: NVIDIA RTX 3090 (24GB VRAM) trở lên
- **CUDA**: 11.7 hoặc cao hơn
- **Python**: 3.8+
- **RAM**: 32GB+ (khuyến nghị)

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/hoangtung386/brain-stroke-segmentation.git
cd brain-stroke-segmentation
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt (tự động / thủ công)

1. Setup tự động (dễ nhất)

```bash
chmod +x setup.sh
./setup.sh
```

2. Hoặc setup thủ công

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc PowerShell trên Windows
venv\Scripts\Activate.ps1  # PowerShell

# Cài đặt PyTorch cho CUDA 11.8 (ví dụ)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Cài dependencies
pip install -r requirements.txt

# Tạo thư mục dữ liệu và kết quả
mkdir -p data/image data/mask checkpoints outputs
```

3. Download the dataset for the project.

```bash
python download_dataset.py
```

Các options hữu ích:

```bash
# Hoặc giữ lại file ZIP sau khi giải nén
python download_dataset.py --keep-zip

# Hoặc không download lại nếu data đã tồn tại
python download_dataset.py --no-overwrite

# Hoặc custom Google Drive IDs
python download_dataset.py --image-id YOUR_ID --mask-id YOUR_ID
```

Notes:
- Nếu dùng Windows cmd hoặc PowerShell, thay `source` bằng `venv\\Scripts\\activate` hoặc `venv\\Scripts\\Activate.ps1`.
- `setup.sh` (nếu có) có thể tự động tạo virtualenv và cài dependencies; file này không được thêm tự động bởi script này — bạn có thể tạo nó theo ý muốn. 

## Chuẩn bị dữ liệu

### Cấu trúc dữ liệu

```
data/
├── image/
│   ├── patient_001/
│   │   ├── slice_001.png
│   │   ├── slice_002.png
│   │   └── ...
│   ├── patient_002/
│   └── ...
└── mask/
    ├── patient_001/
    │   ├── slice_001.png
    │   ├── slice_002.png
    │   └── ...
    ├── patient_002/
    └── ...
```

### Cấu hình đường dẫn

Mở file `config.py` và chỉnh sửa:

```python
BASE_PATH = '/path/to/your/data'  # Thay đổi đường dẫn này
IMAGE_DIR = os.path.join(BASE_PATH, 'image')
MASK_DIR = os.path.join(BASE_PATH, 'mask')
```

## Training

### 1. Cấu hình W&B (optional)

Nếu muốn sử dụng Weights & Biases để tracking:

```bash
wandb login
```

Hoặc đặt `USE_WANDB = False` trong `config.py`

### 2. Chỉnh sửa hyperparameters

Trong file `config.py`, bạn có thể điều chỉnh:

```python
BATCH_SIZE = 32          # Giảm nếu bị out of memory
NUM_EPOCHS = 60         # Số epochs
LEARNING_RATE = 1e-3    # Learning rate
NUM_WORKERS = 4         # Số workers cho DataLoader
```

### 3. Training

1. Chuẩn bị

- Chỉnh sửa `config.py` (điều chỉnh `BASE_PATH`, `BATCH_SIZE`, `NUM_EPOCHS`, ...)
- Đảm bảo dữ liệu đã có trong `data/image` và `data/mask` (hoặc cập nhật `BASE_PATH`)

2. Chạy training

```bash
python train.py
```

3. Resume training từ checkpoint

```bash
# Nếu script tìm thấy checkpoint trong `checkpoints/` nó sẽ resume tự động
# Hoặc chỉ định checkpoint cụ thể
python train.py --checkpoint checkpoints/checkpoint.pth
```

### 4. Resume training từ checkpoint

Script sẽ tự động resume nếu phát hiện checkpoint trong thư mục `checkpoints/`

### 5. Monitor training

- **Console**: Xem metrics trực tiếp trên terminal
- **W&B**: Truy cập dashboard tại https://wandb.ai
- **CSV**: File `outputs/training_history.csv`

## Đánh giá model

### 6. Evaluation

Đánh giá best model:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --num-samples 5
```

Hoặc đánh giá checkpoint cụ thể:

```bash
python evaluate.py --checkpoint checkpoints/checkpoint.pth
```

## Tối ưu cho RTX 3090

### Memory optimization

1. **Giảm batch size** nếu gặp OOM:
```python
BATCH_SIZE = 4  # trong config.py
```

2. **Gradient accumulation**:
```python
# Thêm vào trainer.py
accumulation_steps = 4
for i, (images, masks) in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Mixed precision training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Speed optimization

1. **Tăng num_workers**:
```python
NUM_WORKERS = 8  # Tùy CPU của bạn
```

2. **Pin memory**:
```python
PIN_MEMORY = True
PERSISTENT_WORKERS = True
```

3. **Benchmark mode**:
```python
torch.backends.cudnn.benchmark = True
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Giảm batch size
BATCH_SIZE = 4

# Hoặc giảm image size
IMAGE_SIZE = (256, 256)

# Clear cache
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Slow data loading

```python
# Tăng số workers
NUM_WORKERS = 8

# Sử dụng caching
CACHE_RATE = 0.5  # Cache 50% dữ liệu vào RAM
```

### CUDA out of memory

```bash
# Kiểm tra GPU usage
nvidia-smi

# Kill các process đang dùng GPU
kill -9 <PID>
```

## Kết quả

Model sẽ lưu:
- **Checkpoints**: `checkpoints/checkpoint.pth`
- **Best model**: `checkpoints/best_model.pth`
- **Training history**: `outputs/training_history.csv`
- **Visualizations**: `outputs/*.png`

## License

MIT License

## Liên hệ

Nếu có vấn đề, vui lòng tạo issue trên GitHub hoặc liên hệ: levuhoangtung1542003@gmail.com
