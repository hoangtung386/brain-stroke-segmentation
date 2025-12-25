# Brain Stroke Segmentation - LCNN Architecture

A deep learning project for brain stroke lesion segmentation using **LCNN (Local-Global Combined Neural Network)**, incorporating **SEAN (Symmetry Enhanced Attention Network)** and **ResNeXt50**. This architecture is designed to capture both fine-grained local details and global semantic context, leveraging the inherent symmetry of the brain to improve segmentation accuracy.

![Architecture Overview](https://img.shields.io/badge/Architecture-LCNN%20%2B%20SEAN-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76b900)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/hoangtung386/brain-stroke-lcnn)
[![Acknowledgments](https://img.shields.io/badge/ACKNOWLEDGMENTS-Contributors-orange?style=flat-square&logo=open-source-initiative)](./ACKNOWLEDGMENTS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE.txt)

## Quick Links 🎯

- 🤗 **[Pre-trained Model on Hugging Face](https://huggingface.co/hoangtung386/brain-stroke-lcnn)**
- 📊 **[Dataset Information](#-data-preparation)**
- 🚀 **[Quick Start Guide](#-installation)**
- 📖 **[Documentation](#-project-structure)**
- 💬 **[Contact & Support](#-contact)**

---

## Model Architecture ❄️

![Architectural Model](./Architectural_model.png)

This diagram illustrates the **LCNN (Local-Global Combined Network)** architecture used in this project. It highlights the integration of our **Symmetry Enhanced Attention Network (SEAN)** for capturing local, contralateral features and **ResNeXt50** for global semantic understanding. The parallel processing pathways ensure both fine-grained lesion details and broader context are preserved for accurate segmentation.

### Key Features

*   **Symmetry Enhanced Attention (SEAN)**: Exploits the bilateral symmetry of the human brain. An Alignment Network aligns the input slices, allowing the model to compare features from the contralateral hemisphere to identify anomalies.
*   **Dual-Path Architecture**:
    *   **Local Path (SEAN)**: Processes 3D stacks of adjacent CT slices to capture volumetric context and details.
    *   **Global Path (ResNeXt50)**: Extracts high-level semantic features to reduce false positives.
*   **Combined Loss Function**: Optimized hybrid loss combining:
    *   **Dice Loss**: Handles severe class imbalance (small stroke lesions).
    *   **Cross Entropy Loss**: Ensures pixel-level classification accuracy.
    *   **Alignment Loss**: Enforces symmetry alignment in the SEAN module.

---

## Pre-trained Model 🤗

The trained model is available on Hugging Face Hub for easy download and inference:

**🔗 Model Repository**: [hoangtung386/brain-stroke-lcnn](https://huggingface.co/hoangtung386/brain-stroke-lcnn)

### Download Pre-trained Weights

```python
from huggingface_hub import hf_hub_download

# Download best model checkpoint
model_path = hf_hub_download(
    repo_id="hoangtung386/brain-stroke-lcnn",
    filename="best_model.pth"
)

# Load model
import torch
from models.lcnn import LCNN

model = LCNN(num_channels=1, num_classes=2, T=1)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Quick Inference Example

```python
import torch
from PIL import Image
from torchvision import transforms

# Load and preprocess image
image = Image.open('path/to/ct_scan.png').convert('L')
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2162], std=[0.1841])
])

# Create 3-slice stack (2T+1 where T=1)
image_tensor = transform(image).unsqueeze(0)  # (1, 512, 512)
image_stack = image_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, 512, 512)

# Inference
with torch.no_grad():
    output = model(image_stack)
    prediction = torch.argmax(output, dim=1)

print(f"Prediction shape: {prediction.shape}")
```

---

## Project Structure 📂

```bash
brain-stroke-segmentation/
│
├── config.py                 # Central configuration
├── dataset.py                # Dataset and DataLoader (handles 3D slice stacking)
├── download_dataset.py       # Data download utility
├── trainer.py                # Training loop and CombinedLoss implementation
├── train.py                  # Main entry point for training
├── evaluate.py               # Validation and evaluation script
├── setup.sh                  # Application environment setup
├── requirements.txt          # Python dependencies
│
├── models/
│   ├── __init__.py
│   ├── lcnn.py               # LCNN Main Architecture
│   ├── sean.py               # SEAN + Alignment Network
│   ├── global_path.py        # ResNeXt Global Path
│   └── components.py         # Shared components
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Plotting and overlay tools
│   ├── metrics.py            # Dice, IoU, Precision metrics
│   └── improved_alignment_loss.py  # Advanced loss functions
│
├── data/                     # Dataset storage
│   ├── images/               # CT Images
│   └── masks/                # Segmentation Masks
│
├── checkpoints/              # Model checkpoints
├── outputs/                  # Logs, charts, and visualizations
│
├── ACKNOWLEDGMENTS.md        # Detailed acknowledgments
├── LICENSE.txt               # MIT License
└── README.md                 # This file
```

---

## System Requirements 💻

*   **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent recommended.
*   **OS**: Linux (tested on Ubuntu 20.04/22.04).
*   **CUDA**: Version 11.7 or higher.
*   **Python**: 3.8+.

---

## Installation 🚀

### Option 1: Auto Setup (Recommended)

This script sets up a virtual environment, installs PyTorch (CUDA-optimized), and dependencies.

```bash
# Clone repository
git clone https://github.com/hoangtung386/brain-stroke-segmentation.git
cd brain-stroke-segmentation

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# Clone repository
git clone https://github.com/hoangtung386/brain-stroke-segmentation.git
cd brain-stroke-segmentation

# Create environment
conda create -n stroke_seg_env python=3.12 -y
conda activate stroke_seg_env

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Use Pre-trained Model from Hugging Face

```bash
# Install Hugging Face Hub
pip install huggingface-hub

# Download model in Python (see example above)
```

---

## Data Preparation 📊

The model requires a specific directory structure. You can download the dataset automatically or organize your own.

### Option A: Automatic Download

```bash
python download_dataset.py
```

### Option B: Custom Data

Organize your data as follows:

```text
data/
├── images/
│   ├── patient_001/
│   │   ├── 001.png
│   │   ├── 002.png
│   │   └── ...
│   ├── patient_002/
│   └── ...
└── masks/
    ├── patient_001/
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    ├── patient_002/
    └── ...
```

---

## Training 🤖

### 1. Configure Weights & Biases (Optional)

If you want to use Weights & Biases for tracking:

```bash
wandb login
```

Alternatively, set `USE_WANDB = False` in `config.py`

### 2. Adjust Configuration

Edit `config.py` to adjust hyperparameters if needed:

```python
BATCH_SIZE = 32         # Adjust based on VRAM (use 8-16 for 3090 if OOM)
NUM_EPOCHS = 60         # Total training epochs
LEARNING_RATE = 1e-3    # Initial learning rate
NORMALIZE = True        # Ensure this matches your data stats
```

> **Tip**: Before training, verify normalization stats:
> ```bash
> python -c "from config import Config; Config.compute_normalization_stats(Config.IMAGE_DIR)"
> ```

### 3. Start Training

**New training:**
```bash
python train.py
```

**Resume from checkpoint:**
```bash
python train.py --checkpoint checkpoints/last_checkpoint.pth
```

Training logs, best models (`best_model.pth`), and history (`training_history.csv`) will be saved to `outputs/`.

### 4. Monitoring

*   **Console**: Live metrics (Loss, Dice, LR).
*   **Weights & Biases**: If enabled in `config.py` (`USE_WANDB = True`), run `wandb login` first.

---

## Evaluation 📉

Evaluate the trained model on the validation set to generate metrics and visual overlays.

```bash
# Evaluate best model
python evaluate.py --checkpoint checkpoints/best_model.pth --num-samples 30

# Evaluate Hugging Face model
python evaluate.py --checkpoint path/to/downloaded/best_model.pth --num-samples 30
```

**Output:**
- Report: `outputs/evaluation_report.txt`
- Visualizations: `outputs/overlay_sample_*.png`

---

## Optimization Tips (RTX 3090) ⚡

*   **Mixed Precision**: The trainer uses `torch.cuda.amp` by default for faster training and lower memory usage.
*   **Data Loading**: Set `NUM_WORKERS = 4` or `8` in `config.py` for optimal data throughput. `PIN_MEMORY = True` is enabled by default.
*   **Out of Memory (OOM)**:
    *   Reduce `BATCH_SIZE` to 16, 8, or 4.
    *   Reduce `IMAGE_SIZE` to `(256, 256)` in `config.py`.

---

## Troubleshooting 🛠️

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| **Loss is NaN** | Exploding gradients or bad normalization | Check dataset stats; enable gradient clipping (default in trainer). |
| **Dice Score ~0** | Model learning only background | Check class weights in `trainer.py`; verify mask values are 0/1. |
| **Dimension Errors** | Mismatch in 2D vs 3D shapes | Architecture fix applied; `dataset.py` now handles 3D stacks correctly. |
| **CUDA OOM** | Batch size too large | Decrease `BATCH_SIZE`; use `nvidia-smi` to check VRAM. |

---

## Citation 📚

If you use this work in your research, please cite:

```bibtex
@software{le_vu_hoang_tung_2025_brain_stroke_lcnn,
  author = {Le Vu Hoang Tung},
  title = {Brain Stroke Segmentation using LCNN Architecture},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/hoangtung386/brain-stroke-segmentation},
  note = {Pre-trained model: \url{https://huggingface.co/hoangtung386/brain-stroke-lcnn}}
}
```

---

## License 📄

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

---

## Contact ✉️

**Author**: Le Vu Hoang Tung  
**Email**: levuhoangtung1542003@gmail.com  
**GitHub**: [@hoangtung386](https://github.com/hoangtung386)       
**X (Twitter)**: [@hoangtung386](https://x.com/hoangtung386)  
**Hugging Face**: [@hoangtung386](https://huggingface.co/hoangtung386)

If you encounter any issues, please create an issue on GitHub or contact via email.

---

## Acknowledgments 🤝🏻

See [ACKNOWLEDGMENTS.md](./ACKNOWLEDGMENTS.md) for detailed acknowledgments including:
- Primary author contributions
- AI assistant support (debugging only)
- Technologies and frameworks used
- Research foundations

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hoangtung386/brain-stroke-segmentation&type=Date)](https://star-history.com/#hoangtung386/brain-stroke-segmentation&Date)
