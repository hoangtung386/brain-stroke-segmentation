# Brain Stroke Segmentation - LCNN Architecture

A deep learning project for brain stroke lesion segmentation using **LCNN (Local-Global Combined Neural Network)**, incorporating **SEAN (Symmetry Enhanced Attention Network)** and **ResNeXt50**. This architecture is designed to capture both fine-grained local details and global semantic context, leveraging the inherent symmetry of the brain to improve segmentation accuracy.

![Architecture Overview](https://img.shields.io/badge/Architecture-LCNN%20%2B%20SEAN-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76b900)
[![Acknowledgments](https://img.shields.io/badge/ACKNOWLEDGMENTS-Contributors-orange?style=flat-square&logo=open-source-initiative)](./ACKNOWLEDGMENTS.md)

## 🌟 Model Architecture

![Architectural Model](./Architectural_model.png)

This diagram illustrates the **LCNN (Local-Global Combined Network)** architecture used in this project. It highlights the integration of our **Symmetry Enhanced Attention Network (SEAN)** for capturing local, contralateral features and **ResNeXt50** for global semantic understanding. The parallel processing pathways ensure both fine-grained lesion details and broader context are preserved for accurate segmentation.

*   **Symmetry Enhanced Attention (SEAN)**: Exploits the bilateral symmetry of the human brain. An Alignment Network aligns the input slices, allowing the model to compare features from the contralateral hemisphere to identify anomalies.
*   **Dual-Path Architecture**:
    *   **Local Path (SEAN)**: Processes 3D stacks of adjacent CT slices to capture volumetric context and details.
    *   **Global Path (ResNeXt50)**: Extracts high-level semantic features to reduce false positives.
*   **Combined Loss Function**: Optimized hybrid loss combining:
    *   **Dice Loss**: Handles severe class imbalance (small stroke lesions).
    *   **Cross Entropy Loss**: Ensures pixel-level classification accuracy.
    *   **Alignment Loss**: Enforces symmetry alignment in the SEAN module.

## 📂 Project Structure

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
│   └── global_path.py        # ResNeXt Global Path
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Plotting and overlay tools
│   └── metrics.py            # Dice, IoU, Precision metrics
│
├── data/                     # Dataset storage
│   ├── images/               # CT Images
│   └── masks/                # Segmentation Masks
│
├── checkpoints/              # Checkpoint storage
└── outputs/                  # Logs, charts, and visualizations
```

## 💻 System Requirements

*   **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent recommended.
*   **OS**: Linux (tested on Ubuntu 20.04/22.04).
*   **CUDA**: Version 11.7 or higher.
*   **Python**: 3.8+.

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hoangtung386/brain-stroke-segmentation.git
cd brain-stroke-segmentation
```

### 2. Auto Setup (Recommended)
This script sets up a virtual environment, installs PyTorch (CUDA-optimized), and dependencies.
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Manual Setup (Alternative)
```bash
# Create environment
conda create -n stroke_seg_env python=3.12 -y
conda activate stroke_seg_env

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Preparation

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

## 🤖 Training

### 1. W&B Configuration (optional)

If you want to use Weights & Biases for tracking:

```bash
wandb login
````

Alternatively, set `USE_WANDB = False` in `config.py`

### 2. Configuration
Edit `config.py` to adjust hyperparameters if needed:
```python
BATCH_SIZE = 32         # Adjust based on VRAM (use 8-16 for 3090 if OOM)
NUM_EPOCHS = 60         # Total training epochs
LEARNING_RATE = 1e-3    # Initial learning rate
NORMALIZE = True        # Ensure this matches your data stats
```

> **Tip**: Before training, verify normalization stats:
> `python -c "from config import Config; Config.compute_normalization_stats(Config.IMAGE_DIR)"`

### 3. Start Training
- Start new training
```bash
python train.py
```

- Resume from checkpoint
```bash
python train.py --checkpoint checkpoints/last_checkpoint.pth
```
Training logs, best models (`best_model.pth`), and history (`training_history.csv`) will be saved to `outputs/`.

### 4. Monitoring
*   **Console**: Live metrics (Loss, Dice, LR).
*   **Weights & Biases**: If enabled in `config.py` (`USE_WANDB = True`), run `wandb login` first.

## 📉 Evaluation

Evaluate the trained model on the validation set to generate metrics and visual overlays.

```bash
# Evaluate best model
python evaluate.py --checkpoint checkpoints/best_model.pth --num-samples 30

# Output Report: outputs/evaluation_report.txt
# Visualizations: outputs/overlay_sample_*.png
```

## ⚡ Optimization Tips (RTX 3090)

*   **Mixed Precision**: The trainer uses `torch.cuda.amp` by default for faster training and lower memory usage.
*   **Data Loading**: Set `NUM_WORKERS = 4` or `8` in `config.py` for optimal data throughput. `PIN_MEMORY = True` is enabled by default.
*   **Out of Memory (OOM)**:
    *   Reduce `BATCH_SIZE` to 16, 8, or 4.
    *   Reduce `IMAGE_SIZE` to `(256, 256)` in `config.py`.

## 🛠️ Troubleshooting

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| **Loss is NaN** | Exploding gradients or bad normalization | Check dataset stats; enable gradient clipping (default in trainer). |
| **Dice Score ~0** | Model learning only background | Check class weights in `trainer.py`; verify mask values are 0/1. |
| **Dimension Errors** | Mismatch in 2D vs 3D shapes | Architecture fix applied; `dataset.py` now handles 3D stacks correctly. |
| **CUDA OOM** | Batch size too large | Decrease `BATCH_SIZE`; use `nvidia-smi` to check VRAM. |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

---

## ✉️ Contact

**Author**: Le Vu Hoang Tung  
**Email**: levuhoangtung1542003@gmail.com  
**GitHub**: [@hoangtung386](https://github.com/hoangtung386)       
**X**: [@hoangtung386](https://x.com/hoangtung386) 

If you encounter any issues, please create an issue on GitHub or contact via email.
