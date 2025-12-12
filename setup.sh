#!/bin/bash

# Setup script for Brain Stroke Segmentation project
# Run this on your RTX 3090 workstation

echo "=========================================="
echo "Brain Stroke Segmentation - Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 is not installed. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}Python3 found: $(python3 --version)${NC}"

# Check if CUDA is available
echo -e "\n${YELLOW}Checking CUDA installation...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA found:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo -e "${RED}CUDA not found. Please install CUDA drivers.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch with CUDA support
echo -e "\n${YELLOW}Installing PyTorch with CUDA support...${NC}"
echo "This may take a few minutes..."

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
echo "Detected CUDA version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12.2" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${YELLOW}Installing PyTorch with default CUDA support...${NC}"
    pip install torch torchvision
fi

# Install other requirements
echo -e "\n${YELLOW}Installing other dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p data/image data/mask
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs
echo -e "${GREEN}Directories created.${NC}"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python3 << END
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
END

# Check if data directory is empty
echo -e "\n${YELLOW}Checking data directory...${NC}"
if [ -z "$(ls -A data/image)" ]; then
    echo -e "${YELLOW}Data directory is empty.${NC}"
    echo -e "Please copy your data to:"
    echo -e "  - Images: ${GREEN}data/image/${NC}"
    echo -e "  - Masks:  ${GREEN}data/mask/${NC}"
else
    echo -e "${GREEN}Data found in data directory.${NC}"
fi

# Display next steps
echo -e "\n=========================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "=========================================="
echo -e "\nNext steps:"
echo -e "1. Activate virtual environment: ${GREEN}source venv/bin/activate${NC}"
echo -e "2. Edit config.py to set data paths"
echo -e "3. (Optional) Login to W&B: ${GREEN}wandb login${NC}"
echo -e "4. Start training: ${GREEN}python train.py${NC}"
echo -e "5. Evaluate model: ${GREEN}python evaluate.py --checkpoint checkpoints/best_model.pth${NC}"
echo -e "\n=========================================="

# Create a quick test script
cat > test_gpu.py << 'EOF'
"""Quick test to verify GPU setup"""
import torch

print("="*50)
print("GPU Test Script")
print("="*50)

# Check CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Test tensor operation
    print("\nTesting GPU tensor operation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU tensor operation successful!")
    
    # Test memory
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
else:
    print("\n⚠ CUDA not available. Training will run on CPU.")

print("\n" + "="*50)
EOF

echo -e "\n${GREEN}Created test script: test_gpu.py${NC}"
echo -e "Run ${GREEN}python test_gpu.py${NC} to verify GPU setup"
