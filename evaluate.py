#!/usr/bin/env python3
"""
Validation script to check if all fixes are applied correctly
Run this before training to catch potential issues
"""
import sys
import torch
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_success(text):
    """Print success message"""
    print(f"✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"✗ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠ {text}")


def check_imports():
    """Check if all required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'monai': 'MONAI',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'tqdm': 'tqdm'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed")
            all_installed = False
    
    return all_installed


def check_cuda():
    """Check CUDA availability"""
    print_header("Checking CUDA")
    
    if torch.cuda.is_available():
        print_success(f"CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Total Memory: {total_memory:.2f} GB")
        
        if total_memory < 10:
            print_warning("GPU memory < 10GB. Consider reducing batch size.")
        
        return True
    else:
        print_warning("CUDA not available. Training will be very slow on CPU.")
        return False


def check_config():
    """Check configuration file"""
    print_header("Checking Configuration")
    
    try:
        from config import Config
        
        # Check NUM_CHANNELS
        if Config.NUM_CHANNELS != 1:
            print_error(f"NUM_CHANNELS should be 1 (grayscale), got {Config.NUM_CHANNELS}")
            return False
        print_success("NUM_CHANNELS = 1 (correct for grayscale CT)")
        
        # Check MEAN and STD
        if len(Config.MEAN) != 1 or len(Config.STD) != 1:
            print_error(f"MEAN and STD should have length 1 for grayscale")
            return False
        print_success(f"Normalization: MEAN={Config.MEAN}, STD={Config.STD}")
        
        # Check T value
        if Config.T < 1:
            print_warning(f"T={Config.T}. Consider T>=1 for SEAN to work properly.")
        print_success(f"T = {Config.T} (will use {2*Config.T+1} adjacent slices)")
        
        # Check paths
        if not Path(Config.IMAGE_DIR).exists():
            print_error(f"IMAGE_DIR does not exist: {Config.IMAGE_DIR}")
            return False
        print_success(f"IMAGE_DIR exists: {Config.IMAGE_DIR}")
        
        if not Path(Config.MASK_DIR).exists():
            print_error(f"MASK_DIR does not exist: {Config.MASK_DIR}")
            return False
        print_success(f"MASK_DIR exists: {Config.MASK_DIR}")
        
        return True
        
    except ImportError as e:
        print_error(f"Failed to import config: {e}")
        return False


def check_dataset():
    """Check dataset implementation"""
    print_header("Checking Dataset")
    
    try:
        from config import Config
        from dataset import BrainStrokeDataset, create_dataloaders
        
        # Try to create dataloaders
        train_loader, val_loader = create_dataloaders(Config)
        
        # Check one batch
        for images, masks in train_loader:
            print_success(f"Dataset loads successfully")
            print(f"  Image shape: {images.shape}")
            print(f"  Mask shape: {masks.shape}")
            
            # Validate shapes
            expected_slices = 2 * Config.T + 1
            if images.shape[1] != expected_slices:
                print_error(f"Expected {expected_slices} slices, got {images.shape[1]}")
                return False
            print_success(f"Correct number of slices: {expected_slices}")
            
            if len(masks.shape) != 3:  # (B, H, W)
                print_error(f"Mask should be 3D (B, H, W), got {masks.shape}")
                return False
            print_success(f"Mask has correct shape: (B, H, W)")
            
            # Check mask values
            unique_values = torch.unique(masks)
            print(f"  Mask unique values: {unique_values.tolist()}")
            if not all(v in [0, 1] for v in unique_values.tolist()):
                print_warning(f"Mask contains values other than 0 and 1: {unique_values.tolist()}")
            
            break
        
        return True
        
    except Exception as e:
        print_error(f"Dataset check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model():
    """Check model architecture"""
    print_header("Checking Model")
    
    try:
        from config import Config
        from models.lcnn import LCNN
        
        model = LCNN(
            num_channels=Config.NUM_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            global_impact=Config.GLOBAL_IMPACT,
            local_impact=Config.LOCAL_IMPACT,
            T=Config.T
        )
        
        print_success("Model created successfully")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Total parameters: {num_params:.2f}M")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        B = 2
        num_slices = 2 * Config.T + 1
        H, W = Config.IMAGE_SIZE
        
        dummy_input = torch.randn(B, num_slices, H, W).to(device)
        
        print("  Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
        
        print_success(f"Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Validate output shape
        if output.shape != (B, Config.NUM_CLASSES, H, W):
            print_error(f"Expected output shape ({B}, {Config.NUM_CLASSES}, {H}, {W}), got {output.shape}")
            return False
        print_success("Output shape is correct")
        
        # Test with alignment
        print("  Testing forward pass with alignment...")
        with torch.no_grad():
            output, aligned, params = model(dummy_input, return_alignment=True)
        print_success("Forward pass with alignment successful")
        print(f"  Number of aligned slices: {len(aligned)}")
        
        return True
        
    except Exception as e:
        print_error(f"Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_loss():
    """Check loss function"""
    print_header("Checking Loss Function")
    
    try:
        from trainer import CombinedLoss
        from config import Config
        
        loss_fn = CombinedLoss(
            num_classes=Config.NUM_CLASSES,
            dice_weight=0.5,
            ce_weight=0.5,
            alignment_weight=0.1,
            use_alignment=True
        )
        
        print_success("Loss function created successfully")
        
        # Test loss computation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = loss_fn.to(device)
        
        B = 2
        H, W = 64, 64  # Smaller for testing
        
        outputs = torch.randn(B, Config.NUM_CLASSES, H, W).to(device)
        targets = torch.randint(0, Config.NUM_CLASSES, (B, H, W)).to(device)
        
        # Create dummy aligned slices
        aligned_slices = [torch.randn(B, 1, H, W).to(device) for _ in range(3)]
        
        total_loss, dice_ce, alignment = loss_fn(outputs, targets, aligned_slices)
        
        print_success("Loss computation successful")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Dice+CE loss: {dice_ce.item():.4f}")
        print(f"  Alignment loss: {alignment.item():.4f}")
        
        # Check if losses are reasonable
        if torch.isnan(total_loss):
            print_error("Loss is NaN! Check normalization and loss weights.")
            return False
        
        if total_loss.item() < 0:
            print_error("Loss is negative! Something is wrong.")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Loss check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_trainer():
    """Check trainer implementation"""
    print_header("Checking Trainer")
    
    try:
        from config import Config
        from dataset import create_dataloaders
        from models.lcnn import LCNN
        from trainer import Trainer
        
        # Create small dataset for testing
        train_loader, val_loader = create_dataloaders(Config)
        
        # Create model
        model = LCNN(
            num_channels=Config.NUM_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            global_impact=Config.GLOBAL_IMPACT,
            local_impact=Config.LOCAL_IMPACT,
            T=Config.T
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=Config,
            device=device,
            use_wandb=False  # Disable W&B for testing
        )
        
        print_success("Trainer created successfully")
        
        # Check if gradient clipping is enabled
        if hasattr(trainer, 'config') and hasattr(trainer.config, 'GRAD_CLIP_NORM'):
            print_success(f"Gradient clipping enabled: max_norm={trainer.config.GRAD_CLIP_NORM}")
        
        return True
        
    except Exception as e:
        print_error(f"Trainer check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks"""
    print("\n" + "="*60)
    print("  Brain Stroke Segmentation - Validation Script")
    print("  Checking if all fixes are applied correctly")
    print("="*60)
    
    checks = [
        ("Dependencies", check_imports),
        ("CUDA", check_cuda),
        ("Configuration", check_config),
        ("Dataset", check_dataset),
        ("Model", check_model),
        ("Loss Function", check_loss),
        ("Trainer", check_trainer),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print_error(f"Unexpected error in {name} check: {e}")
            results[name] = False
    
    # Summary
    print_header("Validation Summary")
    
    all_passed = True
    for name, passed in results.items():
        if passed:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. (Optional) Compute normalization stats:")
        print("     python -c 'from config import Config; Config.compute_normalization_stats(Config.IMAGE_DIR)'")
        print("  2. Start training:")
        print("     python train.py")
    else:
        print("✗ Some checks failed. Please fix the issues before training.")
        print("\nRefer to FIXES_SUMMARY.md for detailed instructions.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
