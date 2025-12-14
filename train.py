"""
Main training script
"""
import os
import torch
import gc
from monai.utils import set_determinism

from config import Config
from dataset import create_dataloaders
from models.lcnn import LCNN
from trainer import Trainer


def main():
    """Main training function"""
    
    # Set seed for reproducibility
    set_determinism(seed=Config.SEED)
    
    if Config.DEBUG_MODE:
        torch.autograd.set_detect_anomaly(True)
        print("🔍 Debug mode enabled - anomaly detection ON")
    
    # Create directories
    Config.create_directories()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    train_loader, val_loader = create_dataloaders(Config)
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    model = LCNN(
        num_channels=Config.NUM_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        global_impact=Config.GLOBAL_IMPACT,
        local_impact=Config.LOCAL_IMPACT,
        T=Config.T
    )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    # Clean memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create trainer
    print("\n" + "="*60)
    print("Initializing trainer...")
    print("="*60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=Config,
        device=device,
        use_wandb=Config.USE_WANDB
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train(num_epochs=Config.NUM_EPOCHS)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
