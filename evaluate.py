#!/usr/bin/env python3
"""
Evaluation script for Brain Stroke Segmentation
Generates metrics and visualizations using the trained model
"""
import os
import sys
import torch
import argparse
from pathlib import Path

from config import Config
from dataset import create_dataloaders
from models.lcnn import LCNN
from utils.metrics import SegmentationEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate Brain Stroke Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of visualization samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Evaluation Settings:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Create dataloaders
    print("\nLoading dataset...")
    # Override batch size for evaluation
    # We might want to keep the config batch size for training but use a different one for eval
    # But create_dataloaders uses Config.BATCH_SIZE. 
    # Let's temporarily patch Config for this run if needed, or just rely on val_loader
    # The user might want to adjust batch size via CLI
    original_batch_size = Config.BATCH_SIZE
    Config.BATCH_SIZE = args.batch_size
    
    _, val_loader = create_dataloaders(Config)
    
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = LCNN(
        num_channels=Config.NUM_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        global_impact=Config.GLOBAL_IMPACT,
        local_impact=Config.LOCAL_IMPACT,
        T=Config.T
    )
    
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return 1
        
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle both full checkpoint dict and just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_dice' in checkpoint:
            print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dictionary")
    
    # Initialize evaluator
    evaluator = SegmentationEvaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=Config.NUM_CLASSES,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    print("\nStarting evaluation...")
    
    # 1. Compute and save metrics
    results = evaluator.compute_metrics()
    evaluator.create_summary_report(results)
    evaluator.plot_metrics(results)
    
    # 2. Visualize predictions
    print(f"\nGenerating {args.num_samples} visualization samples...")
    evaluator.visualize_predictions(num_samples=args.num_samples)
    evaluator.plot_per_class_comparison(num_samples=min(3, args.num_samples))
    
    # 3. Confusion Matrix
    evaluator.plot_confusion_analysis()
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    