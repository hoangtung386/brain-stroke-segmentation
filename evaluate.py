"""
Evaluation script for Brain Stroke Segmentation
"""
import os
import argparse
import torch
from monai.utils import set_determinism

from config import Config
from dataset import create_dataloaders
from models.lcnn import LCNN
from utils.metrics import SegmentationEvaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Brain Stroke Segmentation Model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set seed
    set_determinism(seed=Config.SEED)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    Config.create_directories()
    
    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    _, test_loader = create_dataloaders(Config)
    
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
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded from epoch {checkpoint['epoch']} with Dice: {checkpoint['best_dice']:.4f}")
    
    # Create evaluator
    print("\n" + "="*60)
    print("Creating evaluator...")
    print("="*60)
    evaluator = SegmentationEvaluator(
        model=model,
        val_loader=test_loader,
        device=device,
        num_classes=Config.NUM_CLASSES,
        output_dir=Config.OUTPUT_DIR
    )
    
    # Compute metrics
    print("\n" + "="*60)
    print("Computing metrics...")
    print("="*60)
    results = evaluator.compute_metrics()
    
    # Plot metrics comparison
    print("\n" + "="*60)
    print("Plotting metrics comparison...")
    print("="*60)
    evaluator.plot_metrics(results)
    
    # Visualize predictions
    print("\n" + "="*60)
    print("Creating overlay visualizations...")
    print("="*60)
    evaluator.visualize_predictions(num_samples=args.num_samples)
    
    # Per-class comparison
    print("\n" + "="*60)
    print("Creating per-class comparison...")
    print("="*60)
    evaluator.plot_per_class_comparison(num_samples=3)
    
    # Confusion matrix
    print("\n" + "="*60)
    print("Creating confusion matrix...")
    print("="*60)
    evaluator.plot_confusion_analysis()
    
    # Create summary report
    print("\n" + "="*60)
    print("Creating summary report...")
    print("="*60)
    evaluator.create_summary_report(results)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
