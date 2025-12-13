"""
Visualization utilities for segmentation results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def plot_metrics_comparison(results, output_dir):
    """
    Plot metrics comparison across classes
    
    Args:
        results: Dictionary of metrics per class
        output_dir: Directory to save plots
    """
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'specificity']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        classes = list(results.keys())
        values = [results[cls][metric] for cls in classes]
        
        axes[idx].bar(classes, values, color='steelblue', alpha=0.8)
        axes[idx].set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=12)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {save_path}")


def visualize_overlay_predictions(model, val_loader, device, class_names, 
                                   output_dir, num_samples=5):
    """
    Visualize overlay masks for predictions
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device (cuda/cpu)
        class_names: List of class names
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Define colors for different classes
    num_classes = len(class_names)
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink']
    colors = colors[:num_classes]
    cmap = ListedColormap(colors)
    
    samples_shown = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            if samples_shown >= num_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Convert to numpy
            # Use center slice for image (images is B, T, H, W)
            center_slice_idx = images.shape[1] // 2
            img_np = images[0, center_slice_idx].cpu().numpy()
            
            pred_np = preds[0].cpu().numpy()
            mask_np = masks[0].cpu().numpy()
            
            # Ensure mask is 2D (H, W)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_np, cmap='gray')
            axes[0].set_title('CT Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(img_np, cmap='gray')
            axes[1].imshow(mask_np, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes-1)
            axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(img_np, cmap='gray')
            axes[2].imshow(pred_np, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes-1)
            axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Add legend
            legend_elements = [
                Patch(facecolor=colors[i], label=class_names[i])
                for i in range(1, num_classes)
            ]
            fig.legend(handles=legend_elements, loc='lower center', 
                      ncol=min(5, num_classes-1), fontsize=12)
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'overlay_sample_{samples_shown+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            samples_shown += 1
    
    print(f"Overlay visualizations saved to {output_dir}/overlay_sample_*.png")


def plot_per_class_comparison(model, val_loader, device, class_names, 
                               output_dir, num_samples=3):
    """
    Plot detailed per-class segmentation comparison
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device (cuda/cpu)
        class_names: List of class names
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    model.eval()
    num_classes = len(class_names)
    samples_shown = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            if samples_shown >= num_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Convert to numpy
            # Use center slice for image
            center_slice_idx = images.shape[1] // 2
            img_np = images[0, center_slice_idx].cpu().numpy()
            
            pred_np = preds[0].cpu().numpy()
            mask_np = masks[0].cpu().numpy()
            
            # Ensure mask is 2D (H, W)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)
            
            # Create figure with per-class comparison
            fig, axes = plt.subplots(2, num_classes, figsize=(4*num_classes, 8))
            
            for cls_idx in range(num_classes):
                # Ground truth for this class
                gt_mask = (mask_np == cls_idx).astype(float)
                axes[0, cls_idx].imshow(img_np, cmap='gray')
                axes[0, cls_idx].imshow(gt_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                axes[0, cls_idx].set_title(
                    f'{class_names[cls_idx]}\nGround Truth',
                    fontsize=10, fontweight='bold'
                )
                axes[0, cls_idx].axis('off')
                
                # Prediction for this class
                pred_mask = (pred_np == cls_idx).astype(float)
                axes[1, cls_idx].imshow(img_np, cmap='gray')
                axes[1, cls_idx].imshow(pred_mask, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
                axes[1, cls_idx].set_title(
                    f'{class_names[cls_idx]}\nPrediction',
                    fontsize=10, fontweight='bold'
                )
                axes[1, cls_idx].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'per_class_comparison_{samples_shown+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            samples_shown += 1
    
    print(f"Per-class comparison saved to {output_dir}/per_class_comparison_*.png")


def plot_confusion_matrix(cm, class_names, output_dir):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        output_dir: Directory to save plot
    """
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names,
        ax=ax1, 
        cbar_kws={'label': 'Count'}
    )
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.3f', 
        cmap='Greens',
        xticklabels=class_names, 
        yticklabels=class_names,
        ax=ax2, 
        cbar_kws={'label': 'Proportion'}
    )
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(history_csv_path, output_dir):
    """
    Plot training history from CSV
    
    Args:
        history_csv_path: Path to training history CSV
        output_dir: Directory to save plot
    """
    import pandas as pd
    
    if not os.path.exists(history_csv_path):
        print(f"History file not found: {history_csv_path}")
        return
    
    df = pd.read_csv(history_csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Dice plot
    ax2.plot(df['epoch'], df['val_dice'], marker='o', color='green', label='Val Dice')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")
