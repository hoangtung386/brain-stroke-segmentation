"""
Metrics computation and evaluation for segmentation
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from .visualization import (
    plot_metrics_comparison,
    visualize_overlay_predictions,
    plot_per_class_comparison,
    plot_confusion_matrix
)


class SegmentationEvaluator:
    """Evaluator class for segmentation metrics and visualization"""
    
    def __init__(self, model, val_loader, device, num_classes=2, output_dir='./outputs'):
        """
        Args:
            model: PyTorch model
            val_loader: Validation data loader
            device: Device (cuda/cpu)
            num_classes: Number of classes
            output_dir: Directory to save outputs
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.output_dir = output_dir
        
        # Class names
        self.class_names = ['Background', 'Stroke_Region']
        if num_classes > 2:
            self.class_names = [
                'Background',
                'Level_1_ischemic_area',
                'Level_2_ischemic_area',
                'Level_3_ischemic_area',
                'Level_4_ischemic_area',
                'Level_5_ischemic_area'
            ][:num_classes]
        
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_metrics(self):
        """Compute comprehensive metrics for all classes"""
        self.model.eval()
        
        # Initialize metrics storage
        class_metrics = {cls: {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'specificity': []
        } for cls in range(1, self.num_classes)}
        
        print("Computing metrics on validation set...")
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Convert to numpy
                pred_np = preds.cpu().numpy()
                mask_np = masks.cpu().numpy()
                
                # Compute metrics for each class
                for cls in range(1, self.num_classes):
                    pred_cls = (pred_np == cls).astype(np.float32)
                    mask_cls = (mask_np == cls).astype(np.float32)
                    
                    # Skip if no ground truth for this class
                    if mask_cls.sum() == 0:
                        continue
                    
                    # Dice coefficient
                    intersection = (pred_cls * mask_cls).sum()
                    dice = (2 * intersection) / (pred_cls.sum() + mask_cls.sum() + 1e-8)
                    class_metrics[cls]['dice'].append(dice)
                    
                    # IoU (Intersection over Union)
                    union = pred_cls.sum() + mask_cls.sum() - intersection
                    iou = intersection / (union + 1e-8)
                    class_metrics[cls]['iou'].append(iou)
                    
                    # Precision, Recall, Specificity
                    tp = intersection
                    fp = pred_cls.sum() - intersection
                    fn = mask_cls.sum() - intersection
                    tn = ((pred_cls == 0) & (mask_cls == 0)).sum()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    specificity = tn / (tn + fp + 1e-8)
                    
                    class_metrics[cls]['precision'].append(precision)
                    class_metrics[cls]['recall'].append(recall)
                    class_metrics[cls]['specificity'].append(specificity)
        
        # Compute average metrics
        results = {}
        for cls in range(1, self.num_classes):
            results[self.class_names[cls]] = {
                metric: np.mean(values) if values else 0.0
                for metric, values in class_metrics[cls].items()
            }
        
        return results
    
    def plot_metrics(self, results):
        """Plot metrics comparison"""
        plot_metrics_comparison(results, self.output_dir)
    
    def visualize_predictions(self, num_samples=5):
        """Visualize overlay masks for predictions"""
        visualize_overlay_predictions(
            self.model,
            self.val_loader,
            self.device,
            self.class_names,
            self.output_dir,
            num_samples=num_samples
        )
    
    def plot_per_class_comparison(self, num_samples=3):
        """Plot detailed per-class segmentation comparison"""
        plot_per_class_comparison(
            self.model,
            self.val_loader,
            self.device,
            self.class_names,
            self.output_dir,
            num_samples=num_samples
        )
    
    def plot_confusion_analysis(self):
        """Plot confusion matrix for segmentation classes"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        print("Computing confusion matrix...")
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(masks.cpu().numpy().flatten())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))
        
        # Plot
        plot_confusion_matrix(cm, self.class_names, self.output_dir)
    
    def create_summary_report(self, results):
        """Create text summary report"""
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BRAIN STROKE SEGMENTATION EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            for cls_name, metrics in results.items():
                f.write(f"\n{cls_name}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Dice Coefficient:  {metrics['dice']:.4f}\n")
                f.write(f"  IoU:              {metrics['iou']:.4f}\n")
                f.write(f"  Precision:        {metrics['precision']:.4f}\n")
                f.write(f"  Recall:           {metrics['recall']:.4f}\n")
                f.write(f"  Specificity:      {metrics['specificity']:.4f}\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("OVERALL AVERAGES:\n")
            f.write("-"*60 + "\n")
            
            for metric in ['dice', 'iou', 'precision', 'recall', 'specificity']:
                avg = np.mean([results[cls][metric] for cls in results.keys()])
                f.write(f"  Average {metric.upper()}: {avg:.4f}\n")
        
        print(f"\nEvaluation report saved to {report_path}")
        
        # Print to console
        with open(report_path, 'r') as f:
            print(f.read())
