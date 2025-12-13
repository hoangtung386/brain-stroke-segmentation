"""
Fixed Trainer class with proper loss functions
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from torch.cuda.amp import GradScaler, autocast


class CombinedLoss(nn.Module):
    """
    Combined loss: Dice + CE + Alignment
    
    This is the proper loss for SEAN architecture:
    1. Dice Loss: For segmentation quality
    2. Cross Entropy: For pixel-wise classification
    3. Alignment Loss: For symmetry-based alignment (optional)
    """
    
    def __init__(self, num_classes=2, dice_weight=0.5, ce_weight=0.5, 
                 alignment_weight=0.1, use_alignment=True):
        super(CombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.alignment_weight = alignment_weight
        self.use_alignment = use_alignment
        
        # Dice + Cross Entropy loss
        self.dice_ce = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=dice_weight,
            lambda_ce=ce_weight
        )
    
    def forward(self, outputs, targets, aligned_slices=None):
        """
        Args:
            outputs: (B, num_classes, H, W)
            targets: (B, H, W) with class indices
            aligned_slices: List of aligned slices for symmetry loss
        
        Returns:
            total_loss, dice_ce_loss, alignment_loss
        """
        # Main segmentation loss
        dice_ce_loss = self.dice_ce(outputs, targets)
        
        total_loss = dice_ce_loss
        alignment_loss = torch.tensor(0.0, device=outputs.device)
        
        # Add alignment loss if slices are provided
        if self.use_alignment and aligned_slices is not None:
            for aligned_slice in aligned_slices:
                # Flip horizontally and compute L1 difference
                flipped = torch.flip(aligned_slice, dims=[-1])
                alignment_loss += torch.nn.functional.l1_loss(aligned_slice, flipped)
            
            # Average over slices
            alignment_loss = alignment_loss / len(aligned_slices)
            total_loss = total_loss + self.alignment_weight * alignment_loss
        
        return total_loss, dice_ce_loss, alignment_loss


class Trainer:
    """Fixed trainer class for brain stroke segmentation"""
    
    def __init__(self, model, train_loader, val_loader, config, device, use_wandb=False):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device (cuda/cpu)
            use_wandb: Whether to use Weights & Biases
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function with proper configuration
        self.criterion = CombinedLoss(
            num_classes=config.NUM_CLASSES,
            dice_weight=0.5,
            ce_weight=0.5,
            alignment_weight=0.1,
            use_alignment=True
        )
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # AMP Scaler
        self.scaler = GradScaler()
        
        # Cosine annealing scheduler (better than ReduceLROnPlateau for this task)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Dice metric for validation
        self.dice_metric = DiceMetric(
            include_background=False, 
            reduction='mean',
            get_not_nans=True
        )
        
        # Training state
        self.start_epoch = 0
        self.best_dice = 0.0
        self.history = []
        self.wandb_run_id = None
        
        # Paths
        self.checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, 
            'checkpoint.pth'
        )
        self.best_model_path = os.path.join(
            config.CHECKPOINT_DIR, 
            'best_model.pth'
        )
        self.history_csv_path = os.path.join(
            config.OUTPUT_DIR, 
            'training_history.csv'
        )
    
    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config.to_dict(),
            'wandb_run_id': self.wandb_run_id
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"✓ Best model saved! Dice: {val_dice:.4f}")
    
    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_dice = checkpoint['best_dice']
            self.history = checkpoint['history']
            
            if 'wandb_run_id' in checkpoint:
                self.wandb_run_id = checkpoint['wandb_run_id']
            
            print(f"Resumed from epoch {self.start_epoch}, best dice: {self.best_dice:.4f}")
            return True
        return False
    
    def save_history_csv(self):
        """Save training history to CSV"""
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(self.history_csv_path, index=False)
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_dice_ce = 0
        total_alignment = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, masks in pbar:
            images = images.to(self.device)  # (B, 2T+1, H, W)
            masks = masks.to(self.device)    # (B, H, W)
            
            self.optimizer.zero_grad()
            
            # Forward pass with alignment
            with autocast():
                outputs, aligned_slices, _ = self.model(images, return_alignment=True)
                
                # Compute loss
                loss, dice_ce_loss, alignment_loss = self.criterion(
                    outputs, masks, aligned_slices
                )
            
            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_dice_ce += dice_ce_loss.item()
            total_alignment += alignment_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice_ce': f'{dice_ce_loss.item():.4f}',
                'align': f'{alignment_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_dice_ce = total_dice_ce / len(self.train_loader)
        avg_alignment = total_alignment / len(self.train_loader)
        
        return avg_loss, avg_dice_ce, avg_alignment
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        self.dice_metric.reset()
        
        total_val_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                # Compute validation loss
                loss, _, _ = self.criterion(outputs, masks, aligned_slices=None)
                total_val_loss += loss.item()
                
                # Compute Dice metric
                self.dice_metric(y_pred=outputs, y=masks)
        
        val_dice = self.dice_metric.aggregate().item()
        avg_val_loss = total_val_loss / len(self.val_loader)
        
        return val_dice, avg_val_loss
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        resumed = self.load_checkpoint()
        
        # Initialize W&B if enabled
        if self.use_wandb:
            import wandb
            if resumed and self.wandb_run_id:
                wandb.init(
                    project=self.config.WANDB_PROJECT,
                    entity=self.config.WANDB_ENTITY,
                    config=self.config.to_dict(),
                    resume="allow",
                    id=self.wandb_run_id,
                    name=f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            else:
                run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project=self.config.WANDB_PROJECT,
                    entity=self.config.WANDB_ENTITY,
                    config=self.config.to_dict(),
                    name=run_name,
                    tags=["brain-stroke", "segmentation", "LCNN"]
                )
                self.wandb_run_id = wandb.run.id
            
            wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {self.start_epoch} to {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss, dice_ce_loss, alignment_loss = self.train_epoch(epoch + 1)
            
            # Validate
            val_dice, val_loss = self.validate(epoch + 1)
            
            # Step scheduler
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice_ce': dice_ce_loss,
                'train_alignment': alignment_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'learning_rate': current_lr,
                'best_dice': self.best_dice
            }
            
            self.history.append(metrics)
            
            # Print metrics
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"{'='*60}")
            print(f"  Train Loss:       {train_loss:.4f}")
            print(f"    - Dice+CE:      {dice_ce_loss:.4f}")
            print(f"    - Alignment:    {alignment_loss:.4f}")
            print(f"  Val Loss:         {val_loss:.4f}")
            print(f"  Val Dice:         {val_dice:.4f}")
            print(f"  Learning Rate:    {current_lr:.6f}")
            print(f"  Best Dice:        {self.best_dice:.4f}")
            print(f"{'='*60}\n")
            
            # Log to W&B
            if self.use_wandb:
                import wandb
                wandb.log(metrics)
            
            # Save checkpoint
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
            
            self.save_checkpoint(epoch, val_dice, is_best)
            self.save_history_csv()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        print(f"{'='*60}\n")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
