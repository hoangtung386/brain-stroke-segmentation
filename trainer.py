"""
Trainer with NaN detection and recovery
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

from utils.improved_alignment_loss import ImprovedCombinedLoss


class Trainer:
    """
    Trainer with:
    1. NaN detection and recovery
    2. Proper gradient clipping
    3. Loss scaling monitoring
    """
    
    def __init__(self, model, train_loader, val_loader, config, device, use_wandb=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        self.model.to(self.device)
        
        # Loss function
        self.criterion = ImprovedCombinedLoss(
            num_classes=config.NUM_CLASSES,
            dice_weight=0.5,
            ce_weight=0.5,
            alignment_weight=0.1,  # ← ĐÃ GIẢM trong config
            use_alignment=True
        )
        self.criterion.to(self.device)
        
        # Optimizer với weight decay mạnh hơn
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-4,  # ← Tăng từ 1e-5
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # AMP Scaler với init_scale thấp hơn
        self.scaler = GradScaler(init_scale=512)  # ← Giảm từ 2^16
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Dice metric
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
        self.nan_count = 0  # Track số lần gặp NaN
        
        # Paths
        self.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'checkpoint.pth')
        self.best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        self.history_csv_path = os.path.join(config.OUTPUT_DIR, 'training_history.csv')
    
    def check_for_nan(self, tensor, name="tensor"):
        """Check if tensor contains NaN or Inf"""
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}!")
            return True
        if torch.isinf(tensor).any():
            print(f"Inf detected in {name}!")
            return True
        return False
    
    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config.to_dict(),
            'wandb_run_id': self.wandb_run_id,
            'nan_count': self.nan_count
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"Best model saved! Dice: {val_dice:.4f}")
    
    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_dice = checkpoint['best_dice']
            self.history = checkpoint['history']
            
            if 'wandb_run_id' in checkpoint:
                self.wandb_run_id = checkpoint['wandb_run_id']
            if 'nan_count' in checkpoint:
                self.nan_count = checkpoint['nan_count']
            
            print(f"Resumed from epoch {self.start_epoch}, best dice: {self.best_dice:.4f}")
            return True
        return False
    
    def save_history_csv(self):
        """Save training history to CSV"""
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(self.history_csv_path, index=False)
    
    def train_epoch(self, epoch):
        """Train one epoch with NaN detection"""
        self.model.train()
        total_loss = 0
        total_dice_ce = 0
        total_alignment = 0
        total_symmetry = 0
        total_regularization = 0
        total_edge = 0
        
        valid_batches = 0
        nan_in_epoch = False
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Check input data for NaN
            if self.check_for_nan(images, "input images"):
                print(f"Skipping batch {batch_idx} due to NaN in input")
                continue
            
            self.optimizer.zero_grad()
            
            try:
                with autocast():
                    outputs, aligned_slices, alignment_params = self.model(
                        images, return_alignment=True
                    )
                
                # Check outputs
                if self.check_for_nan(outputs, "model outputs"):
                    print(f"Skipping batch {batch_idx} due to NaN in outputs")
                    self.nan_count += 1
                    if self.nan_count > 10:
                        print("Too many NaN occurrences! Stopping training...")
                        nan_in_epoch = True
                        break
                    continue
                
                # Move loss computation outside autocast
                outputs = outputs.float()
                
                original_slices = [
                    images[:, i:i+1, :, :].float()
                    for i in range(images.shape[1])
                ]
                
                if aligned_slices is not None:
                    aligned_slices = [s.float() for s in aligned_slices]
                if alignment_params is not None:
                    alignment_params = [p.float() for p in alignment_params]
                
                # Compute loss
                loss, dice_ce_loss, alignment_loss, align_details = self.criterion(
                    outputs, masks, aligned_slices, alignment_params, original_slices
                )
                
                # Check loss for NaN
                if self.check_for_nan(loss, "loss"):
                    print(f"Skipping batch {batch_idx} due to NaN in loss")
                    self.nan_count += 1
                    continue
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Enhanced gradient clipping
                self.scaler.unscale_(self.optimizer)
                
                # Check gradients for NaN
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if self.check_for_nan(param.grad, f"gradient {name}"):
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    print(f"Skipping batch {batch_idx} due to NaN in gradients")
                    self.nan_count += 1
                    self.optimizer.zero_grad()
                    continue
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.GRAD_CLIP_NORM
                )
                
                # Monitor gradient norm
                if grad_norm > 10.0:
                    print(f"Large gradient norm: {grad_norm:.2f}")
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_dice_ce += dice_ce_loss.item()
                total_alignment += alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss
                
                if align_details:
                    total_symmetry += align_details.get('symmetry', 0)
                    total_regularization += align_details.get('regularization', 0)
                    total_edge += align_details.get('edge_consistency', 0)
                
                valid_batches += 1
                
                # Progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice_ce': f'{dice_ce_loss.item():.4f}',
                    'grad': f'{grad_norm:.2f}',
                    'scale': f'{self.scaler.get_scale():.0f}'
                })
            
            except RuntimeError as e:
                print(f"Runtime error in batch {batch_idx}: {e}")
                self.nan_count += 1
                self.optimizer.zero_grad()
                continue
        
        if nan_in_epoch:
            return None, None, None, None
        
        if valid_batches == 0:
            print("No valid batches in this epoch!")
            return None, None, None, None
        
        avg_loss = total_loss / valid_batches
        avg_dice_ce = total_dice_ce / valid_batches
        avg_alignment = total_alignment / valid_batches
        avg_symmetry = total_symmetry / valid_batches
        avg_regularization = total_regularization / valid_batches
        avg_edge = total_edge / valid_batches
        
        return avg_loss, avg_dice_ce, avg_alignment, {
            'symmetry': avg_symmetry,
            'regularization': avg_regularization,
            'edge_consistency': avg_edge
        }
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        self.dice_metric.reset()
        
        total_val_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                try:
                    with autocast():
                        outputs = self.model(images)
                    
                    outputs = outputs.float()
                    
                    # ============ FIX ĐÂY ============
                    # CHỈ TÍNH DICE + CE, BỎ QUA ALIGNMENT LOSS
                    from monai.losses import DiceCELoss
                    val_criterion = DiceCELoss(
                        include_background=True,
                        to_onehot_y=True,
                        softmax=True,
                        lambda_dice=0.5,
                        lambda_ce=0.5
                    )
                    val_criterion = val_criterion.to(self.device)
                    
                    if masks.ndim == 3:
                        masks_for_loss = masks.unsqueeze(1)
                    else:
                        masks_for_loss = masks
                    
                    loss = val_criterion(outputs, masks_for_loss)
                    # ==================================
                    
                    if not self.check_for_nan(loss, "val_loss"):
                        total_val_loss += loss.item()
                        valid_batches += 1
                    else:
                        print(f"Skipping batch due to NaN in val_loss")
                        continue
                    
                    # Compute Dice metric
                    if masks.ndim == 3:
                        masks_for_metric = masks.unsqueeze(1)
                    else:
                        masks_for_metric = masks
                    
                    self.dice_metric(y_pred=outputs, y=masks_for_metric)
                
                except RuntimeError as e:
                    print(f"Validation error: {e}")
                    continue
        
        if valid_batches == 0:
            print("No valid batches in validation!")
            return 0.0, float('inf')
        
        dice_result = self.dice_metric.aggregate()
        
        if isinstance(dice_result, (list, tuple)):
            val_dice = dice_result[0].item() if len(dice_result) > 0 else 0.0
        else:
            val_dice = dice_result.item()
        
        avg_val_loss = total_val_loss / valid_batches
        
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
                    id=self.wandb_run_id
                )
            else:
                run_name = f"fixed_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project=self.config.WANDB_PROJECT,
                    entity=self.config.WANDB_ENTITY,
                    config=self.config.to_dict(),
                    name=run_name,
                    tags=["brain-stroke", "fixed-nan"]
                )
                self.wandb_run_id = wandb.run.id
            
            wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"\n{'='*60}")
        print(f"Starting FIXED training from epoch {self.start_epoch} to {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Gradient clip: {self.config.GRAD_CLIP_NORM}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            result = self.train_epoch(epoch + 1)
            
            if result[0] is None:
                print(f"Epoch {epoch+1} failed due to NaN. Stopping training.")
                break
            
            train_loss, dice_ce_loss, alignment_loss, align_details = result
            
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
                'best_dice': self.best_dice,
                'nan_count': self.nan_count
            }
            
            self.history.append(metrics)
            
            # Print metrics
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"{'='*60}")
            print(f"  Train Loss:       {train_loss:.4f}")
            print(f"  Val Loss:         {val_loss:.4f}")
            print(f"  Val Dice:         {val_dice:.4f}")
            print(f"  Learning Rate:    {current_lr:.6f}")
            print(f"  Best Dice:        {self.best_dice:.4f}")
            print(f"  NaN Count:        {self.nan_count}")
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
        print(f"Total NaN occurrences: {self.nan_count}")
        print(f"{'='*60}\n")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
            