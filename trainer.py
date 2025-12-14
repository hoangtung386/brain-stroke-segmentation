"""
Trainer with AGGRESSIVE NaN Prevention and Scale Control
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
    Trainer với:
    1. AGGRESSIVE scale control - không để scale vượt quá 4096
    2. Loss value monitoring - detect anomalies TRƯỚC khi backward
    3. Dynamic alignment weight reduction khi gặp NaN
    4. Gradient statistics logging
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
            dice_weight=config.DICE_WEIGHT,
            ce_weight=config.CE_WEIGHT,
            alignment_weight=config.ALIGNMENT_WEIGHT,
            use_alignment=True
        )
        self.criterion.to(self.device)
        
        # Optimizer với gradient clipping trong optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # CRITICAL: AMP Scaler với AGGRESSIVE control
        self.scaler = GradScaler(
            init_scale=256,        # Giảm từ 512
            growth_factor=1.5,     # Tăng chậm hơn (default 2.0)
            backoff_factor=0.25,   # Giảm nhanh hơn khi gặp NaN (default 0.5)
            growth_interval=500,   # Tăng scale sau 500 iterations (default 2000)
        )
        self.max_scale = 4096  # HARD CAP - không để scale vượt quá
        
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
        self.nan_count = 0
        self.consecutive_nans = 0  # Track liên tiếp
        self.alignment_warmup_epochs = 10
        
        # Scale monitoring
        self.scale_history = []
        self.scale_reduced_count = 0
        
        # Paths
        self.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'checkpoint.pth')
        self.best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        self.history_csv_path = os.path.join(config.OUTPUT_DIR, 'training_history.csv')
    
    def check_tensor_health(self, tensor, name="tensor", max_val=100):
        """
        Comprehensive tensor health check
        Returns: (is_healthy, issue_description)
        """
        if torch.isnan(tensor).any():
            return False, f"NaN in {name}"
        if torch.isinf(tensor).any():
            return False, f"Inf in {name}"
        
        # Check extreme values
        max_abs = tensor.abs().max().item()
        if max_abs > max_val:
            return False, f"Extreme value in {name}: {max_abs:.2f}"
        
        return True, "OK"

    def control_scaler_growth(self):
        """
        Kiểm soát scale không được vượt quá MAX_SCALE
        """
        current_scale = self.scaler.get_scale()
        
        if current_scale > self.max_scale:
            print(f"\nScale too high ({current_scale:.0f}), reducing to {self.max_scale}")
            self.scaler._scale.fill_(self.max_scale)
            self.scale_reduced_count += 1
            return True
        
        return False

    def get_alignment_weight(self, epoch):
        """
        Gradually increase alignment weight with NaN-aware reduction
        """
        base_weight = self.config.ALIGNMENT_WEIGHT
        
        # Warmup phase
        if epoch < self.alignment_warmup_epochs:
            base_weight = base_weight * (epoch / self.alignment_warmup_epochs)
        
        # REDUCE if too many consecutive NaNs
        if self.consecutive_nans > 3:
            reduction_factor = 0.5 ** (self.consecutive_nans - 3)
            base_weight = base_weight * reduction_factor
            print(f"Alignment weight reduced to {base_weight:.5f} due to NaNs")
        
        return max(base_weight, 1e-4)  # Minimum threshold
    
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
            'nan_count': self.nan_count,
            'scale_history': self.scale_history,
            'scale_reduced_count': self.scale_reduced_count
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
            if 'scale_history' in checkpoint:
                self.scale_history = checkpoint['scale_history']
            if 'scale_reduced_count' in checkpoint:
                self.scale_reduced_count = checkpoint['scale_reduced_count']
            
            print(f"Resumed from epoch {self.start_epoch}, best dice: {self.best_dice:.4f}")
            return True
        return False
    
    def save_history_csv(self):
        """Save training history to CSV"""
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(self.history_csv_path, index=False)
    
    def train_epoch(self, epoch):
        """Train one epoch with MAXIMUM NaN protection"""
        self.model.train()
        self.consecutive_nans = 0  # Reset at epoch start
        
        total_loss = 0
        total_dice_ce = 0
        total_alignment = 0
        valid_batches = 0
        
        # Get alignment weight for this epoch
        current_alignment_weight = self.get_alignment_weight(epoch)
        self.criterion.alignment_weight = current_alignment_weight
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train] (Align={current_alignment_weight:.4f})")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Control scale BEFORE forward pass
            scale_was_reduced = self.control_scaler_growth()
            current_scale = self.scaler.get_scale()
            
            # Validate and sanitize inputs
            is_healthy, issue = self.check_tensor_health(images, "input_images", max_val=20)
            if not is_healthy:
                print(f"Batch {batch_idx}: {issue} - SKIPPING")
                self.consecutive_nans += 1
                continue
            
            images = torch.clamp(images, -10, 10)
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass with health checks
                with autocast(enabled=True):  # Explicitly enable
                    outputs, aligned_slices, alignment_params = self.model(
                        images, return_alignment=True
                    )
                    
                    # Immediate output check
                    is_healthy, issue = self.check_tensor_health(outputs, "model_outputs", max_val=50)
                    if not is_healthy:
                        print(f"Batch {batch_idx}: {issue} - SKIPPING")
                        self.consecutive_nans += 1
                        torch.cuda.empty_cache()
                        continue
                    
                    outputs = torch.clamp(outputs, -30, 30)
                
                # Prepare for loss (OUTSIDE autocast for stability)
                outputs = outputs.float()
                original_slices = [images[:, i:i+1, :, :].float() for i in range(images.shape[1])]
                
                if aligned_slices is not None:
                    aligned_slices = [torch.clamp(s.float(), -10, 10) for s in aligned_slices]
                if alignment_params is not None:
                    alignment_params = [torch.clamp(p.float(), -0.5, 0.5) for p in alignment_params]
                
                # Compute loss with validation
                loss, dice_ce_loss, alignment_loss, align_details = self.criterion(
                    outputs, masks, aligned_slices, alignment_params, original_slices
                )
                
                # Check loss health
                is_healthy, issue = self.check_tensor_health(loss, "loss", max_val=100)
                if not is_healthy:
                    print(f"Batch {batch_idx}: {issue} - SKIPPING")
                    self.consecutive_nans += 1
                    torch.cuda.empty_cache()
                    continue
                
                # Additional loss magnitude check
                loss_val = loss.item()
                if loss_val > 50:
                    print(f"Large loss detected: {loss_val:.2f}, scaling down")
                    loss = loss * 0.1
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                
                # Unscale and check gradients
                self.scaler.unscale_(self.optimizer)
                
                # Check gradient health
                has_bad_grad = False
                max_grad_norm = 0.0
                grad_stats = []
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        max_grad_norm = max(max_grad_norm, grad_norm)
                        
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Bad gradient in {name}: norm={grad_norm:.2f}")
                            has_bad_grad = True
                            break
                        
                        if grad_norm > 1000:
                            print(f"Extreme gradient in {name}: {grad_norm:.2f}")
                            has_bad_grad = True
                            break
                
                if has_bad_grad:
                    print(f"Batch {batch_idx}: Bad gradients - SKIPPING")
                    self.optimizer.zero_grad()
                    self.consecutive_nans += 1
                    
                    # Force reduce scale after bad gradients
                    if current_scale > 256:
                        new_scale = current_scale * 0.5
                        print(f"Force reducing scale: {current_scale:.0f} → {new_scale:.0f}")
                        self.scaler._scale.fill_(new_scale)
                    
                    torch.cuda.empty_cache()
                    continue
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.GRAD_CLIP_NORM
                )
                
                # Skip if gradient still too large after clipping
                if grad_norm > 10:
                    print(f"Batch {batch_idx}: grad_norm={grad_norm:.2f} still too high - SKIPPING")
                    self.optimizer.zero_grad()
                    self.consecutive_nans += 1
                    continue
                
                # Safe optimizer step
                self.scaler.step(self.optimizer)
                
                # Update scaler with monitoring
                old_scale = self.scaler.get_scale()
                self.scaler.update()
                new_scale = self.scaler.get_scale()
                
                # Log scale changes
                if old_scale != new_scale:
                    change_pct = (new_scale - old_scale) / old_scale * 100
                    if abs(change_pct) > 10:
                        print(f"Scale change: {old_scale:.0f} → {new_scale:.0f} ({change_pct:+.1f}%)")
                
                # Success! Reset consecutive counter
                self.consecutive_nans = 0
                
                # Accumulate metrics
                total_loss += loss.item()
                total_dice_ce += dice_ce_loss.item()
                total_alignment += alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss
                valid_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice_ce': f'{dice_ce_loss.item():.4f}',
                    'grad': f'{grad_norm:.2f}',
                    'scale': f'{new_scale:.0f}'
                })
                
                # Store scale history
                self.scale_history.append(new_scale)
            
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg:
                    print(f"\nOOM at batch {batch_idx}!")
                    print(f"Scale: {current_scale:.0f}")
                    print(f"Consecutive NaNs: {self.consecutive_nans}")
                    
                    # Aggressive cleanup
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    
                    # Force reduce scale on OOM
                    new_scale = min(current_scale * 0.25, 512)
                    print(f"Reducing scale to {new_scale:.0f}")
                    self.scaler._scale.fill_(new_scale)
                    
                    self.consecutive_nans += 1
                    continue
                else:
                    print(f"\nRuntime error in batch {batch_idx}: {error_msg}")
                    self.optimizer.zero_grad()
                    self.consecutive_nans += 1
                    continue
        
        # Epoch summary
        if valid_batches == 0:
            print("\nNO VALID BATCHES IN THIS EPOCH!")
            return None, None, None, None
        
        avg_loss = total_loss / valid_batches
        avg_dice_ce = total_dice_ce / valid_batches
        avg_alignment = total_alignment / valid_batches
        
        print(f"\nEpoch {epoch} Stats:")
        print(f"   Valid batches: {valid_batches}/{len(self.train_loader)}")
        print(f"   NaN count: {self.nan_count}")
        print(f"   Scale reduced: {self.scale_reduced_count} times")
        print(f"   Final scale: {self.scaler.get_scale():.0f}")
        
        return avg_loss, avg_dice_ce, avg_alignment, {}
    
    def validate(self, epoch):
        """Validate model with NaN protection"""
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
                    # Forward without alignment (faster)
                    with autocast(enabled=True):
                        outputs = self.model(images)
                    
                    outputs = outputs.float()
                    
                    # Simple validation loss (no alignment)
                    from monai.losses import DiceCELoss
                    val_criterion = DiceCELoss(
                        include_background=True,
                        to_onehot_y=True,
                        softmax=True,
                        lambda_dice=self.config.DICE_WEIGHT,
                        lambda_ce=self.config.CE_WEIGHT
                    )
                    val_criterion = val_criterion.to(self.device)
                    
                    masks_for_loss = masks.unsqueeze(1) if masks.ndim == 3 else masks
                    loss = val_criterion(outputs, masks_for_loss)
                    
                    # Check loss health
                    is_healthy, issue = self.check_tensor_health(loss, "val_loss", max_val=100)
                    if not is_healthy:
                        print(f"Validation: {issue} - SKIPPING batch")
                        continue
                    
                    total_val_loss += loss.item()
                    valid_batches += 1
                    
                    # Compute Dice
                    masks_for_metric = masks.unsqueeze(1) if masks.ndim == 3 else masks
                    self.dice_metric(y_pred=outputs, y=masks_for_metric)
                
                except RuntimeError as e:
                    print(f"Validation error: {e}")
                    continue
        
        if valid_batches == 0:
            print("No valid validation batches!")
            return 0.0, float('inf')
        
        dice_result = self.dice_metric.aggregate()
        val_dice = dice_result[0].item() if isinstance(dice_result, (list, tuple)) else dice_result.item()
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
                run_name = f"nan_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project=self.config.WANDB_PROJECT,
                    entity=self.config.WANDB_ENTITY,
                    config=self.config.to_dict(),
                    name=run_name,
                    tags=["brain-stroke", "nan-fixed", "scale-controlled"]
                )
                self.wandb_run_id = wandb.run.id
            
            wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"\n{'='*70}")
        print(f"Starting SCALE-CONTROLLED training")
        print(f"{'='*70}")
        print(f"   Epoch range: {self.start_epoch} → {num_epochs}")
        print(f"   Device: {self.device}")
        print(f"   Max scale: {self.max_scale}")
        print(f"   Initial scale: {self.scaler.get_scale():.0f}")
        print(f"   Gradient clip: {self.config.GRAD_CLIP_NORM}")
        print(f"{'='*70}\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            result = self.train_epoch(epoch + 1)
            
            if result[0] is None:
                print(f"Epoch {epoch+1} FAILED - stopping training")
                break
            
            train_loss, dice_ce_loss, alignment_loss, _ = result
            
            # Validate
            val_dice, val_loss = self.validate(epoch + 1)
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'learning_rate': current_lr,
                'best_dice': self.best_dice,
                'nan_count': self.nan_count,
                'scale': self.scaler.get_scale(),
                'scale_reduced_count': self.scale_reduced_count
            }
            
            self.history.append(metrics)
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"{'='*70}")
            print(f"   Train Loss:    {train_loss:.4f}")
            print(f"   Val Loss:      {val_loss:.4f}")
            print(f"   Val Dice:      {val_dice:.4f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Scale:         {self.scaler.get_scale():.0f}")
            print(f"   NaN Count:     {self.nan_count}")
            print(f"{'='*70}\n")
            
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
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"   Best Dice: {self.best_dice:.4f}")
        print(f"   Total NaNs: {self.nan_count}")
        print(f"   Scale reductions: {self.scale_reduced_count}")
        print(f"{'='*70}\n")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
            