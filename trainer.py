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
            dice_weight=config.DICE_WEIGHT,
            ce_weight=config.CE_WEIGHT,
            alignment_weight=config.ALIGNMENT_WEIGHT,
            use_alignment=True
        )
        self.criterion.to(self.device)
        
        # Optimizer với weight decay mạnh hơn
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # AMP Scaler với init_scale thấp hơn
        self.scaler = GradScaler(init_scale=512)
        
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
        self.alignment_warmup_epochs = 10  # Warmup epochs for alignment loss
        
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

    def get_alignment_weight(self, epoch):
        """Gradually increase alignment weight"""
        if epoch < self.alignment_warmup_epochs:
            return self.config.ALIGNMENT_WEIGHT * (epoch / self.alignment_warmup_epochs)
        return self.config.ALIGNMENT_WEIGHT
    
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
        """Train one epoch with enhanced NaN protection"""
        self.model.train()
        self.nan_count = 0  # Reset NaN counter at start of epoch
        total_loss = 0
        total_dice_ce = 0
        total_alignment = 0
        total_symmetry = 0
        total_regularization = 0
        total_edge = 0
        
        valid_batches = 0
        nan_in_epoch = False
        
        # Get current alignment weight
        current_alignment_weight = self.get_alignment_weight(epoch)
        self.criterion.alignment_weight = current_alignment_weight
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train] (AlignW={current_alignment_weight:.4f})")
        for batch_idx, (images, masks) in enumerate(pbar):
            # 🔹 CHECK 1: Validate input data range
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"Skipping batch {batch_idx}: Invalid input data")
                continue

            # Clamp input to prevent extreme values
            images = torch.clamp(images, -10, 10)
            
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                with autocast():
                    outputs, aligned_slices, alignment_params = self.model(
                        images, return_alignment=True
                    )
                
                # 🔹 CHECK 2: Validate model outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Skipping batch {batch_idx}: NaN in model outputs")
                    self.nan_count += 1
                    if self.nan_count > 5:
                        print("Too many NaN batches! Stopping...")
                        return None, None, None, None
                    continue
                
                # Clamp outputs
                outputs = torch.clamp(outputs, -20, 20)
                
                # Move to float32 for loss computation
                outputs = outputs.float()
                original_slices = [s.float() for s in [images[:, i:i+1, :, :] for i in range(images.shape[1])]]
                
                if aligned_slices is not None:
                    aligned_slices = [torch.clamp(s.float(), -10, 10) for s in aligned_slices]
                if alignment_params is not None:
                    alignment_params = [torch.clamp(p.float(), -1, 1) for p in alignment_params]
                
                # Compute loss
                loss, dice_ce_loss, alignment_loss, align_details = self.criterion(
                    outputs, masks, aligned_slices, alignment_params, original_slices
                )
                
                # 🔹 CHECK 3: Validate loss values
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Skipping batch {batch_idx}: NaN/Inf in loss")
                    self.nan_count += 1
                    continue
                
                # 🔹 CHECK 4: Loss magnitude check
                if loss.item() > 100:
                    print(f"Warning: Very large loss ({loss.item():.2f})")
                    # Scale down loss
                    loss = loss * 0.1
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                
                # 🔹 CHECK 5: Gradient validation
                self.scaler.unscale_(self.optimizer)
                
                # Check for NaN gradients
                has_nan_grad = False
                max_grad_norm = 0.0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN gradient in {name}")
                            has_nan_grad = True
                            break
                        max_grad_norm = max(max_grad_norm, param.grad.abs().max().item())
                
                if has_nan_grad:
                    print(f"Skipping batch {batch_idx}: NaN in gradients")
                    self.optimizer.zero_grad()
                    self.nan_count += 1
                    continue
                
                # 🔹 CHECK 6: Monitor gradient norm
                if max_grad_norm > 100:
                    print(f"Very large gradient: {max_grad_norm:.2f}")
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.GRAD_CLIP_NORM
                )
                
                # 🔹 CHECK 7: Skip update if gradient is too large
                if grad_norm > 50:
                    print(f"Skipping optimizer step: grad_norm={grad_norm:.2f}")
                    self.optimizer.zero_grad()
                    continue
                
                self.scaler.step(self.optimizer)
                
                # 🔹 CHECK 8: Monitor scaler
                old_scale = self.scaler.get_scale()
                self.scaler.update()
                new_scale = self.scaler.get_scale()
                
                if new_scale < old_scale * 0.5:
                    print(f"Scaler reduced: {old_scale:.0f} → {new_scale:.0f}")

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
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    self.nan_count += 1 # Count OOM as error to be aware
                    continue
                else:
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
                    
                    # Chỉ tính Dice + CE, bỏ qua Alignment Loss
                    from monai.losses import DiceCELoss
                    val_criterion = DiceCELoss(
                        include_background=True,
                        to_onehot_y=True,
                        softmax=True,
                        lambda_dice=self.config.DICE_WEIGHT,
                        lambda_ce=self.config.CE_WEIGHT
                    )
                    val_criterion = val_criterion.to(self.device)
                    
                    if masks.ndim == 3:
                        masks_for_loss = masks.unsqueeze(1)
                    else:
                        masks_for_loss = masks
                    
                    loss = val_criterion(outputs, masks_for_loss)
                    
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
            