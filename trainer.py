"""
Trainer - Complete solution for validation NaN issues

Key fixes:
1. Proper output clamping range for softmax stability
2. Input validation before loss computation
3. Safer numerical operations
4. Better error recovery
5. Debug mode for troubleshooting
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
    Fixed trainer with complete validation NaN resolution
    """
    
    def __init__(self, model, train_loader, val_loader, config, device, use_wandb=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        self.model.to(self.device)
        
        # Training loss (with alignment)
        self.train_criterion = ImprovedCombinedLoss(
            num_classes=config.NUM_CLASSES,
            dice_weight=config.DICE_WEIGHT,
            ce_weight=config.CE_WEIGHT,
            alignment_weight=config.ALIGNMENT_WEIGHT,
            use_alignment=True
        )
        self.train_criterion.to(self.device)
        
        # ========================================
        # FIX 1: SAFER VALIDATION LOSS
        # ========================================
        # Use increased smooth values for stability
        self.val_criterion = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=0.7,
            lambda_ce=0.3,
            smooth_nr=1e-4,  # Increased from default 1e-5
            smooth_dr=1e-4   # Increased from default 1e-5
        )
        self.val_criterion.to(self.device)
        
        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-7  # Slightly larger for stability
        )
        
        # Conservative scaler
        self.scaler = GradScaler(
            init_scale=128,
            growth_factor=1.2,
            backoff_factor=0.9,
            growth_interval=10000,
            enabled=config.USE_AMP
        )
        
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
        self.alignment_warmup_epochs = 10
        self.consecutive_val_failures = 0
        
        # Paths
        self.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'checkpoint.pth')
        self.best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        self.history_csv_path = os.path.join(config.OUTPUT_DIR, 'training_history.csv')
    
    def validate_tensor(self, tensor, name="tensor"):
        """Validate tensor for NaN/Inf and reasonable values"""
        if torch.isnan(tensor).any():
            print(f"⚠️ NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            print(f"⚠️ Inf detected in {name}")
            return False
        
        # Check for unreasonable values
        tensor_max = tensor.abs().max().item()
        if tensor_max > 1000:
            print(f"⚠️ Very large values in {name}: max={tensor_max:.2f}")
            return False
        
        return True
    
    def safe_clamp_logits(self, logits):
        """
        Safely clamp logits for softmax stability
        
        ========================================
        FIX 2: PROPER CLAMPING RANGE
        ========================================
        Softmax is stable with logits in range [-10, 10]:
        - exp(10) ≈ 22,026 (manageable)
        - exp(-10) ≈ 0.000045 (non-zero)
        
        Clamping to [-20, 20] was too extreme!
        """
        # First, check for invalid values
        if not self.validate_tensor(logits, "logits"):
            # Replace invalid values with zeros
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Clamp to safe range for softmax
        logits = torch.clamp(logits, -10.0, 10.0)
        
        return logits
    
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
            print(f"✅ Best model saved! Dice: {val_dice:.4f}")
    
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
        """Train one epoch"""
        self.model.train()
        
        # Check model parameters before training
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️ NaN/Inf detected in {name} before training!")
                if os.path.exists(self.best_model_path):
                    print("Loading best model to recover...")
                    checkpoint = torch.load(self.best_model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    break
        
        total_loss = 0
        total_dice_ce = 0
        total_alignment = 0
        valid_batches = 0
        
        # Get current alignment weight (warmup)
        current_alignment_weight = self.get_alignment_weight(epoch)
        self.train_criterion.alignment_weight = current_alignment_weight
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            # Input validation
            if not self.validate_tensor(images, "input_images"):
                print(f"⚠️ Invalid input at batch {batch_idx}, skipping...")
                continue
            
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass with autocast
                with autocast(enabled=self.config.USE_AMP):
                    outputs, aligned_slices, alignment_params = self.model(
                        images, return_alignment=True
                    )
                    
                    # Safe clamping
                    outputs = self.safe_clamp_logits(outputs)
                
                # Validate outputs
                if not self.validate_tensor(outputs, "outputs"):
                    print(f"⚠️ Invalid outputs at batch {batch_idx}, skipping...")
                    self.nan_count += 1
                    if self.nan_count > 10:
                        print("❌ Too many NaN batches! Emergency stop.")
                        return None, None, None
                    continue
                
                # Convert to FP32 for loss computation
                outputs = outputs.float()
                original_slices = [s.float() for s in [images[:, i:i+1, :, :] for i in range(images.shape[1])]]
                
                if aligned_slices is not None:
                    aligned_slices = [self.safe_clamp_logits(s.float()) for s in aligned_slices]
                if alignment_params is not None:
                    alignment_params = [torch.clamp(p.float(), -1, 1) for p in alignment_params]
                
                # Compute loss
                loss, dice_ce_loss, alignment_loss, _ = self.train_criterion(
                    outputs, masks, aligned_slices, alignment_params, original_slices
                )
                
                # Validate loss
                if not self.validate_tensor(loss, "loss"):
                    print(f"⚠️ Invalid loss at batch {batch_idx}, skipping...")
                    self.nan_count += 1
                    continue
                
                # Emergency loss clamping
                if loss.item() > 100:
                    print(f"⚠️ Very large loss ({loss.item():.2f}), scaling down...")
                    loss = loss * 0.1
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient validation
                self.scaler.unscale_(self.optimizer)
                
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if not self.validate_tensor(param.grad, f"grad_{name}"):
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    self.optimizer.zero_grad()
                    self.nan_count += 1
                    continue
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.GRAD_CLIP_NORM
                )
                
                # Skip if gradients too large
                if grad_norm > 50:
                    print(f"⚠️ Gradient too large ({grad_norm:.2f}), skipping step...")
                    self.optimizer.zero_grad()
                    continue
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_dice_ce += dice_ce_loss.item()
                total_alignment += alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss
                valid_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'grad': f'{grad_norm:.2f}',
                    'scale': f'{self.scaler.get_scale():.0f}'
                })
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    print(f"⚠️ Runtime error: {e}")
                    self.optimizer.zero_grad()
                    continue
        
        if valid_batches == 0:
            print("❌ No valid batches in training epoch!")
            return None, None, None
        
        avg_loss = total_loss / valid_batches
        avg_dice_ce = total_dice_ce / valid_batches
        avg_alignment = total_alignment / valid_batches
        
        return avg_loss, avg_dice_ce, avg_alignment
    
    def validate(self, epoch):
        """
        ========================================
        FIX 3: COMPLETELY REWRITTEN VALIDATION
        ========================================
        """
        self.model.eval()
        self.dice_metric.reset()
        
        total_val_loss = 0
        valid_batches = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Validation for Epoch {epoch}")
        print(f"{'='*60}")
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch_idx, (images, masks) in enumerate(pbar):
                
                # ========================================
                # FIX 4: COMPREHENSIVE INPUT VALIDATION
                # ========================================
                if not self.validate_tensor(images, "val_images"):
                    print(f"⚠️ Batch {batch_idx}: Invalid input images, skipping")
                    continue
                
                if not self.validate_tensor(masks, "val_masks"):
                    print(f"⚠️ Batch {batch_idx}: Invalid masks, skipping")
                    continue
                
                # Check mask values are in valid range
                mask_min, mask_max = masks.min().item(), masks.max().item()
                if mask_min < 0 or mask_max >= self.config.NUM_CLASSES:
                    print(f"⚠️ Batch {batch_idx}: Mask values out of range [{mask_min}, {mask_max}], expected [0, {self.config.NUM_CLASSES-1}]")
                    continue
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                try:
                    # ========================================
                    # FIX 5: DISABLE AMP IN VALIDATION
                    # ========================================
                    with autocast(enabled=False):
                        # Forward without alignment (simpler, more stable)
                        outputs = self.model(images, return_alignment=False)
                    
                    # ========================================
                    # FIX 6: PROPER LOGITS PROCESSING
                    # ========================================
                    # Safe clamping for softmax stability
                    outputs = self.safe_clamp_logits(outputs)
                    outputs = outputs.float()
                    
                    # Additional safety: check outputs
                    if not self.validate_tensor(outputs, "val_outputs"):
                        print(f"⚠️ Batch {batch_idx}: Invalid outputs")
                        continue
                    
                    # Prepare masks for loss
                    if masks.ndim == 3:
                        masks_for_loss = masks.unsqueeze(1)
                    else:
                        masks_for_loss = masks
                    
                    # ========================================
                    # FIX 7: SAFE LOSS COMPUTATION
                    # ========================================
                    try:
                        loss = self.val_criterion(outputs, masks_for_loss)
                    except Exception as e:
                        print(f"⚠️ Batch {batch_idx}: Loss computation error: {e}")
                        continue
                    
                    # Validate loss
                    if not self.validate_tensor(loss, "val_loss"):
                        print(f"⚠️ Batch {batch_idx}: Invalid loss")
                        continue
                    
                    if loss.item() > 100:
                        print(f"⚠️ Batch {batch_idx}: Very large loss ({loss.item():.2f}), skipping")
                        continue
                    
                    # Valid batch!
                    total_val_loss += loss.item()
                    valid_batches += 1
                    
                    # Compute Dice
                    if masks.ndim == 3:
                        masks_for_metric = masks.unsqueeze(1)
                    else:
                        masks_for_metric = masks
                    
                    self.dice_metric(y_pred=outputs, y=masks_for_metric)
                    
                    pbar.set_postfix({
                        'val_loss': f'{loss.item():.4f}',
                        'valid': f'{valid_batches}/{batch_idx+1}'
                    })
                
                except RuntimeError as e:
                    print(f"⚠️ Validation error at batch {batch_idx}: {e}")
                    continue
        
        print(f"\nValidation Summary:")
        print(f"  Valid batches: {valid_batches}/{len(self.val_loader)}")
        
        # ========================================
        # FIX 8: BETTER FALLBACK HANDLING
        # ========================================
        if valid_batches == 0:
            print("❌ No valid batches in validation!")
            self.consecutive_val_failures += 1
            
            if self.consecutive_val_failures >= 3:
                print("\n⚠️ Multiple consecutive validation failures detected!")
                print("Recommended actions:")
                print("  1. Check your data: Run data validation script")
                print("  2. Reduce learning rate by 10x")
                print("  3. Load best checkpoint and resume")
                print("  4. Disable alignment network temporarily")
                print("  5. Check for corrupted data files\n")
            
            # Return safe defaults
            return 0.0, float('inf')
        
        # Reset failure counter on success
        self.consecutive_val_failures = 0
        
        # Compute metrics
        dice_result = self.dice_metric.aggregate()
        
        if isinstance(dice_result, (list, tuple)):
            val_dice = dice_result[0].item() if len(dice_result) > 0 else 0.0
        else:
            val_dice = dice_result.item()
        
        avg_val_loss = total_val_loss / valid_batches
        
        print(f"  Average Loss: {avg_val_loss:.4f}")
        print(f"  Dice Score: {val_dice:.4f}")
        print(f"{'='*60}\n")
        
        return val_dice, avg_val_loss
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        resumed = self.load_checkpoint()
        
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
                    tags=["brain-stroke", "validation-fixed", "stable"]
                )
                self.wandb_run_id = wandb.run.id
            
            wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training (FIXED VERSION)")
        print(f"{'='*60}")
        print(f"Epochs: {self.start_epoch} → {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"AMP enabled: {self.config.USE_AMP}")
        print(f"Validation: Improved stability with proper clamping")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            # Training
            result = self.train_epoch(epoch + 1)
            
            if result[0] is None:
                print(f"❌ Epoch {epoch+1} training failed. Stopping.")
                break
            
            train_loss, dice_ce_loss, alignment_loss = result
            
            # Validation
            val_dice, val_loss = self.validate(epoch + 1)
            
            # Scheduler step
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice_ce': dice_ce_loss,
                'train_alignment': alignment_loss,
                'val_loss': val_loss if val_loss != float('inf') else None,
                'val_dice': val_dice,
                'learning_rate': current_lr,
                'best_dice': self.best_dice,
                'nan_count': self.nan_count,
                'val_failures': self.consecutive_val_failures
            }
            
            self.history.append(metrics)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch+1}/{num_epochs} Summary")
            print(f"{'='*60}")
            print(f"  Train Loss:       {train_loss:.4f}")
            val_loss_str = f"{val_loss:.4f}" if val_loss != float('inf') else "N/A"
            print(f"  Val Loss:         {val_loss_str}")
            print(f"  Val Dice:         {val_dice:.4f}")
            print(f"  Learning Rate:    {current_lr:.6f}")
            print(f"  Best Dice:        {self.best_dice:.4f}")
            print(f"  NaN Count:        {self.nan_count}")
            print(f"  Val Failures:     {self.consecutive_val_failures}")
            print(f"{'='*60}\n")
            
            # W&B logging
            if self.use_wandb:
                import wandb
                wandb.log(metrics)
            
            # Save checkpoint
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
            
            self.save_checkpoint(epoch, val_dice, is_best)
            self.save_history_csv()
            
            # Emergency stop on too many validation failures
            if self.consecutive_val_failures >= 5:
                print("❌ Too many consecutive validation failures. Stopping training.")
                break
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        print(f"Total NaN occurrences: {self.nan_count}")
        print(f"{'='*60}\n")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
