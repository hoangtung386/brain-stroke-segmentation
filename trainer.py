"""
Trainer class for model training
"""
import os
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


class Trainer:
    """Trainer class for brain stroke segmentation"""
    
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
        
        # Loss and optimizer
        self.criterion = DiceLoss(to_onehot_y=True, softmax=True)
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=config.SCHEDULER_PATIENCE, 
            factor=config.SCHEDULER_FACTOR
        )
        self.dice_metric = DiceMetric(include_background=False, reduction='mean')
        
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
            print(f"Best model saved! Dice: {val_dice:.4f}")
    
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
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        self.dice_metric.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                self.dice_metric(y_pred=outputs, y=masks)
        
        val_dice = self.dice_metric.aggregate().item()
        return val_dice
    
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
                    tags=["brain-stroke", "segmentation"]
                )
                self.wandb_run_id = wandb.run.id
            
            wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"\nStarting training from epoch {self.start_epoch} to {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self.train_epoch(epoch + 1)
            val_dice = self.validate(epoch + 1)
            self.scheduler.step(val_dice)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_dice': val_dice,
                'learning_rate': current_lr,
                'best_dice': self.best_dice
            }
            
            self.history.append(metrics)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Best Dice: {self.best_dice:.4f}")
            
            if self.use_wandb:
                import wandb
                wandb.log(metrics)
            
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
            
            self.save_checkpoint(epoch, val_dice, is_best)
            self.save_history_csv()
        
        print(f"\nTraining complete! Best Dice: {self.best_dice:.4f}")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
