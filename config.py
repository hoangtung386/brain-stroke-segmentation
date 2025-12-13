"""
Fixed Configuration file for Brain Stroke Segmentation project
"""
import os

class Config:
    """Configuration class for training parameters"""
    
    # Seed for reproducibility
    SEED = 42
    
    # Data paths
    BASE_PATH = './data' 
    IMAGE_DIR = os.path.join(BASE_PATH, 'images')
    MASK_DIR = os.path.join(BASE_PATH, 'masks')
    OUTPUT_DIR = './outputs'
    CHECKPOINT_DIR = './checkpoints'

    # Data split
    TRAIN_SPLIT = 0.8
    
    # Model parameters
    NUM_CHANNELS = 1
    NUM_CLASSES = 2
    INIT_FEATURES = 32
    IMAGE_SIZE = (512, 512)
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 300
    LEARNING_RATE = 1e-3
    
    # DataLoader parameters
    NUM_WORKERS = 4
    CACHE_RATE = 0
    PIN_MEMORY = True  # Changed to True for better performance
    PERSISTENT_WORKERS = True
    
    # Model architecture
    T = 1  # Number of adjacent slices (will use 2T+1 = 3 slices total)
    NUM_PARTITIONS_H = 4
    NUM_PARTITIONS_W = 4
    GLOBAL_IMPACT = 0.3
    LOCAL_IMPACT = 0.7
    
    # Normalization parameters (grayscale CT)
    # These should be computed from your dataset
    # For now, using standard values
    MEAN = [0.216229]  # Single channel for grayscale
    STD = [0.335106]   # Single channel for grayscale
    
    # Loss weights
    DICE_WEIGHT = 0.5
    CE_WEIGHT = 0.5
    ALIGNMENT_WEIGHT = 0.3
    
    # W&B settings
    USE_WANDB = True
    WANDB_PROJECT = "Advanced-Lightweight-CNN-segment-Stroke"
    WANDB_ENTITY = None  # Your W&B username
    
    # Scheduler parameters (using CosineAnnealingWarmRestarts now)
    SCHEDULER_T0 = 10
    SCHEDULER_T_MULT = 2
    SCHEDULER_ETA_MIN = 1e-6
    
    # Gradient clipping
    GRAD_CLIP_NORM = 1.0
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            'seed': cls.SEED,
            'train_split': cls.TRAIN_SPLIT,
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'image_size': cls.IMAGE_SIZE,
            'init_features': cls.INIT_FEATURES,
            'num_channels': cls.NUM_CHANNELS,
            'num_classes': cls.NUM_CLASSES,
            'T': cls.T,
            'global_impact': cls.GLOBAL_IMPACT,
            'local_impact': cls.LOCAL_IMPACT,
            'dice_weight': cls.DICE_WEIGHT,
            'ce_weight': cls.CE_WEIGHT,
            'alignment_weight': cls.ALIGNMENT_WEIGHT,
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        print(f"Created directories: {cls.OUTPUT_DIR}, {cls.CHECKPOINT_DIR}")
    
    @classmethod
    def compute_normalization_stats(cls, image_dir):
        """
        Compute mean and std from dataset for normalization
        Run this once before training to get proper values
        """
        from PIL import Image
        import numpy as np
        from tqdm import tqdm
        
        print("Computing normalization statistics...")
        
        pixel_values = []
        
        # Walk through all images
        for root, dirs, files in os.walk(image_dir):
            for file in tqdm(files):
                if file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    pixel_values.append(img_array.flatten())
        
        # Compute statistics
        all_pixels = np.concatenate(pixel_values)
        mean = all_pixels.mean()
        std = all_pixels.std()
        
        print(f"\nDataset Statistics:")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std:  {std:.6f}")
        print(f"\nUpdate config.py with:")
        print(f"  MEAN = [{mean:.6f}]")
        print(f"  STD = [{std:.6f}]")
        
        return mean, std
