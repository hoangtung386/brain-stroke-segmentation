"""
FIXED Configuration - Normalization values corrected
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
    BATCH_SIZE = 8  # Giảm xuống 8 để ổn định hơn
    NUM_EPOCHS = 300
    LEARNING_RATE = 5e-4  # Giảm từ 1e-3 để tránh gradient explosion
    
    # DataLoader parameters
    NUM_WORKERS = 16
    CACHE_RATE = 0
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Model architecture
    T = 1
    NUM_PARTITIONS_H = 4
    NUM_PARTITIONS_W = 4
    GLOBAL_IMPACT = 0.3
    LOCAL_IMPACT = 0.7
    
    # NORMALIZED VALUES (after ToTensor)
    # Original: 55.1385 ± 46.2948 (range 0-255)
    # After ToTensor (÷255): values in [0, 1]
    MEAN = [55.1385 / 255.0]  # Optimization
    STD = [46.2948 / 255.0]   # = 0.1841
    
    LEARNING_RATE = 1e-4      # Reduced from 5e-4
    WEIGHT_DECAY = 1e-4

    # Training Stability
    GRAD_CLIP_NORM = 0.5      # Reduced from 1.0
    USE_AMP = True            # Automatic Mixed Precision
    DETECT_ANOMALY = False    # Set True for debugging

    # Loss Weights
    DICE_WEIGHT = 1.0
    CE_WEIGHT = 1.0           # Restored CE_WEIGHT
    FOCAL_WEIGHT = 1.0
    ALIGNMENT_WEIGHT = 0.0    # Temporarily disabled from 0.3
    PERCEPTUAL_WEIGHT = 0.1   
    
    # W&B settings
    USE_WANDB = True
    WANDB_PROJECT = "Advanced-Lightweight-CNN-segment-Stroke"
    WANDB_ENTITY = None
    
    # Scheduler parameters
    SCHEDULER_T0 = 10
    SCHEDULER_T_MULT = 2
    SCHEDULER_ETA_MIN = 1e-6
    
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
        