"""
Optimized to prevent NaN and OOM issues
"""
import os

class Config:
    # Basic settings
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
    
    # Batch size
    BATCH_SIZE = 4
    NUM_EPOCHS = 300
    LEARNING_RATE = 1e-4
    
    # DataLoader parameters
    NUM_WORKERS = 4
    CACHE_RATE = 0
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Model architecture
    T = 1                       # Number of adjacent slices
    NUM_PARTITIONS_H = 4
    NUM_PARTITIONS_W = 4
    GLOBAL_IMPACT = 0.3
    LOCAL_IMPACT = 0.7
    
    # Normalization
    MEAN = [55.1385 / 255.0]    # = 0.2162
    STD = [46.2948 / 255.0]     # = 0.1841
    
    WEIGHT_DECAY = 1e-4
    
    # Training stability
    GRAD_CLIP_NORM = 0.5
    USE_AMP = True
    DEBUG_MODE = False
    DETECT_ANOMALY = False
    
    # Loss weights
    DICE_WEIGHT = 0.7
    CE_WEIGHT = 0.3
    FOCAL_WEIGHT = 1.0
    ALIGNMENT_WEIGHT = 0.005
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
    EARLY_STOPPING_PATIENCE = 30
    
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
            'grad_clip_norm': cls.GRAD_CLIP_NORM,
            'use_amp': cls.USE_AMP,
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        print(f"✅ Directories created: {cls.OUTPUT_DIR}, {cls.CHECKPOINT_DIR}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("⚙️ CURRENT CONFIGURATION")
        print("="*60)
        print(f"Batch Size:        {cls.BATCH_SIZE}")
        print(f"Learning Rate:     {cls.LEARNING_RATE}")
        print(f"Epochs:            {cls.NUM_EPOCHS}")
        print(f"Image Size:        {cls.IMAGE_SIZE}")
        print(f"Gradient Clip:     {cls.GRAD_CLIP_NORM}")
        print(f"Alignment Weight:  {cls.ALIGNMENT_WEIGHT}")
        print(f"Use AMP:           {cls.USE_AMP}")
        print(f"Debug Mode:        {cls.DEBUG_MODE}")
        print("="*60 + "\n")


# Configuration profiles for different use cases
class FastDebugConfig(Config):
    """Fast debug configuration for quick testing"""
    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    NUM_WORKERS = 2
    DEBUG_MODE = True
    DETECT_ANOMALY = True
    USE_WANDB = False


class ConservativeConfig(Config):
    """Ultra-conservative config for maximum stability"""
    BATCH_SIZE = 1
    LEARNING_RATE = 5e-5
    GRAD_CLIP_NORM = 0.25
    ALIGNMENT_WEIGHT = 0.001
    USE_AMP = False


class AggressiveConfig(Config):
    """Aggressive config for faster training (use with caution)"""
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4
    NUM_WORKERS = 8
    GRAD_CLIP_NORM = 2.0
    ALIGNMENT_WEIGHT = 0.02
    