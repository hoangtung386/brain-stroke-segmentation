"""
STABILITY-FOCUSED Configuration
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
    BATCH_SIZE = 8  # Keep at 8 for stability
    NUM_EPOCHS = 300
    LEARNING_RATE = 5e-5  # Conservative LR
    
    # DataLoader parameters
    NUM_WORKERS = 8
    CACHE_RATE = 0
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Model architecture
    T = 1
    NUM_PARTITIONS_H = 4
    NUM_PARTITIONS_W = 4
    GLOBAL_IMPACT = 0.3
    LOCAL_IMPACT = 0.7
    
    # Normalization (after ToTensor)
    MEAN = [55.1385 / 255.0]
    STD = [46.2948 / 255.0]
    
    WEIGHT_DECAY = 1e-4

    # CRITICAL: Training Stability Settings
    GRAD_CLIP_NORM = 0.5          # Increased from 0.25 for safety
    USE_AMP = True                # Keep AMP enabled
    DEBUG_MODE = False            # Turn off for production
    DETECT_ANOMALY = False        # Turn off (causes slowdown)
    
    # NEW: AMP Scale Control
    AMP_INIT_SCALE = 256          # Start lower (was 512)
    AMP_MAX_SCALE = 4096          # Hard cap
    AMP_GROWTH_FACTOR = 1.5       # Slower growth (default 2.0)
    AMP_BACKOFF_FACTOR = 0.25     # Faster reduction (default 0.5)
    AMP_GROWTH_INTERVAL = 500     # Wait longer (default 2000)

    # ADJUSTED: Loss Weights (reduced alignment)
    DICE_WEIGHT = 0.5
    CE_WEIGHT = 0.5
    FOCAL_WEIGHT = 1.0
    ALIGNMENT_WEIGHT = 0.05       # REDUCED from 0.1 - will increase gradually
    PERCEPTUAL_WEIGHT = 0.0       # Disabled for stability
    
    # W&B settings
    USE_WANDB = True
    WANDB_PROJECT = "Advanced-Lightweight-CNN-segment-Stroke"
    WANDB_ENTITY = None
    
    # Scheduler parameters
    SCHEDULER_T0 = 10
    SCHEDULER_T_MULT = 2
    SCHEDULER_ETA_MIN = 1e-6
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 30  # Increased patience
    
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
            'amp_max_scale': cls.AMP_MAX_SCALE,
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        print(f"Created directories: {cls.OUTPUT_DIR}, {cls.CHECKPOINT_DIR}")
        