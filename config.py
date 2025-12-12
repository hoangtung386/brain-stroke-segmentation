"""
Configuration file for Brain Stroke Segmentation project
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
    NUM_CHANNELS = 3
    NUM_CLASSES = 2
    INIT_FEATURES = 32
    IMAGE_SIZE = (512, 512)
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    LEARNING_RATE = 1e-3
    
    # DataLoader parameters
    NUM_WORKERS = 4  # Tăng lên cho RTX 3090
    CACHE_RATE = 0
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Model architecture
    T = 1  # Number of adjacent slices
    NUM_PARTITIONS_H = 4
    NUM_PARTITIONS_W = 4
    GLOBAL_IMPACT = 0.3
    LOCAL_IMPACT = 0.7
    
    # Normalization parameters (RGB)
    MEAN = [0.216229, 0.216229, 0.216229]
    STD = [0.335106, 0.335106, 0.335106]
    
    # W&B settings
    USE_WANDB = True
    WANDB_PROJECT = "my-2D-Unet-segment-Stroke"
    WANDB_ENTITY = None  # Your W&B username
    
    # Scheduler parameters
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
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
            'cache_rate': cls.CACHE_RATE,
            'num_workers': cls.NUM_WORKERS,
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        print(f"Created directories: {cls.OUTPUT_DIR}, {cls.CHECKPOINT_DIR}")
