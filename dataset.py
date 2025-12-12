"""
Dataset module for Brain Stroke Segmentation
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class BrainStrokeDataset(Dataset):
    """Custom dataset for brain stroke segmentation"""
    
    def __init__(self, image_paths, mask_paths, transform=None, target_transform=None):
        """
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            transform: Transformations for images
            target_transform: Transformations for masks
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform
        
        assert len(self.image_paths) == len(self.mask_paths), \
            "Image and mask lists must have the same length"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask


def get_transforms(config):
    """Get image and mask transformations"""
    
    # Transform for RGB images
    image_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # Transform for mask images
    def target_transform(target):
        # QUAN TRỌNG: Dùng Nearest Neighbor để không làm hỏng giá trị pixel của mask
        img = transforms.Resize(config.IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST)(target)
        img = transforms.functional.pil_to_tensor(img)
        
        # FIX LỖI Ở ĐÂY: Chuyển tất cả giá trị > 0 thành 1
        # Điều này xử lý cả trường hợp mask là 255 hoặc mask bị noise
        img = (img > 0).to(torch.long)
        
        return img
    
    return image_transform, target_transform


def prepare_data_paths(config):
    """
    Prepare train and test data paths
    
    Returns:
        train_image_paths, train_mask_paths, test_image_paths, test_mask_paths
    """
    image_dir = config.IMAGE_DIR
    mask_dir = config.MASK_DIR
    
    # Get all subfolders
    all_subfolders = [
        f for f in os.listdir(image_dir) 
        if os.path.isdir(os.path.join(image_dir, f))
    ]
    
    # Split subfolders into train and test
    train_subfolders, test_subfolders = train_test_split(
        all_subfolders,
        test_size=1 - config.TRAIN_SPLIT,
        random_state=config.SEED
    )
    
    print(f"Number of training subfolders: {len(train_subfolders)}")
    print(f"Number of testing subfolders: {len(test_subfolders)}")
    
    # Collect file paths
    train_image_paths = []
    train_mask_paths = []
    test_image_paths = []
    test_mask_paths = []
    
    # Training paths
    for subfolder in train_subfolders:
        image_subfolder_path = os.path.join(image_dir, subfolder)
        mask_subfolder_path = os.path.join(mask_dir, subfolder)
        
        for filename in os.listdir(image_subfolder_path):
            if filename.endswith('.png'):
                train_image_paths.append(os.path.join(image_subfolder_path, filename))
                train_mask_paths.append(os.path.join(mask_subfolder_path, filename))
    
    # Testing paths
    for subfolder in test_subfolders:
        image_subfolder_path = os.path.join(image_dir, subfolder)
        mask_subfolder_path = os.path.join(mask_dir, subfolder)
        
        for filename in os.listdir(image_subfolder_path):
            if filename.endswith('.png'):
                test_image_paths.append(os.path.join(image_subfolder_path, filename))
                test_mask_paths.append(os.path.join(mask_subfolder_path, filename))
    
    print(f"Training images: {len(train_image_paths)}")
    print(f"Training masks: {len(train_mask_paths)}")
    print(f"Testing images: {len(test_image_paths)}")
    print(f"Testing masks: {len(test_mask_paths)}")
    
    return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths


def create_dataloaders(config):
    """
    Create train and test dataloaders
    
    Returns:
        train_loader, test_loader
    """
    # Get data paths
    train_img_paths, train_mask_paths, test_img_paths, test_mask_paths = \
        prepare_data_paths(config)
    
    # Get transformations
    image_transform, target_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = BrainStrokeDataset(
        train_img_paths, train_mask_paths,
        transform=image_transform,
        target_transform=target_transform
    )
    
    test_dataset = BrainStrokeDataset(
        test_img_paths, test_mask_paths,
        transform=image_transform,
        target_transform=target_transform
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader
