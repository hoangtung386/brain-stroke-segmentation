"""
Fixed Dataset module for Brain Stroke Segmentation
Supports loading adjacent slices for SEAN architecture
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np


class BrainStrokeDataset(Dataset):
    """Custom dataset for brain stroke segmentation with 3D slice support"""
    
    def __init__(self, patient_folders, image_dir, mask_dir, T=1, 
                 transform=None, target_transform=None):
        """
        Args:
            patient_folders: List of patient folder names
            image_dir: Base directory for images
            mask_dir: Base directory for masks
            T: Number of adjacent slices (will use 2T+1 total slices)
            transform: Transformations for images
            target_transform: Transformations for masks
        """
        self.patient_folders = patient_folders
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.T = T
        self.transform = transform
        self.target_transform = target_transform
        
        # Build index: list of (patient_folder, slice_idx, total_slices)
        self.samples = []
        for patient_folder in patient_folders:
            patient_image_path = os.path.join(image_dir, patient_folder)
            if not os.path.isdir(patient_image_path):
                continue
            
            # Get all slice filenames and sort
            slice_files = sorted([
                f for f in os.listdir(patient_image_path) 
                if f.endswith('.png')
            ])
            
            num_slices = len(slice_files)
            if num_slices == 0:
                continue
            
            # Add each valid slice (with enough context for T)
            for i in range(num_slices):
                self.samples.append((patient_folder, i, num_slices, slice_files))
        
        print(f"Dataset initialized with {len(self.samples)} samples from {len(patient_folders)} patients")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_slice(self, patient_folder, slice_files, idx):
        """Load a single slice"""
        image_path = os.path.join(self.image_dir, patient_folder, slice_files[idx])
        mask_path = os.path.join(self.mask_dir, patient_folder, slice_files[idx])
        
        image = Image.open(image_path).convert('L')  # Grayscale for CT
        mask = Image.open(mask_path).convert('L')
        
        return image, mask
    
    def __getitem__(self, index):
        patient_folder, center_idx, num_slices, slice_files = self.samples[index]
        
        # Collect 2T+1 adjacent slices
        slices_to_load = []
        for offset in range(-self.T, self.T + 1):
            slice_idx = center_idx + offset
            # Handle boundaries with replication
            slice_idx = max(0, min(num_slices - 1, slice_idx))
            slices_to_load.append(slice_idx)
        
        # Load all slices
        image_slices = []
        mask_center = None
        
        for i, slice_idx in enumerate(slices_to_load):
            image, mask = self._load_slice(patient_folder, slice_files, slice_idx)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                # Remove channel dim (1, H, W) -> (H, W)
                # This ensures stack creates (2T+1, H, W) instead of (2T+1, 1, H, W)
                if image.shape[0] == 1:
                    image = image.squeeze(0)
            
            image_slices.append(image)
            
            # Only keep mask for center slice
            if i == self.T:  # Center slice
                if self.target_transform:
                    mask_center = self.target_transform(mask)
        
        # Stack slices: (2T+1, H, W)
        image_stack = torch.stack(image_slices, dim=0)
        
        return image_stack, mask_center


def get_transforms(config):
    """Get image and mask transformations"""
    
    # Transform for grayscale CT images
    image_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        # Normalize grayscale CT (single channel)
        transforms.Normalize(mean=[config.MEAN[0]], std=[config.STD[0]])
    ])
    
    # Transform for mask images
    def target_transform(target):
        # Use Nearest Neighbor to preserve mask values
        img = transforms.Resize(
            config.IMAGE_SIZE, 
            interpolation=transforms.InterpolationMode.NEAREST
        )(target)
        img = transforms.functional.pil_to_tensor(img)
        
        # Convert all values > 0 to 1 (binary segmentation)
        # This handles both 255 masks and noisy masks
        img = (img > 0).to(torch.long).squeeze(0)  # Remove channel dim -> (H, W)
        
        return img
    
    return image_transform, target_transform


def prepare_data_paths(config):
    """
    Prepare train and test patient folders
    
    Returns:
        train_folders, test_folders
    """
    image_dir = config.IMAGE_DIR
    
    # Get all patient folders
    all_folders = [
        f for f in os.listdir(image_dir) 
        if os.path.isdir(os.path.join(image_dir, f))
    ]
    
    # Split by patient folders (not individual slices) to avoid data leakage
    train_folders, test_folders = train_test_split(
        all_folders,
        test_size=1 - config.TRAIN_SPLIT,
        random_state=config.SEED
    )
    
    print(f"Number of training patients: {len(train_folders)}")
    print(f"Number of testing patients: {len(test_folders)}")
    
    # Count total slices
    def count_slices(folders):
        total = 0
        for folder in folders:
            folder_path = os.path.join(image_dir, folder)
            total += len([f for f in os.listdir(folder_path) if f.endswith('.png')])
        return total
    
    print(f"Training slices: {count_slices(train_folders)}")
    print(f"Testing slices: {count_slices(test_folders)}")
    
    return train_folders, test_folders


def create_dataloaders(config):
    """
    Create train and test dataloaders
    
    Returns:
        train_loader, test_loader
    """
    # Get patient folder splits
    train_folders, test_folders = prepare_data_paths(config)
    
    # Get transformations
    image_transform, target_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = BrainStrokeDataset(
        train_folders,
        config.IMAGE_DIR,
        config.MASK_DIR,
        T=config.T,
        transform=image_transform,
        target_transform=target_transform
    )
    
    test_dataset = BrainStrokeDataset(
        test_folders,
        config.IMAGE_DIR,
        config.MASK_DIR,
        T=config.T,
        transform=image_transform,
        target_transform=target_transform
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader
