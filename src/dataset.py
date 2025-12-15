"""
PyTorch Dataset class for Bone Age Prediction
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    TRAIN_AUGMENTATION
)


class BoneAgeDataset(Dataset):
    """
    PyTorch Dataset for Bone Age Prediction
    
    Args:
        dataframe: DataFrame with columns: image_path, bone_age_years, sex_binary, age_category
        transform: torchvision transforms to apply
        augment: Whether to apply data augmentation (for training)
    """
    
    def __init__(self, dataframe, transform=None, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
        
        # Validate required columns
        required_cols = ['image_path', 'bone_age_years', 'sex_binary', 'age_category']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        if isinstance(img_path, str):
            img_path = Path(img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        bone_age = torch.tensor(row['bone_age_years'], dtype=torch.float32)
        sex = torch.tensor(row['sex_binary'], dtype=torch.long)
        age_category = torch.tensor(row['age_category'], dtype=torch.long)
        
        return {
            'image': image,
            'bone_age': bone_age,
            'sex': sex,
            'age_category': age_category,
            'image_id': row.get('image_id', idx)
        }


def get_train_transforms():
    """
    Get training transforms with augmentation
    Optimized for medical images - moderate augmentation
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomRotation(
            degrees=TRAIN_AUGMENTATION['rotation_range'],
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomAffine(
            degrees=0,
            scale=TRAIN_AUGMENTATION['zoom_range'],
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.ColorJitter(
            brightness=TRAIN_AUGMENTATION['brightness_range'],
            contrast=TRAIN_AUGMENTATION['contrast_range']
        ),
        T.RandomHorizontalFlip(p=TRAIN_AUGMENTATION['horizontal_flip_prob']),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """
    Get validation/test transforms without augmentation
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_data_loaders(train_df, val_df, batch_size, num_workers=4):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = BoneAgeDataset(
        train_df,
        transform=get_train_transforms(),
        augment=True
    )
    
    val_dataset = BoneAgeDataset(
        val_df,
        transform=get_val_transforms(),
        augment=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"✓ Created DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


class TestDataset(Dataset):
    """
    Dataset for test images (without labels)
    
    Args:
        image_dir: Directory containing test images
        image_ids: List of image IDs
        transform: Transform to apply
    """
    
    def __init__(self, image_dir, image_ids, sex_data=None, transform=None):
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.sex_data = sex_data  # Dict mapping image_id to sex
        self.transform = transform if transform else get_val_transforms()
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Try common image extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = self.image_dir / f"{image_id}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for ID: {image_id}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get sex if available
        sex = 0  # Default to female if not available
        if self.sex_data and image_id in self.sex_data:
            sex = 1 if self.sex_data[image_id] == 'M' else 0
        
        return {
            'image': image,
            'sex': torch.tensor(sex, dtype=torch.long),
            'image_id': image_id
        }


# Test the dataset
if __name__ == "__main__":
    from data_preprocessing import load_splits
    
    print("Testing BoneAgeDataset...")
    
    # Load splits
    train_df, val_df, _ = load_splits()
    
    # Create dataset
    dataset = BoneAgeDataset(train_df, transform=get_train_transforms(), augment=True)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Bone age: {sample['bone_age']:.2f} years")
    print(f"  Sex: {sample['sex'].item()} (0=F, 1=M)")
    print(f"  Age category: {sample['age_category'].item()}")
    print(f"  Image ID: {sample['image_id']}")
    
    # Test data loader
    from config import BATCH_SIZE
    train_loader, val_loader = create_data_loaders(train_df, val_df, BATCH_SIZE, num_workers=2)
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch structure:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Bone ages: {batch['bone_age'].shape}")
    print(f"  Sex: {batch['sex'].shape}")
    print(f"  Age categories: {batch['age_category'].shape}")
    
    print("\n✓ Dataset test complete!")