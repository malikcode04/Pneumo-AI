"""
PyTorch Dataset for chest X-ray images.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from loguru import logger


class ChestXRayDataset(Dataset):
    """
    Dataset class for chest X-ray images with pneumonia labels.
    
    Supports multi-class classification:
    - 0: Normal
    - 1: Bacterial Pneumonia
    - 2: Viral Pneumonia
    """
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        class_mapping: Optional[Dict[int, str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            csv_file: Path to CSV file with columns [image_id, label]
            image_dir: Directory containing images
            transform: Optional transform to apply to images
            class_mapping: Optional mapping from class indices to names
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_mapping = class_mapping or {
            0: "Normal",
            1: "Bacterial Pneumonia",
            2: "Viral Pneumonia"
        }
        
        # Load metadata
        self.metadata = pd.read_csv(csv_file)
        
        # Validate columns
        required_cols = ['image_id', 'label']
        if not all(col in self.metadata.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Validate labels
        valid_labels = set(self.class_mapping.keys())
        invalid_labels = set(self.metadata['label'].unique()) - valid_labels
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}")
        
        logger.info(f"Loaded dataset with {len(self)} samples from {csv_file}")
        self._log_class_distribution()
    
    def _log_class_distribution(self):
        """Log class distribution for monitoring."""
        class_counts = self.metadata['label'].value_counts().sort_index()
        logger.info("Class distribution:")
        for label, count in class_counts.items():
            class_name = self.class_mapping[label]
            percentage = (count / len(self)) * 100
            logger.info(f"  {class_name} (class {label}): {count} ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image_tensor, label, metadata_dict)
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        label = int(row['label'])
        
        # Load image
        image_path = self.image_dir / image_id
        
        # Try multiple extensions if not found
        if not image_path.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_path = self.image_dir / f"{image_id}{ext}"
                if test_path.exists():
                    image_path = test_path
                    break
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        try:
            image = self._load_image(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Metadata
        metadata = {
            'image_id': image_id,
            'image_path': str(image_path),
            'class_name': self.class_mapping[label]
        }
        
        return image, label, metadata
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        # Load with OpenCV
        image = cv2.imread(str(image_path))
        
        if image is None:
            # Fallback to PIL
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalance.
        
        Returns:
            Tensor of class weights
        """
        class_counts = self.metadata['label'].value_counts().sort_index()
        total = len(self)
        
        # Inverse frequency weighting
        weights = []
        for label in sorted(self.class_mapping.keys()):
            count = class_counts.get(label, 1)
            weight = total / (len(self.class_mapping) * count)
            weights.append(weight)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights: {weights.tolist()}")
        
        return weights
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Get per-sample weights for weighted sampling.
        
        Returns:
            Array of sample weights
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = np.array([class_weights[label] for label in self.metadata['label']])
        return sample_weights


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    image_dir: str,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        image_dir: Directory containing images
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_weighted_sampling: Whether to use weighted sampling for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ChestXRayDataset(train_csv, image_dir, train_transform)
    val_dataset = ChestXRayDataset(val_csv, image_dir, val_transform)
    test_dataset = ChestXRayDataset(test_csv, image_dir, val_transform)
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        logger.info("Using weighted random sampling for training")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
