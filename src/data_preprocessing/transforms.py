"""
Image transforms and augmentations for chest X-rays.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Callable


def get_train_transforms(
    image_size: int = 224,
    rotation_limit: int = 15,
    shift_limit: float = 0.1,
    scale_limit: float = 0.1,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    clahe_clip_limit: float = 2.0,
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225)
) -> Callable:
    """
    Get training transforms with medical-appropriate augmentations.
    
    Args:
        image_size: Target image size
        rotation_limit: Maximum rotation angle in degrees
        shift_limit: Maximum shift as fraction of image size
        scale_limit: Maximum scale change as fraction
        brightness_limit: Maximum brightness change
        contrast_limit: Maximum contrast change
        clahe_clip_limit: CLAHE clip limit for contrast enhancement
        normalize_mean: Normalization mean
        normalize_std: Normalization std
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
        
        # Geometric augmentations (medical-safe)
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotation_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        
        # CLAHE for contrast enhancement (standard in medical imaging)
        A.CLAHE(
            clip_limit=clahe_clip_limit,
            tile_grid_size=(8, 8),
            p=0.5
        ),
        
        # Brightness and contrast
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5
        ),
        
        # Gamma correction
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Gaussian noise (simulates sensor noise)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Gaussian blur (simulates motion blur)
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # Grid distortion (subtle, for robustness)
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),
        
        # Normalize
        A.Normalize(mean=normalize_mean, std=normalize_std),
        
        # Convert to tensor
        ToTensorV2()
    ])


def get_val_transforms(
    image_size: int = 224,
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225)
) -> Callable:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
        normalize_mean: Normalization mean
        normalize_std: Normalization std
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2()
    ])


def get_tta_transforms(
    image_size: int = 224,
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225),
    num_augmentations: int = 5
) -> list:
    """
    Get test-time augmentation transforms.
    
    Args:
        image_size: Target image size
        normalize_mean: Normalization mean
        normalize_std: Normalization std
        num_augmentations: Number of TTA variants
        
    Returns:
        List of transform pipelines
    """
    base_transform = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
    ]
    
    normalize_and_tensor = [
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2()
    ]
    
    tta_variants = [
        # Original
        A.Compose(base_transform + normalize_and_tensor),
        
        # Slight rotation left
        A.Compose(base_transform + [
            A.Rotate(limit=(-5, -5), border_mode=cv2.BORDER_CONSTANT, p=1.0)
        ] + normalize_and_tensor),
        
        # Slight rotation right
        A.Compose(base_transform + [
            A.Rotate(limit=(5, 5), border_mode=cv2.BORDER_CONSTANT, p=1.0)
        ] + normalize_and_tensor),
        
        # Brightness adjustment
        A.Compose(base_transform + [
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
        ] + normalize_and_tensor),
        
        # CLAHE enhancement
        A.Compose(base_transform + [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
        ] + normalize_and_tensor),
    ]
    
    return tta_variants[:num_augmentations]


def get_inference_transform(
    image_size: int = 224,
    apply_clahe: bool = True,
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225)
) -> Callable:
    """
    Get inference transform with optional CLAHE enhancement.
    
    Args:
        image_size: Target image size
        apply_clahe: Whether to apply CLAHE enhancement
        normalize_mean: Normalization mean
        normalize_std: Normalization std
        
    Returns:
        Albumentations transform pipeline
    """
    transforms = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4)
    ]
    
    if apply_clahe:
        transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
    
    transforms.extend([
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)
