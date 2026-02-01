"""
Lung segmentation for quality control and heatmap masking.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Tuple, Optional
from loguru import logger
from pathlib import Path


class SimpleLungSegmenter:
    """
    Simple lung segmentation using traditional computer vision.
    For production, consider using pretrained U-Net from MONAI or torchvision.
    """
    
    def __init__(self):
        """Initialize lung segmenter."""
        self.kernel_size = 5
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment lung regions from chest X-ray.
        
        Args:
            image: Input image (H, W, 3) or (H, W) in RGB/grayscale
            
        Returns:
            Binary mask (H, W) with lung regions
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        
        # Remove small noise
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find largest connected components (lungs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        # Create mask with largest components (excluding background)
        if num_labels > 1:
            # Sort by area (excluding background at index 0)
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_indices = np.argsort(areas)[::-1]
            
            # Take top 2 components (left and right lung)
            mask = np.zeros_like(closed)
            for idx in sorted_indices[:2]:
                component_idx = idx + 1  # Offset for background
                mask[labels == component_idx] = 255
        else:
            mask = closed
        
        # Smooth mask
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask.astype(np.uint8)
    
    def get_lung_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of lung regions.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = mask.shape
            return 0, 0, w, h
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    def get_anatomical_features(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Extract anatomical features for integrity validation.
        Detects Heart Silhouette and Rib textures.
        """
        features = {}
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        mid_x = w // 2
        
        # Central cardiac region (Heart silhouette check)
        cardiac_region = gray[int(h*0.3):int(h*0.7), int(mid_x - w*0.1):int(mid_x + w*0.1)]
        features['cardiac_brightness'] = np.mean(cardiac_region) if cardiac_region.size > 0 else 0
        
        # Rib Frequency (Vertical gradient analysis)
        roi = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        features['vertical_gradient_variance'] = np.var(sobel_y)
        
        # Symmetry
        left_half = mask[:, :mid_x]
        right_half = mask[:, mid_x:]
        l_area = np.sum(left_half > 0)
        r_area = np.sum(right_half > 0)
        features['symmetry_ratio'] = min(l_area, r_area) / max(l_area, r_area + 1e-6)
        
        return features


class QualityControl:
    """Quality control checks for chest X-ray images."""
    
    def __init__(self):
        """Initialize quality control."""
        self.segmenter = SimpleLungSegmenter()
    
    def check_quality(self, image: np.ndarray) -> Tuple[float, dict]:
        """
        Perform quality checks on chest X-ray.
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            
        Returns:
            Tuple of (quality_score, warnings_dict)
        """
        warnings = {}
        scores = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Check 1: Brightness
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:
            warnings['underexposed'] = f"Image appears underexposed (mean brightness: {mean_brightness:.1f})"
            scores.append(0.5)
        elif mean_brightness > 200:
            warnings['overexposed'] = f"Image appears overexposed (mean brightness: {mean_brightness:.1f})"
            scores.append(0.7)
        else:
            scores.append(1.0)
        
        # Check 2: Contrast
        contrast = np.std(gray)
        if contrast < 30:
            warnings['low_contrast'] = f"Low contrast detected (std: {contrast:.1f})"
            scores.append(0.6)
        else:
            scores.append(1.0)
        
        # Check 3: Rotation (check if image is roughly aligned)
        rotation_score = self._check_rotation(gray)
        if rotation_score < 0.8:
            warnings['rotation'] = "Image may be rotated or misaligned"
        scores.append(rotation_score)
        
        # Check 4: Clipping (check if edges are clipped)
        clipping_score = self._check_clipping(gray)
        if clipping_score < 0.9:
            warnings['clipping'] = "Image edges may be clipped"
        scores.append(clipping_score)
        
        # Check 5: Lung visibility
        lung_mask = self.segmenter.segment(image)
        lung_area_ratio = np.sum(lung_mask > 0) / lung_mask.size
        if lung_area_ratio < 0.1:
            warnings['lung_visibility'] = "Lung regions may not be clearly visible"
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Overall quality score
        quality_score = np.mean(scores)
        
        return quality_score, warnings
    
    def _check_rotation(self, gray: np.ndarray) -> float:
        """Check if image is rotated."""
        # Use Hough line transform to detect dominant angles
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None:
            return 1.0
        
        # Calculate dominant angle
        angles = []
        for line in lines[:20]:  # Check top 20 lines
            rho, theta = line[0]
            angle = np.degrees(theta)
            angles.append(angle)
        
        if not angles:
            return 1.0
        
        # Check if angles are close to 0, 90, or 180 (properly aligned)
        median_angle = np.median(angles)
        deviation = min(
            abs(median_angle),
            abs(median_angle - 90),
            abs(median_angle - 180)
        )
        
        # Score based on deviation
        score = max(0.0, 1.0 - deviation / 45.0)
        return score
    
    def _check_clipping(self, gray: np.ndarray) -> float:
        """Check if image edges are clipped."""
        h, w = gray.shape
        border_width = 10
        
        # Check border pixels
        top_border = gray[:border_width, :]
        bottom_border = gray[-border_width:, :]
        left_border = gray[:, :border_width]
        right_border = gray[:, -border_width:]
        
        # Calculate how many border pixels are pure black or white
        borders = np.concatenate([
            top_border.flatten(),
            bottom_border.flatten(),
            left_border.flatten(),
            right_border.flatten()
        ])
        
        clipped_pixels = np.sum((borders < 5) | (borders > 250))
        total_border_pixels = len(borders)
        
        clipping_ratio = clipped_pixels / total_border_pixels
        
        # Score based on clipping ratio
        score = max(0.0, 1.0 - clipping_ratio)
        return score


def apply_lung_mask_to_heatmap(
    heatmap: np.ndarray,
    image: np.ndarray,
    segmenter: Optional[SimpleLungSegmenter] = None
) -> np.ndarray:
    """
    Apply lung mask to heatmap to suppress non-lung activations.
    
    Args:
        heatmap: Heatmap array (H, W)
        image: Original image (H, W, 3) or (H, W)
        segmenter: Optional lung segmenter instance
        
    Returns:
        Masked heatmap
    """
    if segmenter is None:
        segmenter = SimpleLungSegmenter()
    
    # Get lung mask
    lung_mask = segmenter.segment(image)
    
    # Resize mask to heatmap size if needed
    if lung_mask.shape != heatmap.shape:
        lung_mask = cv2.resize(lung_mask, (heatmap.shape[1], heatmap.shape[0]))
    
    # Normalize mask to [0, 1]
    lung_mask = lung_mask.astype(np.float32) / 255.0
    
    # Apply mask with smooth blending
    masked_heatmap = heatmap * lung_mask
    
    return masked_heatmap
