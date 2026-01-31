"""
Grad-CAM++ implementation for explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from loguru import logger


class GradCAMPlusPlus:
    """
    Grad-CAM++ for generating heatmaps.
    
    Reference: Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based 
    Visual Explanations for Deep Convolutional Networks"
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM++.
        
        Args:
            model: Model instance
            target_layer: Target layer for CAM generation
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
        
        logger.info(f"Grad-CAM++ initialized with target layer: {target_layer}")
    
    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_image: Input tensor (1, 3, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate alpha (Grad-CAM++ weights)
        alpha_numer = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                      (activations * gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0.0,
            alpha_denom,
            torch.ones_like(alpha_denom)
        )
        alpha = alpha_numer / alpha_denom
        
        # Calculate weights
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_heatmap_overlay(
        self,
        input_image: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Generate heatmap overlay on original image.
        
        Args:
            input_image: Preprocessed input tensor (1, 3, H, W)
            original_image: Original image (H, W, 3) in RGB
            target_class: Target class for CAM
            colormap: OpenCV colormap
            alpha: Overlay transparency
            
        Returns:
            Overlay image (H, W, 3)
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Resize CAM to original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlay


def create_gradcam(
    model: nn.Module,
    target_layer: Optional[nn.Module] = None
) -> GradCAMPlusPlus:
    """
    Factory function to create Grad-CAM++ instance.
    
    Args:
        model: Model instance
        target_layer: Target layer (if None, uses model's get_target_layer method)
        
    Returns:
        GradCAMPlusPlus instance
    """
    if target_layer is None:
        if hasattr(model, 'get_target_layer'):
            target_layer = model.get_target_layer()
        else:
            raise ValueError("Target layer must be provided or model must have get_target_layer method")
    
    return GradCAMPlusPlus(model, target_layer)


def apply_lung_mask_to_cam(
    cam: np.ndarray,
    lung_mask: np.ndarray,
    smooth: bool = True
) -> np.ndarray:
    """
    Apply lung mask to CAM to suppress non-lung activations.
    
    Args:
        cam: CAM heatmap (H, W)
        lung_mask: Binary lung mask (H, W)
        smooth: Whether to smooth the mask boundary
        
    Returns:
        Masked CAM
    """
    # Resize mask if needed
    if lung_mask.shape != cam.shape:
        lung_mask = cv2.resize(lung_mask, (cam.shape[1], cam.shape[0]))
    
    # Normalize mask to [0, 1]
    mask = lung_mask.astype(np.float32) / 255.0
    
    # Smooth mask boundary
    if smooth:
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
    # Apply mask
    masked_cam = cam * mask
    
    # Renormalize
    if masked_cam.max() > 0:
        masked_cam = masked_cam / masked_cam.max()
    
    return masked_cam
