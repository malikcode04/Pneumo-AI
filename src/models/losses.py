"""
Loss functions for pneumonia classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from loguru import logger


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor (num_classes,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        logger.info(f"Focal Loss initialized: gamma={gamma}, alpha={alpha}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits tensor (B, num_classes)
            targets: Target labels (B,)
            
        Returns:
            Loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal term
        focal_term = (1 - target_probs) ** self.gamma
        
        # Apply focal term
        loss = focal_term * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance."""
    
    def __init__(self, weight: Optional[torch.Tensor] = None):
        """
        Initialize Weighted Cross Entropy Loss.
        
        Args:
            weight: Class weights tensor (num_classes,)
        """
        super().__init__()
        self.weight = weight
        logger.info(f"Weighted CE Loss initialized: weight={weight}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross entropy loss.
        
        Args:
            inputs: Logits tensor (B, num_classes)
            targets: Target labels (B,)
            
        Returns:
            Loss value
        """
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=self.weight)


class CombinedLoss(nn.Module):
    """Combined loss function (Focal + Cross Entropy)."""
    
    def __init__(
        self,
        focal_weight: float = 0.7,
        ce_weight: float = 0.3,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize Combined Loss.
        
        Args:
            focal_weight: Weight for focal loss
            ce_weight: Weight for cross entropy loss
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
            class_weights: Class weights for cross entropy
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_loss = WeightedCrossEntropyLoss(weight=class_weights)
        
        logger.info(f"Combined Loss initialized: focal_weight={focal_weight}, "
                   f"ce_weight={ce_weight}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            inputs: Logits tensor (B, num_classes)
            targets: Target labels (B,)
            
        Returns:
            Loss value
        """
        focal = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        
        return self.focal_weight * focal + self.ce_weight * ce


def create_loss_function(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: Optional[list] = None,
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss ('focal', 'cross_entropy', 'weighted_ce', 'combined')
        class_weights: Class weights for weighted losses
        focal_alpha: Alpha values for focal loss
        focal_gamma: Gamma value for focal loss
        
    Returns:
        Loss function module
    """
    if loss_type == 'focal':
        alpha = torch.tensor(focal_alpha) if focal_alpha else None
        return FocalLoss(alpha=alpha, gamma=focal_gamma)
    
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'combined':
        alpha = torch.tensor(focal_alpha) if focal_alpha else None
        return CombinedLoss(
            focal_weight=0.7,
            ce_weight=0.3,
            alpha=alpha,
            gamma=focal_gamma,
            class_weights=class_weights
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
