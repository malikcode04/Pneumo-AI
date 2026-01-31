"""
EfficientNet-B4 model for pneumonia classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple
from loguru import logger


class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 based classifier for pneumonia detection.
    
    Features:
    - Pretrained ImageNet weights via timm
    - Custom classification head
    - MC Dropout for uncertainty estimation
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize EfficientNet-B4 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pretrained EfficientNet-B4
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        if pretrained:
            logger.info("Loaded EfficientNet-B4 with ImageNet pretrained weights")
        else:
            logger.info("Initialized EfficientNet-B4 without pretrained weights")
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
        
        logger.info(f"EfficientNet-B4 classifier initialized: {num_classes} classes, "
                   f"dropout={dropout_rate}")
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Logits tensor (B, num_classes) or tuple of (logits, features)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Features tensor (B, num_features)
        """
        return self.backbone(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            x: Input tensor (B, 3, H, W)
            num_samples: Number of MC dropout samples
            
        Returns:
            Tuple of (mean_predictions, std_predictions, entropy)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        self.eval()  # Disable dropout
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (num_samples, B, num_classes)
        
        # Calculate statistics
        mean_probs = predictions.mean(dim=0)  # (B, num_classes)
        std_probs = predictions.std(dim=0)  # (B, num_classes)
        
        # Calculate predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)  # (B,)
        
        return mean_probs, std_probs, entropy
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
    def get_target_layer(self) -> nn.Module:
        """
        Get target layer for Grad-CAM.
        
        Returns:
            Target layer module
        """
        # For EfficientNet-B4, use the last convolutional block
        return self.backbone.blocks[-1]


def create_efficientnet_b4(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> EfficientNetB4Classifier:
    """
    Factory function to create EfficientNet-B4 classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout probability
        
    Returns:
        EfficientNetB4Classifier instance
    """
    model = EfficientNetB4Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    return model
