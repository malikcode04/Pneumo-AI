"""
DenseNet-121 model for pneumonia classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
from loguru import logger


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 based classifier for pneumonia detection.
    
    Features:
    - Pretrained ImageNet weights
    - Custom classification head
    - MC Dropout for uncertainty estimation
    - Attention mechanism
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize DenseNet-121 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Load pretrained DenseNet-121
        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.backbone = models.densenet121(weights=weights)
            logger.info("Loaded DenseNet-121 with ImageNet pretrained weights")
        else:
            self.backbone = models.densenet121(weights=None)
            logger.info("Initialized DenseNet-121 without pretrained weights")
        
        # Get number of features from backbone
        num_features = self.backbone.classifier.in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(num_features, num_features // 4),
                nn.ReLU(inplace=True),
                nn.Linear(num_features // 4, num_features),
                nn.Sigmoid()
            )
        
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
        
        logger.info(f"DenseNet-121 classifier initialized: {num_classes} classes, "
                   f"dropout={dropout_rate}, attention={use_attention}")
    
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
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
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
        features = self.backbone(x)
        
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        return features
    
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
        # For DenseNet-121, use the last dense block
        return self.backbone.features.denseblock4


def create_densenet121(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    use_attention: bool = True
) -> DenseNet121Classifier:
    """
    Factory function to create DenseNet-121 classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout probability
        use_attention: Whether to use attention mechanism
        
    Returns:
        DenseNet121Classifier instance
    """
    model = DenseNet121Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_attention=use_attention
    )
    return model
