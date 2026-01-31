"""
Model ensemble for improved performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from loguru import logger


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple models for improved predictions.
    
    Supports:
    - Weighted averaging
    - Uncertainty aggregation
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize model ensemble.
        
        Args:
            models: List of model instances
            weights: Optional weights for each model (must sum to 1.0)
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        # Set weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        logger.info(f"Model ensemble initialized with {len(models)} models, "
                   f"weights={weights}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Ensemble logits (B, num_classes)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (num_models, B, num_classes)
        
        # Weighted average
        weights = self.weights.to(predictions.device).view(-1, 1, 1)
        ensemble_probs = (predictions * weights).sum(dim=0)  # (B, num_classes)
        
        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-10)
        
        return ensemble_logits
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        mc_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        
        Combines model ensemble uncertainty with MC Dropout uncertainty.
        
        Args:
            x: Input tensor (B, 3, H, W)
            mc_samples: Number of MC dropout samples per model
            
        Returns:
            Tuple of (mean_predictions, std_predictions, entropy)
        """
        all_predictions = []
        
        # Get predictions from each model with MC Dropout
        for model in self.models:
            if hasattr(model, 'predict_with_uncertainty'):
                mean_probs, _, _ = model.predict_with_uncertainty(x, mc_samples)
                all_predictions.append(mean_probs)
            else:
                # Fallback to regular forward pass
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    all_predictions.append(probs)
        
        # Stack predictions
        predictions = torch.stack(all_predictions)  # (num_models, B, num_classes)
        
        # Weighted average
        weights = self.weights.to(predictions.device).view(-1, 1, 1)
        mean_probs = (predictions * weights).sum(dim=0)  # (B, num_classes)
        
        # Calculate standard deviation across models
        std_probs = torch.sqrt(
            ((predictions - mean_probs.unsqueeze(0)) ** 2 * weights).sum(dim=0)
        )
        
        # Calculate predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        return mean_probs, std_probs, entropy
    
    def eval(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            model.eval()
        return super().eval()
    
    def train(self, mode: bool = True):
        """Set all models to training mode."""
        for model in self.models:
            model.train(mode)
        return super().train(mode)


def create_ensemble(
    model_configs: List[dict],
    weights: Optional[List[float]] = None
) -> ModelEnsemble:
    """
    Factory function to create model ensemble.
    
    Args:
        model_configs: List of model configuration dicts
            Each dict should have:
            - 'type': 'densenet121' or 'efficientnet_b4'
            - 'checkpoint': Path to checkpoint file
            - 'num_classes': Number of classes
            - 'dropout_rate': Dropout rate
        weights: Optional ensemble weights
        
    Returns:
        ModelEnsemble instance
    """
    from .densenet import create_densenet121
    from .efficientnet import create_efficientnet_b4
    
    models = []
    
    for config in model_configs:
        model_type = config['type']
        num_classes = config.get('num_classes', 3)
        dropout_rate = config.get('dropout_rate', 0.3)
        
        # Create model
        if model_type == 'densenet121':
            model = create_densenet121(
                num_classes=num_classes,
                pretrained=False,
                dropout_rate=dropout_rate
            )
        elif model_type == 'efficientnet_b4':
            model = create_efficientnet_b4(
                num_classes=num_classes,
                pretrained=False,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint if provided
        if 'checkpoint' in config:
            checkpoint = torch.load(config['checkpoint'], map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint for {model_type}: {config['checkpoint']}")
        
        models.append(model)
    
    # Create ensemble
    ensemble = ModelEnsemble(models=models, weights=weights)
    
    return ensemble
