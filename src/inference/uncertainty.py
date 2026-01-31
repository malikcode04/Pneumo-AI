"""
Uncertainty estimation using MC Dropout.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from loguru import logger


class UncertaintyEstimator:
    """Estimate prediction uncertainty using MC Dropout."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        num_samples: int = 20,
        device: str = 'cuda'
    ):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: Model with dropout layers
            num_samples: Number of MC dropout samples
            device: Device to use
        """
        self.model = model
        self.num_samples = num_samples
        self.device = device
        
        logger.info(f"Uncertainty estimator initialized: num_samples={num_samples}")
    
    def estimate(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, float]]:
        """
        Estimate uncertainty for input.
        
        Args:
            input_tensor: Input tensor (1, 3, H, W) or (B, 3, H, W)
            
        Returns:
            Tuple of (mean_probs, std_probs, entropy, uncertainty_metrics)
        """
        input_tensor = input_tensor.to(self.device)
        
        # Enable dropout only (keep BatchNorm in eval mode)
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        # Collect predictions
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1)
                predictions.append(probs)
        
        # Disable dropout
        self.model.eval()
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (num_samples, B, num_classes)
        
        # Calculate statistics
        mean_probs = predictions.mean(dim=0)  # (B, num_classes)
        std_probs = predictions.std(dim=0)  # (B, num_classes)
        
        # Calculate predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)  # (B,)
        
        # Calculate additional uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(
            predictions, mean_probs, std_probs, entropy
        )
        
        return mean_probs, std_probs, entropy.item(), uncertainty_metrics
    
    def _calculate_uncertainty_metrics(
        self,
        predictions: torch.Tensor,
        mean_probs: torch.Tensor,
        std_probs: torch.Tensor,
        entropy: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics."""
        metrics = {}
        
        # Predictive entropy (already calculated)
        metrics['predictive_entropy'] = entropy.mean().item()
        
        # Mutual information (epistemic uncertainty)
        # MI = E[H(y|x,w)] - H(E[y|x,w])
        expected_entropy = -torch.sum(
            predictions * torch.log(predictions + 1e-10), dim=2
        ).mean(dim=0)  # (B,)
        
        mutual_info = expected_entropy - entropy
        metrics['mutual_information'] = mutual_info.mean().item()
        
        # Variation ratio (1 - max_prob)
        max_prob = mean_probs.max(dim=1)[0]
        variation_ratio = 1 - max_prob
        metrics['variation_ratio'] = variation_ratio.mean().item()
        
        # Confidence (max probability)
        metrics['confidence'] = max_prob.mean().item()
        
        # Standard deviation of predicted class
        predicted_class = mean_probs.argmax(dim=1)
        predicted_class_std = std_probs.gather(
            1, predicted_class.unsqueeze(1)
        ).squeeze(1)
        metrics['predicted_class_std'] = predicted_class_std.mean().item()
        
        return metrics
    
    def should_flag_for_review(
        self,
        entropy: float,
        confidence: float,
        entropy_threshold: float = 0.3,
        confidence_threshold: float = 0.6
    ) -> Tuple[bool, str]:
        """
        Determine if prediction should be flagged for manual review.
        
        Args:
            entropy: Predictive entropy
            confidence: Prediction confidence
            entropy_threshold: Threshold for high entropy
            confidence_threshold: Threshold for low confidence
            
        Returns:
            Tuple of (should_flag, reason)
        """
        if entropy > entropy_threshold:
            return True, f"High uncertainty (entropy={entropy:.3f})"
        
        if confidence < confidence_threshold:
            return True, f"Low confidence (confidence={confidence:.3f})"
        
        return False, "Confident prediction"


def calculate_calibration_metrics(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate calibration metrics (ECE, MCE).
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
        predictions: Predicted classes (N,)
        targets: True classes (N,)
        num_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    # Get confidence (max probability)
    confidences = probabilities.max(axis=1)
    accuracies = (predictions == targets).astype(float)
    
    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Calculate ECE and MCE
    ece = 0.0
    mce = 0.0
    
    for i in range(num_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() > 0:
            bin_confidence = confidences[bin_mask].mean()
            bin_accuracy = accuracies[bin_mask].mean()
            bin_size = bin_mask.sum() / len(confidences)
            
            calibration_error = abs(bin_confidence - bin_accuracy)
            ece += bin_size * calibration_error
            mce = max(mce, calibration_error)
    
    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce
    }
