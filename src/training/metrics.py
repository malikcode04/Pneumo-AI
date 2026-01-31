"""
Metrics for model evaluation.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from typing import Dict, Tuple
from loguru import logger


class MetricsCalculator:
    """Calculate comprehensive metrics for pneumonia classification."""
    
    def __init__(self, num_classes: int = 3, class_names: list = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
    def calculate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            predictions: Predicted class labels (N,)
            targets: True class labels (N,)
            probabilities: Predicted probabilities (N, num_classes)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = support[i]
        
        # Macro averages
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = precision_w
        metrics['recall_weighted'] = recall_w
        metrics['f1_weighted'] = f1_w
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm
        
        # Clinical metrics (for binary: pneumonia vs normal)
        if self.num_classes == 3:
            # Treat classes 1 and 2 (bacterial and viral) as positive
            binary_targets = (targets > 0).astype(int)
            binary_predictions = (predictions > 0).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(binary_targets, binary_predictions).ravel()
            
            # Sensitivity (Recall for pneumonia)
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # PPV (Positive Predictive Value)
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # NPV (Negative Predictive Value)
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # AUC metrics (if probabilities provided)
        if probabilities is not None:
            try:
                # One-vs-rest AUC
                if self.num_classes > 2:
                    metrics['auc_macro'] = roc_auc_score(
                        targets, probabilities, multi_class='ovr', average='macro'
                    )
                    metrics['auc_weighted'] = roc_auc_score(
                        targets, probabilities, multi_class='ovr', average='weighted'
                    )
                else:
                    metrics['auc'] = roc_auc_score(targets, probabilities[:, 1])
                
                # Average Precision (PR-AUC)
                for i, class_name in enumerate(self.class_names):
                    binary_targets = (targets == i).astype(int)
                    metrics[f'pr_auc_{class_name}'] = average_precision_score(
                        binary_targets, probabilities[:, i]
                    )
                
                metrics['pr_auc_macro'] = np.mean([
                    metrics[f'pr_auc_{class_name}'] for class_name in self.class_names
                ])
            
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
        
        return metrics
    
    def calculate_recall_at_precision(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        target_precision: float = 0.95,
        positive_class: int = 1
    ) -> float:
        """
        Calculate recall at a target precision threshold.
        
        Args:
            targets: True labels
            probabilities: Predicted probabilities
            target_precision: Target precision threshold
            positive_class: Positive class index
            
        Returns:
            Recall at target precision
        """
        from sklearn.metrics import precision_recall_curve
        
        # Binary targets for positive class
        binary_targets = (targets == positive_class).astype(int)
        class_probs = probabilities[:, positive_class]
        
        # Get precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            binary_targets, class_probs
        )
        
        # Find recall at target precision
        valid_indices = precisions >= target_precision
        if np.any(valid_indices):
            recall_at_precision = np.max(recalls[valid_indices])
        else:
            recall_at_precision = 0.0
        
        return recall_at_precision


def calculate_batch_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate metrics for a single batch.
    
    Args:
        outputs: Model outputs (B, num_classes)
        targets: True labels (B,)
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    _, predictions = torch.max(outputs, dim=1)
    
    # Calculate accuracy
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    
    # Calculate per-class accuracy
    num_classes = outputs.size(1)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for i in range(num_classes):
        class_mask = targets == i
        if class_mask.sum() > 0:
            class_correct[i] = (predictions[class_mask] == targets[class_mask]).sum().item()
            class_total[i] = class_mask.sum().item()
    
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
    
    for i in range(num_classes):
        if class_total[i] > 0:
            metrics[f'accuracy_class_{i}'] = (class_correct[i] / class_total[i]).item()
        else:
            metrics[f'accuracy_class_{i}'] = 0.0
    
    return metrics
