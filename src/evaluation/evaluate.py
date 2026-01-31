"""
Comprehensive model evaluation script.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from sklearn.metrics import confusion_matrix, classification_report
from loguru import logger

from src.utils import Config, setup_logging, get_device
from src.data_preprocessing import create_data_loaders, get_val_transforms
from src.models import create_densenet121
from src.training import MetricsCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate pneumonia detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    return parser.parse_args()


def evaluate_model(model, data_loader, device, class_names):
    """Evaluate model on dataset."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("Running evaluation...")
    
    with torch.no_grad():
        for images, targets, metadata in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    metrics_calculator = MetricsCalculator(
        num_classes=len(class_names),
        class_names=class_names
    )
    
    metrics = metrics_calculator.calculate(
        all_predictions,
        all_targets,
        all_probabilities
    )
    
    return metrics, all_predictions, all_targets, all_probabilities


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_class_metrics(metrics, class_names, output_path):
    """Plot per-class metrics."""
    metrics_to_plot = ['precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric_name in enumerate(metrics_to_plot):
        values = [metrics[f'{metric_name}_{class_name}'] for class_name in class_names]
        
        axes[idx].bar(class_names, values, color=['#3498db', '#e74c3c', '#f39c12'])
        axes[idx].set_title(f'{metric_name.capitalize()} by Class', fontweight='bold')
        axes[idx].set_ylabel(metric_name.capitalize())
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Class metrics plot saved to {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("PNEUMONIA DETECTION SYSTEM - EVALUATION")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    if args.device != 'auto':
        config.set('inference.device', args.device)
    device = get_device(config)
    
    # Load model
    logger.info("Loading model...")
    model = create_densenet121(
        num_classes=config.get('model.num_classes', 3),
        pretrained=False,
        dropout_rate=config.get('model.dropout_rate', 0.3)
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    logger.info(f"Model loaded from {args.checkpoint}")
    
    # Create data loader
    logger.info("Loading test data...")
    val_transform = get_val_transforms(
        image_size=config.get('data.image_size', 224)
    )
    
    _, _, test_loader = create_data_loaders(
        train_csv=config.get('data.train_csv'),
        val_csv=config.get('data.val_csv'),
        test_csv=config.get('data.test_csv'),
        image_dir=config.get('data.image_dir'),
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=config.get('training.batch_size', 32),
        num_workers=4,
        use_weighted_sampling=False
    )
    
    # Evaluate
    class_names = list(config.get('data.classes', {}).values())
    metrics, predictions, targets, probabilities = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    
    if 'sensitivity' in metrics:
        logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"PPV: {metrics['ppv']:.4f}")
        logger.info(f"NPV: {metrics['npv']:.4f}")
    
    logger.info("")
    logger.info("Per-Class Metrics:")
    for class_name in class_names:
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
        logger.info(f"    Recall: {metrics[f'recall_{class_name}']:.4f}")
        logger.info(f"    F1: {metrics[f'f1_{class_name}']:.4f}")
    
    logger.info("=" * 80)
    
    # Save metrics to JSON
    metrics_json = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items() if k != 'confusion_matrix'}
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Metrics saved to {output_dir / 'metrics.json'}")
    
    # Plot confusion matrix
    cm = metrics['confusion_matrix']
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    
    # Plot class metrics
    plot_class_metrics(metrics, class_names, output_dir / 'class_metrics.png')
    
    # Save classification report
    report = classification_report(targets, predictions, target_names=class_names)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {output_dir / 'classification_report.txt'}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
