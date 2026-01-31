"""
Main prediction script for pneumonia detection.
"""

import torch
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger

from src.utils import Config, setup_logging, get_device
from src.models import create_densenet121
from src.inference import PneumoniaPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to chest X-ray image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_recall.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (JSON)')
    parser.add_argument('--save-heatmap', action='store_true',
                       help='Save heatmap visualization')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    return parser.parse_args()


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("PNEUMONIA DETECTION SYSTEM - INFERENCE")
    logger.info("=" * 80)
    
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
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"Model loaded from {checkpoint_path}")
    
    # Create predictor
    predictor = PneumoniaPredictor(
        model=model,
        device=device,
        class_names=list(config.get('data.classes', {}).values()),
        use_uncertainty=True,
        use_gradcam=True,
        use_lung_masking=config.get('explainability.gradcam.use_lung_masking', True),
        mc_samples=config.get('explainability.uncertainty.num_samples', 20)
    )
    
    # Load and predict
    logger.info(f"Processing image: {args.image}")
    
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return
    
    result = predictor.predict(args.image, return_heatmap=True)
    
    # Display results
    logger.info("=" * 80)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Predicted Class: {result['predicted_label']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info("")
    logger.info("Class Probabilities:")
    for class_name, prob in result['probabilities'].items():
        logger.info(f"  {class_name}: {prob:.2%}")
    logger.info("")
    logger.info("Uncertainty Metrics:")
    logger.info(f"  Entropy: {result['uncertainty']['entropy']:.4f}")
    logger.info(f"  Confidence: {result['uncertainty'].get('confidence', 0):.2%}")
    logger.info("")
    logger.info("Quality Assessment:")
    logger.info(f"  Quality Score: {result['quality']['score']:.2%}")
    if result['quality']['warnings']:
        logger.warning("  Warnings:")
        for warning_type, warning_msg in result['quality']['warnings'].items():
            logger.warning(f"    - {warning_msg}")
    logger.info("")
    logger.info("Triage Recommendation:")
    if result['triage']['flag_for_review']:
        logger.warning(f"  ⚠️  FLAG FOR REVIEW: {result['triage']['reason']}")
    else:
        logger.info(f"  ✓ {result['triage']['reason']}")
    logger.info("")
    logger.info("Clinical Interpretation:")
    logger.info(f"  {result['clinical_metrics']['interpretation']}")
    logger.info("=" * 80)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare JSON-serializable results
        json_result = {
            'image_path': str(args.image),
            'predicted_class': int(result['predicted_class']),
            'predicted_label': result['predicted_label'],
            'confidence': float(result['confidence']),
            'probabilities': {k: float(v) for k, v in result['probabilities'].items()},
            'uncertainty': {
                'entropy': float(result['uncertainty']['entropy']),
                'confidence': float(result['uncertainty'].get('confidence', 0)),
                'variation_ratio': float(result['uncertainty'].get('variation_ratio', 0))
            },
            'quality': {
                'score': float(result['quality']['score']),
                'warnings': result['quality']['warnings']
            },
            'triage': result['triage'],
            'clinical_metrics': result['clinical_metrics']
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    # Save heatmap
    if args.save_heatmap and result['heatmap_overlay'] is not None:
        heatmap_path = Path(args.image).parent / f"{Path(args.image).stem}_heatmap.png"
        Image.fromarray(result['heatmap_overlay']).save(heatmap_path)
        logger.info(f"Heatmap saved to {heatmap_path}")


if __name__ == '__main__':
    main()
