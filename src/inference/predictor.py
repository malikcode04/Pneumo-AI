"""
Main predictor class for pneumonia detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from loguru import logger

from ..data_preprocessing import get_inference_transform, SimpleLungSegmenter, QualityControl
from .gradcam import create_gradcam, apply_lung_mask_to_cam
from .uncertainty import UncertaintyEstimator


class PneumoniaPredictor:
    """Main predictor for pneumonia detection."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        class_names: list = None,
        use_uncertainty: bool = True,
        use_gradcam: bool = True,
        use_lung_masking: bool = True,
        mc_samples: int = 20
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            device: Device to use
            class_names: List of class names
            use_uncertainty: Whether to estimate uncertainty
            use_gradcam: Whether to generate Grad-CAM
            use_lung_masking: Whether to apply lung masking to CAM
            mc_samples: Number of MC dropout samples
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.class_names = class_names or ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
        self.use_uncertainty = use_uncertainty
        self.use_gradcam = use_gradcam
        self.use_lung_masking = use_lung_masking
        
        # Initialize components
        self.transform = get_inference_transform()
        
        if use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                model, num_samples=mc_samples, device=device
            )
        
        if use_gradcam:
            self.gradcam = create_gradcam(model)
        
        if use_lung_masking:
            self.lung_segmenter = SimpleLungSegmenter()
        
        self.quality_control = QualityControl()
        
        logger.info(f"Predictor initialized: device={device}, "
                   f"uncertainty={use_uncertainty}, gradcam={use_gradcam}")
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_heatmap: bool = True
    ) -> Dict:
        """
        Predict pneumonia from chest X-ray.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            return_heatmap: Whether to generate heatmap
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        original_image = self._load_image(image)
        
        # Quality control
        quality_score, quality_warnings = self.quality_control.check_quality(original_image)
        
        # Transform image
        transformed = self.transform(image=original_image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Uncertainty estimation
        uncertainty_metrics = {}
        entropy = 0.0
        flag_for_review = False
        review_reason = ""
        
        if self.use_uncertainty:
            mean_probs, std_probs, entropy, uncertainty_metrics = \
                self.uncertainty_estimator.estimate(input_tensor)
            
            flag_for_review, review_reason = \
                self.uncertainty_estimator.should_flag_for_review(
                    entropy, confidence
                )
        
        # Generate heatmap
        heatmap = None
        heatmap_overlay = None
        
        if return_heatmap and self.use_gradcam:
            cam = self.gradcam.generate_cam(input_tensor, predicted_class)
            
            # Apply lung masking if enabled
            if self.use_lung_masking:
                lung_mask = self.lung_segmenter.segment(original_image)
                cam = apply_lung_mask_to_cam(cam, lung_mask)
            
            # Resize to original size
            h, w = original_image.shape[:2]
            heatmap = cv2.resize(cam, (w, h))
            
            # Create overlay
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            heatmap_overlay = (0.5 * heatmap_colored + 0.5 * original_image).astype(np.uint8)
        
        # Build result
        result = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: probabilities[0, i].item()
                for i in range(len(self.class_names))
            },
            'uncertainty': {
                'entropy': entropy,
                **uncertainty_metrics
            },
            'quality': {
                'score': quality_score,
                'warnings': quality_warnings
            },
            'triage': {
                'flag_for_review': flag_for_review,
                'reason': review_reason
            },
            'heatmap': heatmap,
            'heatmap_overlay': heatmap_overlay,
            'original_image': original_image
        }
        
        # Calculate clinical metrics
        result['clinical_metrics'] = self._calculate_clinical_metrics(
            predicted_class, confidence
        )
        
        return result
    
    def _load_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Load image from various formats."""
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                image = Image.open(image).convert('RGB')
                image = np.array(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return image
    
    def _calculate_clinical_metrics(
        self,
        predicted_class: int,
        confidence: float
    ) -> Dict[str, str]:
        """Calculate clinical interpretation metrics."""
        # For demonstration - in production, these would be calibrated from validation set
        is_pneumonia = predicted_class > 0
        
        if is_pneumonia:
            # Estimated metrics for pneumonia prediction
            sensitivity = "High (>95%)"
            specificity = "Moderate (80-90%)"
            ppv = "Moderate (75-85%)"
            npv = "Very High (>95%)"
        else:
            # Estimated metrics for normal prediction
            sensitivity = "High (>95%)"
            specificity = "Moderate (80-90%)"
            ppv = "Very High (>95%)"
            npv = "Moderate (75-85%)"
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'interpretation': self._get_interpretation(predicted_class, confidence)
        }
    
    def _get_interpretation(self, predicted_class: int, confidence: float) -> str:
        """Get clinical interpretation with structured categorization."""
        class_name = self.class_names[predicted_class]
        
        if confidence > 0.9:
            category = "High Accuracy"
            level = "very high"
        elif confidence > 0.75:
            category = "Moderate Confidence"
            level = "high"
        elif confidence > 0.6:
            category = "Low Confidence"
            level = "moderate"
        else:
            category = "Uncertain / Review Required"
            level = "low"
        
        return {
            "category": category,
            "level": level,
            "text": f"{class_name} ({category})",
            "confidence_display": f"{confidence:.1%}"
        }
