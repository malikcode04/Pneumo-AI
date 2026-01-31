# 🫁 Pneumonia Detection - Quick Start Demo

This notebook demonstrates how to use the pneumonia detection system for inference.

## Setup

```python
import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.models import create_densenet121
from src.inference import PneumoniaPredictor
from src.utils import Config

%matplotlib inline
```

## Load Model

```python
# Load configuration
config = Config('../configs/config.yaml')

# Create model
model = create_densenet121(
    num_classes=3,
    pretrained=False,
    dropout_rate=0.3
)

# Load checkpoint
checkpoint_path = '../checkpoints/best_recall.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"✅ Model loaded from {checkpoint_path}")
print(f"📊 Device: {device}")
```

## Create Predictor

```python
predictor = PneumoniaPredictor(
    model=model,
    device=device,
    class_names=['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia'],
    use_uncertainty=True,
    use_gradcam=True,
    use_lung_masking=True,
    mc_samples=20
)

print("✅ Predictor initialized")
```

## Run Inference

```python
# Load sample image
image_path = '../data/test/sample.jpg'
image = Image.open(image_path)

# Display original
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title('Original Chest X-Ray')
plt.axis('off')
plt.show()

# Predict
result = predictor.predict(image_path, return_heatmap=True)
```

## Display Results

```python
print("=" * 80)
print("PREDICTION RESULTS")
print("=" * 80)
print(f"Predicted Class: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print()
print("Class Probabilities:")
for class_name, prob in result['probabilities'].items():
    print(f"  {class_name}: {prob:.2%}")
print()
print(f"Uncertainty (Entropy): {result['uncertainty']['entropy']:.4f}")
print(f"Quality Score: {result['quality']['score']:.2%}")
print()
if result['triage']['flag_for_review']:
    print(f"⚠️  TRIAGE: {result['triage']['reason']}")
else:
    print(f"✅ {result['triage']['reason']}")
print("=" * 80)
```

## Visualize Heatmap

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original
axes[0].imshow(result['original_image'])
axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Heatmap
axes[1].imshow(result['heatmap'], cmap='jet')
axes[1].set_title('Grad-CAM++ Heatmap', fontsize=14, fontweight='bold')
axes[1].axis('off')

# Overlay
axes[2].imshow(result['heatmap_overlay'])
axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Batch Inference

```python
import glob

# Get all test images
test_images = glob.glob('../data/test/*.jpg')[:5]

results = []
for img_path in test_images:
    result = predictor.predict(img_path, return_heatmap=False)
    results.append({
        'image': img_path,
        'prediction': result['predicted_label'],
        'confidence': result['confidence']
    })

# Display results
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

## Export Results

```python
import json

# Save to JSON
with open('prediction_results.json', 'w') as f:
    json.dump({
        'image': image_path,
        'prediction': result['predicted_label'],
        'confidence': float(result['confidence']),
        'probabilities': {k: float(v) for k, v in result['probabilities'].items()},
        'uncertainty': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in result['uncertainty'].items()},
        'clinical_metrics': result['clinical_metrics']
    }, f, indent=2)

print("✅ Results saved to prediction_results.json")
```

## Conclusion

This notebook demonstrated:
- ✅ Loading a trained pneumonia detection model
- ✅ Running inference on chest X-rays
- ✅ Visualizing Grad-CAM++ heatmaps
- ✅ Interpreting uncertainty and triage recommendations
- ✅ Batch processing multiple images

For production deployment, use the Streamlit web app or FastAPI server.
