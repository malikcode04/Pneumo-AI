"""
Generate synthetic data for local verification.
Creates dummy X-ray images and CSV files.
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import os

def create_synthetic_xray(path):
    """Create a synthetic X-ray looking image."""
    # Create empty image
    img = np.zeros((224, 224), dtype=np.uint8)
    
    # Add "lungs" (just ellipses)
    cv2.ellipse(img, (70, 112), (40, 80), 0, 0, 360, (100, 100, 100), -1)
    cv2.ellipse(img, (154, 112), (40, 80), 0, 0, 360, (100, 100, 100), -1)
    
    # Add some noise/texture
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Save
    cv2.imwrite(str(path), img)

def main():
    print("Generating synthetic dataset for verification...")
    
    base_dir = Path("pneumonia-detector/data")
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    classes = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}
    
    # Generate 10 images per class
    for label, class_name in classes.items():
        for i in range(10):
            img_id = f"synthetic_{class_name.lower().replace(' ', '_')}_{i}.jpg"
            img_path = images_dir / img_id
            
            create_synthetic_xray(img_path)
            
            records.append({
                "image_id": img_id,
                "label": label,
                "class_name": class_name
            })
            
    # Create DataFrames
    df = pd.DataFrame(records)
    
    # Save CSVs
    train_df = df.sample(frac=0.6, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)
    
    train_df.to_csv(base_dir / "train.csv", index=False)
    val_df.to_csv(base_dir / "val.csv", index=False)
    test_df.to_csv(base_dir / "test.csv", index=False)
    
    print(f"Created {len(df)} synthetic images in {images_dir}")
    print("Dataset setup complete.")

if __name__ == "__main__":
    main()
