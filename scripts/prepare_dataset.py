"""
Prepare dataset from Kaggle Chest X-Ray Pneumonia format.
Creates train/val/test CSV files from directory structure.
"""

import pandas as pd
from pathlib import Path
import argparse
import shutil
from sklearn.model_selection import train_test_split
from loguru import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare pneumonia dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to chest_xray directory')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for processed data')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split ratio')
    return parser.parse_args()


def create_csv_from_directory(data_dir: Path, split: str, label_mapping: dict) -> pd.DataFrame:
    """Create CSV from directory structure."""
    records = []
    
    split_dir = data_dir / split
    
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name.upper()
        
        # Determine label
        if 'NORMAL' in class_name:
            label = 0
        elif 'BACTERIA' in class_name:
            label = 1
        elif 'VIRUS' in class_name:
            label = 2
        else:
            logger.warning(f"Unknown class: {class_name}, skipping")
            continue
        
        # Get all images
        for img_path in class_dir.glob('*.jpeg'):
            records.append({
                'image_id': img_path.name,
                'label': label,
                'class_name': label_mapping[label],
                'original_path': str(img_path)
            })
        
        for img_path in class_dir.glob('*.jpg'):
            records.append({
                'image_id': img_path.name,
                'label': label,
                'class_name': label_mapping[label],
                'original_path': str(img_path)
            })
        
        for img_path in class_dir.glob('*.png'):
            records.append({
                'image_id': img_path.name,
                'label': label,
                'class_name': label_mapping[label],
                'original_path': str(img_path)
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Found {len(df)} images in {split} split")
    logger.info(f"Class distribution:\n{df['class_name'].value_counts()}")
    
    return df


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("DATASET PREPARATION")
    logger.info("=" * 80)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Label mapping
    label_mapping = {
        0: 'Normal',
        1: 'Bacterial Pneumonia',
        2: 'Viral Pneumonia'
    }
    
    # Process train split
    logger.info("Processing train split...")
    train_df = create_csv_from_directory(data_dir, 'train', label_mapping)
    
    # Process test split (use as test)
    logger.info("Processing test split...")
    test_df = create_csv_from_directory(data_dir, 'test', label_mapping)
    
    # Split train into train and val
    logger.info(f"Splitting train into train/val ({1-args.val_split:.0%}/{args.val_split:.0%})...")
    
    train_final, val_df = train_test_split(
        train_df,
        test_size=args.val_split,
        stratify=train_df['label'],
        random_state=42
    )
    
    # Copy images to centralized directory
    logger.info("Copying images to centralized directory...")
    
    for df in [train_final, val_df, test_df]:
        for _, row in df.iterrows():
            src = Path(row['original_path'])
            dst = images_dir / row['image_id']
            
            if not dst.exists():
                shutil.copy2(src, dst)
    
    logger.info(f"Copied {len(list(images_dir.glob('*')))} images")
    
    # Save CSV files
    train_csv = output_dir / 'train.csv'
    val_csv = output_dir / 'val.csv'
    test_csv = output_dir / 'test.csv'
    
    train_final[['image_id', 'label']].to_csv(train_csv, index=False)
    val_df[['image_id', 'label']].to_csv(val_csv, index=False)
    test_df[['image_id', 'label']].to_csv(test_csv, index=False)
    
    logger.info(f"Saved train CSV: {train_csv} ({len(train_final)} samples)")
    logger.info(f"Saved val CSV: {val_csv} ({len(val_df)} samples)")
    logger.info(f"Saved test CSV: {test_csv} ({len(test_df)} samples)")
    
    # Print final statistics
    logger.info("=" * 80)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total images: {len(train_final) + len(val_df) + len(test_df)}")
    logger.info(f"Train: {len(train_final)} ({len(train_final)/(len(train_final)+len(val_df)+len(test_df)):.1%})")
    logger.info(f"Val: {len(val_df)} ({len(val_df)/(len(train_final)+len(val_df)+len(test_df)):.1%})")
    logger.info(f"Test: {len(test_df)} ({len(test_df)/(len(train_final)+len(val_df)+len(test_df)):.1%})")
    logger.info("")
    logger.info("Train class distribution:")
    logger.info(train_final['class_name'].value_counts())
    logger.info("")
    logger.info("Val class distribution:")
    logger.info(val_df['class_name'].value_counts())
    logger.info("")
    logger.info("Test class distribution:")
    logger.info(test_df['class_name'].value_counts())
    logger.info("=" * 80)
    logger.info("Dataset preparation completed!")


if __name__ == '__main__':
    main()
