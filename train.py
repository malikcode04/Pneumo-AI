"""
Main training script for pneumonia detection model.
"""

import torch
import torch.optim as optim
from pathlib import Path
import argparse
from loguru import logger

from src.utils import Config, setup_logging, get_device
from src.data_preprocessing import (
    create_data_loaders,
    get_train_transforms,
    get_val_transforms
)
from src.models import create_densenet121, create_loss_function
from src.training import (
    PneumoniaTrainer,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train pneumonia detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("PNEUMONIA DETECTION SYSTEM - TRAINING")
    logger.info("=" * 80)
    
    # Get device
    if args.device != 'auto':
        config.set('inference.device', args.device)
    device = get_device(config)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    train_transform = get_train_transforms(
        image_size=config.get('data.image_size', 224),
        rotation_limit=config.get('data.augmentation.rotation_limit', 15),
        shift_limit=config.get('data.augmentation.shift_limit', 0.1),
        scale_limit=config.get('data.augmentation.scale_limit', 0.1),
        brightness_limit=config.get('data.augmentation.brightness_limit', 0.2),
        contrast_limit=config.get('data.augmentation.contrast_limit', 0.2),
        clahe_clip_limit=config.get('data.augmentation.clahe_clip_limit', 2.0),
        normalize_mean=tuple(config.get('data.normalize_mean', [0.485, 0.456, 0.406])),
        normalize_std=tuple(config.get('data.normalize_std', [0.229, 0.224, 0.225]))
    )
    
    val_transform = get_val_transforms(
        image_size=config.get('data.image_size', 224),
        normalize_mean=tuple(config.get('data.normalize_mean', [0.485, 0.456, 0.406])),
        normalize_std=tuple(config.get('data.normalize_std', [0.229, 0.224, 0.225]))
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_csv=config.get('data.train_csv'),
        val_csv=config.get('data.val_csv'),
        test_csv=config.get('data.test_csv'),
        image_dir=config.get('data.image_dir'),
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config.get('training.batch_size', 32),
        num_workers=4,
        use_weighted_sampling=config.get('training.loss.use_class_weights', True)
    )
    
    # Get class weights from training data
    class_weights = train_loader.dataset.get_class_weights()
    
    # Create model
    logger.info("Creating model...")
    model = create_densenet121(
        num_classes=config.get('model.num_classes', 3),
        pretrained=config.get('model.pretrained', True),
        dropout_rate=config.get('model.dropout_rate', 0.3),
        use_attention=config.get('model.use_attention', True)
    )
    
    # Create loss function
    loss_config = config.get('training.loss')
    criterion = create_loss_function(
        loss_type=loss_config.get('type', 'focal'),
        class_weights=class_weights,
        focal_alpha=loss_config.get('focal_alpha'),
        focal_gamma=loss_config.get('focal_gamma', 2.0)
    )
    
    # Create optimizer
    optimizer_type = config.get('training.optimizer', 'adam')
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('training.learning_rate', 0.0001),
            weight_decay=config.get('training.weight_decay', 0.0001)
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('training.learning_rate', 0.0001),
            weight_decay=config.get('training.weight_decay', 0.0001)
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('training.learning_rate', 0.0001),
            momentum=0.9,
            weight_decay=config.get('training.weight_decay', 0.0001)
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    
    # Create trainer
    trainer = PneumoniaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=config.get('training.use_amp', True),
        num_classes=config.get('model.num_classes', 3),
        class_names=list(config.get('data.classes', {}).values())
    )
    
    # Create callbacks
    early_stopping = None
    if config.get('training.early_stopping.enabled', True):
        early_stopping = EarlyStopping(
            patience=config.get('training.early_stopping.patience', 10),
            min_delta=config.get('training.early_stopping.min_delta', 0.001),
            mode=config.get('training.early_stopping.mode', 'max'),
            monitor=config.get('training.early_stopping.monitor', 'val_recall')
        )
    
    checkpoint_callback = ModelCheckpoint(
        save_dir=config.get('training.checkpoint.save_dir', 'checkpoints'),
        monitor=config.get('training.checkpoint.monitor', 'val_recall'),
        mode=config.get('training.checkpoint.mode', 'max'),
        save_best_only=config.get('training.checkpoint.save_best_only', True),
        filename_prefix='pneumonia_detector'
    )
    
    scheduler_config = config.get('training.scheduler')
    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_config.get('type', 'reduce_on_plateau'),
        mode='min',
        patience=scheduler_config.get('patience', 5),
        factor=scheduler_config.get('factor', 0.5),
        min_lr=scheduler_config.get('min_lr', 0.000001)
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        num_epochs=config.get('training.num_epochs', 50),
        early_stopping=early_stopping,
        checkpoint=checkpoint_callback,
        scheduler=scheduler
    )
    
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
