"""
Main trainer class for pneumonia detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from loguru import logger

from .metrics import MetricsCalculator, calculate_batch_metrics
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


class PneumoniaTrainer:
    """Trainer for pneumonia detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        use_amp: bool = True,
        num_classes: int = 3,
        class_names: list = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to use
            use_amp: Whether to use automatic mixed precision
            num_classes: Number of classes
            class_names: List of class names
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=num_classes,
            class_names=class_names
        )
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.train_history = []
        self.val_history = []
        
        logger.info(f"Trainer initialized: device={device}, use_amp={use_amp}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, (images, targets, metadata) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with AMP
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        metrics = self.metrics_calculator.calculate(
            all_predictions, all_targets, all_probabilities
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for images, targets, metadata in progress_bar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Accumulate metrics
                total_loss += loss.item()
                
                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        metrics = self.metrics_calculator.calculate(
            all_predictions, all_targets, all_probabilities
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def fit(
        self,
        num_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        scheduler: Optional[LearningRateScheduler] = None
    ):
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping: Optional early stopping callback
            checkpoint: Optional checkpoint callback
            scheduler: Optional learning rate scheduler
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Log metrics
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"Recall: {train_metrics['recall_macro']:.4f}")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"Recall: {val_metrics['recall_macro']:.4f}")
            
            # Learning rate scheduler
            if scheduler:
                if scheduler.scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Save checkpoint
            if checkpoint:
                monitor_value = val_metrics.get(checkpoint.monitor, val_metrics['loss'])
                checkpoint(
                    epoch=self.current_epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    current_value=monitor_value,
                    metrics=val_metrics
                )
            
            # Early stopping
            if early_stopping:
                monitor_value = val_metrics.get(early_stopping.monitor, val_metrics['loss'])
                if early_stopping(monitor_value):
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed")
