"""
Training callbacks for model training.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger


class EarlyStopping:
    """Early stopping to stop training when monitored metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        monitor: str = 'val_loss'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            monitor: Metric name to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
        logger.info(f"Early stopping initialized: monitor={monitor}, patience={patience}, "
                   f"mode={mode}, min_delta={min_delta}")
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current value of monitored metric
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check if improved
        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            logger.info(f"Metric improved: {self.monitor}={current_value:.4f}")
        else:
            self.counter += 1
            logger.info(f"No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.should_stop = True
                logger.warning(f"Early stopping triggered after {self.counter} epochs")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        filename_prefix: str = 'model'
    ):
        """
        Initialize model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'max' or 'min'
            save_best_only: Whether to save only the best model
            filename_prefix: Prefix for checkpoint filenames
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename_prefix = filename_prefix
        
        self.best_value = None
        self.best_epoch = None
        
        logger.info(f"Model checkpoint initialized: save_dir={save_dir}, "
                   f"monitor={monitor}, mode={mode}")
    
    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        current_value: float,
        metrics: dict
    ) -> Optional[str]:
        """
        Save checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            current_value: Current value of monitored metric
            metrics: Dictionary of all metrics
            
        Returns:
            Path to saved checkpoint or None
        """
        # Check if this is the best model
        is_best = False
        if self.best_value is None:
            is_best = True
        elif self.mode == 'max':
            is_best = current_value > self.best_value
        else:
            is_best = current_value < self.best_value
        
        # Save checkpoint
        if is_best or not self.save_best_only:
            if is_best:
                self.best_value = current_value
                self.best_epoch = epoch
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                self.monitor: current_value
            }
            
            # Determine filename
            if is_best:
                filename = f"{self.filename_prefix}_best_{self.monitor.replace('/', '_')}.pth"
            else:
                filename = f"{self.filename_prefix}_epoch_{epoch:03d}.pth"
            
            filepath = self.save_dir / filename
            
            # Save
            torch.save(checkpoint, filepath)
            logger.info(f"Saved checkpoint: {filepath} ({self.monitor}={current_value:.4f})")
            
            return str(filepath)
        
        return None


class LearningRateScheduler:
    """Learning rate scheduler wrapper."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'reduce_on_plateau',
        **kwargs
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler
            **kwargs: Additional arguments for scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        logger.info(f"Learning rate scheduler initialized: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """
        Step the scheduler.
        
        Args:
            metric: Metric value (required for ReduceLROnPlateau)
        """
        if self.scheduler_type == 'reduce_on_plateau':
            if metric is None:
                raise ValueError("Metric required for ReduceLROnPlateau")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")
    
    def get_last_lr(self) -> float:
        """Get last learning rate."""
        return self.optimizer.param_groups[0]['lr']
