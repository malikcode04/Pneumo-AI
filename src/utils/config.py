"""
Utility functions for configuration management.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import os


class Config:
    """Configuration manager for the pneumonia detection system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _validate_config(self):
        """Validate configuration structure."""
        required_sections = ['data', 'model', 'training', 'clinical', 'inference']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.architecture')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.architecture')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration. If None, overwrites original.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config['data']
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config['model']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config['training']
    
    @property
    def clinical(self) -> Dict[str, Any]:
        """Get clinical configuration."""
        return self.config['clinical']
    
    @property
    def inference(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self.config['inference']
    
    @property
    def explainability(self) -> Dict[str, Any]:
        """Get explainability configuration."""
        return self.config.get('explainability', {})
    
    @property
    def app(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.config.get('app', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})


def setup_logging(config: Config):
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration object
    """
    log_config = config.logging_config
    log_dir = Path(log_config.get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sink=lambda msg: print(msg, end=''),
        format=log_config.get('format', '{time} | {level} | {message}'),
        level=log_config.get('level', 'INFO'),
        colorize=True
    )
    
    # Add file logger
    logger.add(
        sink=log_dir / "pneumonia_detector.log",
        format=log_config.get('format', '{time} | {level} | {message}'),
        level=log_config.get('level', 'INFO'),
        rotation=log_config.get('rotation', '500 MB'),
        retention=log_config.get('retention', '30 days'),
        compression="zip"
    )
    
    logger.info("Logging configured successfully")


def get_device(config: Config) -> str:
    """
    Get device for model inference/training.
    
    Args:
        config: Configuration object
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    import torch
    
    device_config = config.get('inference.device', 'auto')
    
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    return device
