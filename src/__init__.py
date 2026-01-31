"""Pneumonia Detection System - Source Package."""

__version__ = "1.0.0"
__author__ = "Medical AI Team"
__description__ = "Production-ready pneumonia detection from chest X-rays"

from . import data_preprocessing
from . import models
from . import training
from . import inference
from . import evaluation
from . import utils

__all__ = [
    'data_preprocessing',
    'models',
    'training',
    'inference',
    'evaluation',
    'utils'
]
