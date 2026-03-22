"""
autoDETR: Automated DETR object detection experimentation platform
Transformer encoder-decoder with ResNet50 backbone + automated experiment loop
"""

__version__ = "0.2.0"

from detr.model import DETR
from detr.data import DETRDataset, collate_fn
from detr.loss import DETRLoss, HungarianMatcher, compute_total_loss
from detr.config import get_model_config, get_training_config, get_evaluation_config

__all__ = [
    "DETR",
    "DETRDataset",
    "collate_fn",
    "DETRLoss",
    "HungarianMatcher",
    "compute_total_loss",
    "get_model_config",
    "get_training_config",
    "get_evaluation_config",
]
