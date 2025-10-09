"""
DETR: DEtection TRansformer
Object detection using Transformer encoder-decoder with ResNet50 backbone
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from detr.model import DETR
from detr.data import DETRDataset, collate_fn
from detr.loss import DETRLoss, HungarianMatcher, compute_total_loss

__all__ = [
    "DETR",
    "DETRDataset",
    "collate_fn",
    "DETRLoss",
    "HungarianMatcher",
    "compute_total_loss",
]
