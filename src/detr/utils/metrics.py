"""
Evaluation metrics
"""

import torch
from typing import Tuple


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy

    Args:
        outputs: Model outputs (logits) of shape (batch, num_classes)
        targets: Ground truth labels of shape (batch,)

    Returns:
        Accuracy as float
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    """
    Calculate comprehensive metrics including per-class accuracy

    Args:
        outputs: Model outputs (logits) of shape (batch, num_classes)
        targets: Ground truth labels of shape (batch,)
        num_classes: Number of classes

    Returns:
        Dictionary containing various metrics
    """
    _, predicted = torch.max(outputs, dim=1)

    # Overall accuracy
    accuracy = calculate_accuracy(outputs, targets)

    # Per-class accuracy
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for pred, target in zip(predicted, targets):
        if pred == target:
            class_correct[target] += 1
        class_total[target] += 1

    class_accuracy = (class_correct / (class_total + 1e-6)).tolist()

    # Confusion matrix elements
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, target in zip(predicted, targets):
        confusion[target, pred] += 1

    return {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'confusion_matrix': confusion.tolist(),
        'num_samples': len(targets)
    }
