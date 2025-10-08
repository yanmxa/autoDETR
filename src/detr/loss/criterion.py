"""
DETR Loss Criterion.

This module implements the complete loss computation for DETR, including:
1. Hungarian matching between predictions and ground truth
2. Classification loss (cross-entropy)
3. Bounding box regression loss (L1)
4. Generalized IoU loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DETRLoss(nn.Module):
    """
    Complete loss function for DETR object detection.

    The loss computation involves two main steps:
    1. Use Hungarian matcher to find optimal assignment between predictions and targets
    2. Compute supervised losses (classification + box regression) for matched pairs

    Key design choices:
    - Uses focal loss / weighted cross-entropy to handle class imbalance
      (most queries predict "no object")
    - Combines L1 loss and GIoU loss for box regression
    - Normalizes losses by number of objects (not number of queries)
    """

    def __init__(self, num_classes: int, matcher, weight_dict: dict, eos_coef: float = 0.1):
        """
        Initialize DETR loss criterion.

        Args:
            num_classes: Number of object categories (excluding "no-object" class)
            matcher: HungarianMatcher module for bipartite matching
            weight_dict: Dictionary with loss component weights:
                - 'class_weighting': Weight for classification loss
                - 'bbox_weighting': Weight for L1 box loss
                - 'giou_weighting': Weight for GIoU loss
            eos_coef: Relative weight for "no-object" class in classification loss.
                      Lower value (e.g., 0.1) reduces importance of background predictions,
                      addressing class imbalance (most queries predict background).

        Example:
            weight_dict = {
                'class_weighting': 1.0,
                'bbox_weighting': 5.0,
                'giou_weighting': 2.0
            }
            criterion = DETRLoss(num_classes=91, matcher=matcher,
                               weight_dict=weight_dict, eos_coef=0.1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        # Create class weights for handling imbalance
        # Shape: [num_classes + 1], where last index is "no-object" class
        # All foreground classes have weight 1.0, background has weight eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def classification_loss(self, predictions, targets, indices):
        """
        Compute classification loss using cross-entropy.

        Strategy:
        1. Create target tensor where matched queries get their object class,
           unmatched queries get "no-object" class (index = num_classes)
        2. Apply weighted cross-entropy with lower weight for "no-object"

        Args:
            predictions: Dictionary with "pred_logits" [B, num_queries, num_classes+1]
            targets: List of target dictionaries with "labels"
            indices: Matching indices from Hungarian matcher

        Returns:
            Dictionary with 'loss_ce' key containing classification loss

        Example:
            If num_classes=5, num_queries=10, and 3 queries are matched:
            - 3 queries will have target classes 0-4 (object classes)
            - 7 queries will have target class 5 (no-object)
        """
        assert 'pred_logits' in predictions

        src_logits = predictions['pred_logits']  # [B, num_queries, num_classes+1]

        # Get indices of matched queries across entire batch
        # idx = (batch_indices, query_indices)
        idx = self._get_matched_query_indices(indices)

        # Concatenate ground truth labels for all matched queries
        # target_classes_o: [total_matched_queries]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Initialize all queries as "no-object" (class = num_classes)
        # Shape: [batch_size, num_queries]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )

        # Set matched queries to their ground truth object classes
        target_classes[idx] = target_classes_o

        # Compute weighted cross-entropy loss
        # transpose(1,2) makes shape [B, num_classes+1, num_queries] for F.cross_entropy
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )

        return {'loss_ce': loss_ce}

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes):
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
        x_c, y_c, w, h = boxes.unbind(-1)
        b = [
            (x_c - 0.5 * w),  # x1
            (y_c - 0.5 * h),  # y1
            (x_c + 0.5 * w),  # x2
            (y_c + 0.5 * h)   # y2
        ]
        return torch.stack(b, dim=-1)

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """Compute IoU between two sets of boxes in (x1, y1, x2, y2) format."""
        # Compute box areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Compute union
        union = area1[:, None] + area2 - inter

        # Compute IoU
        eps = 1e-7
        iou = inter / (union + eps)
        return iou, union

    @staticmethod
    def _generalized_box_iou(boxes1, boxes2):
        """
        Compute Generalized IoU between two sets of boxes.
        Both sets expected in (x1, y1, x2, y2) format.
        """
        # Compute regular IoU
        iou, union = DETRLoss._box_iou(boxes1, boxes2)

        # Compute smallest enclosing box
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        area = wh[:, :, 0] * wh[:, :, 1]

        eps = 1e-7
        # GIoU = IoU - (area of enclosing box - union) / area of enclosing box
        return iou - (area - union) / (area + eps)

    def box_loss(self, predictions, targets, indices, num_boxes):
        """
        Compute bounding box regression losses (L1 + GIoU).

        Only matched queries contribute to box losses. Unmatched queries
        (predicting "no-object") don't have box supervision.

        Args:
            predictions: Dictionary with "pred_boxes" [B, num_queries, 4]
            targets: List of target dicts with "boxes" [num_objects, 4]
            indices: Matching from Hungarian matcher
            num_boxes: Total number of objects in batch (for normalization)

        Returns:
            Dictionary with:
                - 'loss_bbox': L1 loss between matched boxes
                - 'loss_giou': GIoU loss between matched boxes

        Box format:
            All boxes are in normalized cxcywh format:
            (center_x, center_y, width, height) where values are in [0, 1]

        Why two box losses?
            - L1 loss: Provides strong gradients for precise localization
            - GIoU loss: Scale-invariant, works better for boxes of different sizes
        """
        assert 'pred_boxes' in predictions

        # Get matched predictions
        idx = self._get_matched_query_indices(indices)
        src_boxes = predictions['pred_boxes'][idx]  # [total_matched, 4]

        # Concatenate all ground truth boxes that were matched
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )  # [total_matched, 4]

        # 1. L1 Loss (mean absolute error for each coordinate)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')  # [total_matched, 4]

        # 2. GIoU Loss
        # Convert from cxcywh to xyxy format for IoU computation
        loss_giou = 1 - torch.diag(self._generalized_box_iou(
            self._box_cxcywh_to_xyxy(src_boxes),
            self._box_cxcywh_to_xyxy(target_boxes)
        ))  # [total_matched]

        # Normalize by total number of objects in the batch
        # This ensures loss magnitude is independent of batch size
        losses = {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes
        }

        return losses

    def _get_matched_query_indices(self, indices):
        """
        Convert per-image matching indices to batch indices.

        Args:
            indices: List of (query_idx, target_idx) tuples, one per batch element

        Returns:
            Tuple of (batch_indices, query_indices) for indexing predictions

        Example:
            Input: [(tensor([0,3]), tensor([0,1])),  # Image 0: queries 0,3 matched
                    (tensor([2]), tensor([0]))]       # Image 1: query 2 matched

            Output: (tensor([0,0,1]),  # batch indices
                     tensor([0,3,2]))  # query indices

            This can be used as: predictions[batch_idx, query_idx]
        """
        # Create batch indices: [0,0,...,1,1,...,2,2,...]
        # Each image contributes as many indices as it has matched queries
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])

        # Concatenate query indices from all images
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def forward(self, predictions, targets):
        """
        Compute total DETR loss.

        Pipeline:
        1. Hungarian matching: Find optimal assignment
        2. Classification loss: For all queries (matched + unmatched)
        3. Box losses: Only for matched queries

        Args:
            predictions: Dictionary containing:
                - "pred_logits": [batch_size, num_queries, num_classes+1]
                - "pred_boxes": [batch_size, num_queries, 4] in cxcywh format

            targets: List of dicts (len = batch_size), each with:
                - "labels": [num_objects] - class indices
                - "boxes": [num_objects, 4] - boxes in cxcywh format

        Returns:
            Dictionary with structure:
            {
                'labels': {'loss_ce': tensor(...)},
                'boxes': {
                    'loss_bbox': tensor(...),
                    'loss_giou': tensor(...)
                }
            }

        Final weighted loss should be computed as:
            total_loss = (
                weight_dict['class_weighting'] * loss_dict['labels']['loss_ce'] +
                weight_dict['bbox_weighting'] * loss_dict['boxes']['loss_bbox'] +
                weight_dict['giou_weighting'] * loss_dict['boxes']['loss_giou']
            )
        """
        # Step 1: Perform Hungarian matching
        indices = self.matcher(predictions, targets)

        # Ensure target tensors have correct dtypes
        device = next(iter(predictions.values())).device
        targets = [
            {
                'labels': t['labels'].to(torch.long),
                'boxes': t['boxes'].to(torch.float32)
            }
            for t in targets
        ]

        # Step 2: Count total objects for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=device
        ).clamp(min=1)  # Avoid division by zero

        # Step 3: Compute all loss components
        return {
            'labels': self.classification_loss(predictions, targets, indices),
            'boxes': self.box_loss(predictions, targets, indices, num_boxes)
        }


def compute_total_loss(loss_dict, weight_dict):
    """
    Compute weighted sum of all loss components.

    Args:
        loss_dict: Output from DETRLoss.forward()
        weight_dict: Dictionary with loss weights

    Returns:
        Scalar tensor representing total weighted loss

    Example:
        >>> loss_dict = criterion(predictions, targets)
        >>> total_loss = compute_total_loss(loss_dict, weight_dict)
        >>> total_loss.backward()
    """
    total_loss = (
        weight_dict['class_weighting'] * loss_dict['labels']['loss_ce'] +
        weight_dict['bbox_weighting'] * loss_dict['boxes']['loss_bbox'] +
        weight_dict['giou_weighting'] * loss_dict['boxes']['loss_giou']
    )
    return total_loss
