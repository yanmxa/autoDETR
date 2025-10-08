"""
Hungarian Matcher for DETR object detection.

This module implements the bipartite matching between predictions and ground truth
using the Hungarian algorithm (Kuhn-Munkres algorithm).
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Computes an optimal assignment between predictions and ground truth using
    the Hungarian algorithm.

    The matching is performed based on a weighted combination of:
    - Classification cost (negative probability of the correct class)
    - L1 distance between predicted and ground truth boxes
    - Generalized IoU cost between boxes

    This is a key component of DETR's training process, as it determines which
    predicted query should be matched to which ground truth object.
    """

    def __init__(self, weight_dict: dict):
        """
        Initialize the Hungarian Matcher.

        Args:
            weight_dict: Dictionary containing matching cost weights:
                - 'class_weighting': Weight for classification cost
                - 'bbox_weighting': Weight for L1 box coordinate cost
                - 'giou_weighting': Weight for GIoU cost

        Raises:
            AssertionError: If required weights are missing or all weights are zero
        """
        super().__init__()

        # Validate weight dict contains all required keys
        assert weight_dict.get('class_weighting') is not None and \
               weight_dict.get('bbox_weighting') is not None and \
               weight_dict.get('giou_weighting') is not None, \
               "Weight dict must contain weighting for all three losses: giou, class and bbox."

        # Ensure at least one weight is non-zero
        assert weight_dict.get('class_weighting') != 0 or \
               weight_dict.get('bbox_weighting') != 0 or \
               weight_dict.get('giou_weighting') != 0, \
               "At least one loss weight must be non-zero."

        self.class_weighting = weight_dict.get('class_weighting')
        self.bbox_weighting = weight_dict.get('bbox_weighting')
        self.giou_weighting = weight_dict.get('giou_weighting')

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
        iou, union = HungarianMatcher._box_iou(boxes1, boxes2)

        # Compute smallest enclosing box
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        area = wh[:, :, 0] * wh[:, :, 1]

        eps = 1e-7
        # GIoU = IoU - (area of enclosing box - union) / area of enclosing box
        return iou - (area - union) / (area + eps)

    @torch.no_grad()
    def forward(self, predictions, targets):
        """
        Perform bipartite matching between predictions and ground truth.

        For each image in the batch, this computes:
        1. Cost matrix C[i,j] = cost of matching prediction i to target j
        2. Optimal assignment using Hungarian algorithm

        Args:
            predictions: Dictionary containing:
                - "pred_logits": [batch_size, num_queries, num_classes] - class logits
                - "pred_boxes": [batch_size, num_queries, 4] - predicted boxes in cxcywh format

            targets: List of dictionaries (length = batch_size), each containing:
                - "labels": [num_objects] - class indices for this image
                - "boxes": [num_objects, 4] - ground truth boxes in cxcywh format

        Returns:
            List of tuples (index_pred, index_target) for each batch element, where:
                - index_pred: Indices of selected predictions (matched queries)
                - index_target: Indices of corresponding ground truth objects

        Example:
            If batch_size=2 and result is:
            [(tensor([0, 3, 7]), tensor([0, 1, 2])),
             (tensor([2, 5]), tensor([0, 1]))]

            This means:
            - Image 0: query 0->gt 0, query 3->gt 1, query 7->gt 2
            - Image 1: query 2->gt 0, query 5->gt 1
        """
        indices = []

        for batch_idx, target in enumerate(targets):
            # Extract predictions for this image
            batch_logits = predictions["pred_logits"][batch_idx]  # [num_queries, num_classes]
            batch_boxes = predictions["pred_boxes"][batch_idx]     # [num_queries, 4]
            batch_prob = batch_logits.softmax(-1)                  # [num_queries, num_classes]

            # Extract ground truth for this image
            tgt_labels = target["labels"].to(torch.long)           # [num_objects]
            tgt_boxes = target["boxes"].to(batch_boxes.dtype)      # [num_objects, 4]

            # 1. Classification cost: -P(correct class)
            # Shape: [num_queries, num_objects]
            # For each query-target pair, this is the negative probability
            # that the query predicts the target's class
            cost_class = -batch_prob[:, tgt_labels]

            # 2. L1 box regression cost
            # Shape: [num_queries, num_objects]
            # Pairwise L1 distance between all predicted and ground truth boxes
            cost_bbox = torch.cdist(batch_boxes, tgt_boxes, p=1)

            # 3. GIoU cost
            # Shape: [num_queries, num_objects]
            # Negative GIoU (we want to minimize cost, so higher IoU = lower cost)
            cost_giou = -self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(batch_boxes),
                self._box_cxcywh_to_xyxy(tgt_boxes)
            )

            # Compute final cost matrix as weighted sum
            # Shape: [num_queries, num_objects]
            cost_matrix = (
                self.bbox_weighting * cost_bbox +
                self.class_weighting * cost_class +
                self.giou_weighting * cost_giou
            ).cpu()

            # Solve the assignment problem using Hungarian algorithm
            # Returns:
            #   ii: row indices (query indices)
            #   jj: column indices (target indices)
            ii, jj = linear_sum_assignment(cost_matrix)

            # Convert to tensors and store
            indices.append((
                torch.as_tensor(ii, dtype=torch.int64),
                torch.as_tensor(jj, dtype=torch.int64)
            ))

        return indices
