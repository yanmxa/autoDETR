"""
DETR (Detection Transformer) Model Implementation.

This module implements the complete DETR architecture including:
- ResNet-50 backbone
- Transformer encoder-decoder
- Classification and bounding box prediction heads
- 2D sine-cosine positional encoding
"""

import torch
import torch.nn as nn
import math
from torchvision.models import resnet50, ResNet50_Weights
from detr.utils import display_model_info, display_checkpoint_loaded


def _get_1d_sincos_pos_embed(length: int, dim: int, temperature: float = 10000.0, device=None):
    """
    Generate 1D sine-cosine positional embeddings.

    Args:
        length: Sequence length
        dim: Embedding dimension (must be even)
        temperature: Temperature parameter for frequency scaling
        device: Target device

    Returns:
        Positional embeddings of shape (length, dim)
    """
    assert dim % 2 == 0, "Embedding dimension must be even"

    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(temperature) / dim)
    )  # (dim/2)

    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

    return pe  # (L, dim)


def build_2d_sincos_position_embedding(height: int, width: int, dim: int, device=None):
    """
    Create 2D sine-cosine positional encoding.

    Strategy:
    - First half of dimensions encode Y-axis positions
    - Second half encode X-axis positions
    - Each spatial position gets a unique embedding

    Args:
        height: Feature map height
        width: Feature map width
        dim: Embedding dimension (must be even)
        device: Target device

    Returns:
        Positional embeddings of shape (1, H*W, dim)

    Example:
        For a 7x7 feature map with 256-dim embeddings:
        >>> pos_embed = build_2d_sincos_position_embedding(7, 7, 256)
        >>> pos_embed.shape
        torch.Size([1, 49, 256])
    """
    assert dim % 2 == 0, "Positional embedding dimension must be even"

    dim_half = dim // 2

    # Generate 1D embeddings for each axis
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)  # (H, dim/2)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)   # (W, dim/2)

    # Combine to create 2D grid
    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)      # Y-axis encoding
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)    # X-axis encoding

    # Flatten spatial dimensions
    pos = pos.view(1, height * width, dim)  # (1, H*W, dim)

    return pos


class DETR(nn.Module):
    """
    DETR: Detection Transformer for end-to-end object detection.

    Architecture:
        Input Image
            ↓
        ResNet-50 Backbone
            ↓
        Conv 2048→hidden_dim
            ↓
        Flatten + Positional Encoding
            ↓
        Transformer Encoder
            ↓
        Transformer Decoder (with learnable queries)
            ↓
        Classification Head + BBox Head
            ↓
        Predictions

    Key Features:
    - No hand-crafted components (anchors, NMS)
    - Direct set prediction using learnable object queries
    - Parallel prediction of all objects
    - Global reasoning via self-attention
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        dropout: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize DETR model.

        Args:
            num_classes: Number of object categories (excluding background)
            hidden_dim: Transformer hidden dimension
            nheads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_queries: Number of object queries
            dropout: Dropout rate
            verbose: Whether to print model info

        Example:
            >>> model = DETR(num_classes=91, num_queries=100)
            >>> output = model(images)  # images: [B, 3, H, W]
            >>> output['pred_logits'].shape  # [B, 100, 92]
            >>> output['pred_boxes'].shape   # [B, 100, 4]
        """
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.dropout = dropout

        # ============================================================
        # 1. Backbone: ResNet-50
        # ============================================================
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove final FC layer (we don't need ImageNet classification)
        self.backbone.fc = nn.Identity()

        # ============================================================
        # 2. Feature Projection: 2048 → hidden_dim
        # ============================================================
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # ============================================================
        # 3. Transformer
        # ============================================================
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,  # Standard 4x expansion
            dropout=dropout,
            batch_first=True
        )

        # ============================================================
        # 4. Prediction Heads
        # ============================================================
        # Classification: num_classes + 1 (extra class for "no object")
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)

        # Bounding box: 4 coordinates (cx, cy, w, h) normalized to [0, 1]
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # ============================================================
        # 5. Learnable Object Queries
        # ============================================================
        # These are learned during training to specialize for different objects
        self.query_pos = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # ============================================================
        # 6. Layer Normalization
        # ============================================================
        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

        # Display model info using centralized display function
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

            display_model_info(
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                nheads=self.nheads,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                num_queries=self.num_queries,
                dropout=self.dropout,
                total_params=total_params,
                trainable_params=trainable_params
            )

    def forward(self, inputs):
        """
        Forward pass through DETR.

        Args:
            inputs: Input images [B, 3, H, W]

        Returns:
            Dictionary containing:
                - 'pred_logits': [B, num_queries, num_classes+1] - class predictions
                - 'pred_boxes': [B, num_queries, 4] - box predictions (cx, cy, w, h)

        Example:
            >>> model = DETR(num_classes=3)
            >>> images = torch.randn(2, 3, 224, 224)
            >>> output = model(images)
            >>> output['pred_logits'].shape
            torch.Size([2, 100, 4])
            >>> output['pred_boxes'].shape
            torch.Size([2, 100, 4])
        """
        # ============================================================
        # 1. Backbone Feature Extraction
        # ============================================================
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # Output: [B, 2048, H/32, W/32]

        # ============================================================
        # 2. Feature Projection
        # ============================================================
        feat = self.conv(x)  # [B, hidden_dim, Hf, Wf]
        bsz, d_model, Hf, Wf = feat.shape

        # ============================================================
        # 3. Prepare Encoder Input
        # ============================================================
        # Flatten spatial dimensions: [B, hidden_dim, Hf, Wf] → [B, Hf*Wf, hidden_dim]
        src = feat.flatten(2).permute(0, 2, 1)  # [B, Hf*Wf, hidden_dim]

        # Add positional encoding
        pos = build_2d_sincos_position_embedding(
            Hf, Wf, d_model, device=feat.device
        )  # [1, Hf*Wf, hidden_dim]

        src = self.norm_src(src + pos)

        # ============================================================
        # 4. Prepare Decoder Input
        # ============================================================
        # Decoder target: zero content + learned query positional encodings
        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        # ============================================================
        # 5. Transformer Processing
        # ============================================================
        hs = self.transformer(src=src, tgt=tgt)  # [B, num_queries, hidden_dim]

        # ============================================================
        # 6. Prediction Heads
        # ============================================================
        # Classification logits
        pred_logits = self.linear_class(hs)  # [B, num_queries, num_classes+1]

        # Bounding box coordinates (apply sigmoid to normalize to [0, 1])
        pred_boxes = self.linear_bbox(hs).sigmoid()  # [B, num_queries, 4]

        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }

    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained weights with rich logging.

        Args:
            checkpoint_path: Path to checkpoint file

        Example:
            >>> model = DETR(num_classes=91)
            >>> model.load_pretrained('detr_checkpoint.pth')
        """
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(state_dict)
            display_checkpoint_loaded(checkpoint_path, success=True)

        except Exception as e:
            display_checkpoint_loaded(checkpoint_path, success=False, error_msg=str(e))
            raise


if __name__ == '__main__':
    """Test DETR model."""
    from torchinfo import summary
    from rich import box
    from rich.panel import Panel
    from detr.utils import print_message

    # ============================================================
    # Demo Configuration (Lightweight for Testing)
    # ============================================================
    # Using 1 encoder + 1 decoder layer for FAST testing/development
    #
    # Key Parameters Explained:
    # -------------------------
    # • num_encoder_layers / num_decoder_layers:
    #   - Controls reasoning depth (more layers = better accuracy)
    #   - 1 layer: Fast (~50 FPS), AP ~23% (low accuracy)
    #   - 6 layers: Slow (~28 FPS), AP ~42% (good accuracy)
    #   - Production standard: 6+6 layers
    #
    # • nheads (attention heads):
    #   - Number of parallel attention mechanisms
    #   - Each head focuses on different aspects of the image
    #   - Example with 8 heads:
    #     * Head 1: Focuses on object boundaries
    #     * Head 2: Focuses on textures
    #     * Head 3: Focuses on spatial relationships
    #     * Head 4-8: Other patterns
    #   - More heads = richer feature representation
    #   - Standard: 8 heads (good balance)
    #   - Trade-off: More heads → More computation
    #
    # • num_queries:
    #   - Number of "detection slots" (max objects to detect)
    #   - Each query becomes a specialized object detector
    #   - 25 queries: Good for simple scenes
    #   - 100 queries: Standard for COCO (dense scenes)
    #
    # • AP (Average Precision):
    #   - Detection accuracy metric (0-100%, higher is better)
    #   - COCO benchmark uses AP@[0.5:0.95]
    #   - Good model: AP > 40%
    #
    # Configuration Comparison:
    # ┌─────────┬────────┬──────┬─────────┬──────────┬─────────────┐
    # │ Config  │ Layers │ Heads│ Params  │ Speed    │ AP (COCO)   │
    # ├─────────┼────────┼──────┼─────────┼──────────┼─────────────┤
    # │ Demo    │ 1+1    │ 8    │ ~26M    │ ~50 FPS  │ ~23% (low)  │
    # │ Light   │ 2+2    │ 8    │ ~28M    │ ~45 FPS  │ ~30%        │
    # │ Medium  │ 4+4    │ 8    │ ~35M    │ ~35 FPS  │ ~36%        │
    # │ Standard│ 6+6    │ 8    │ ~41M    │ ~28 FPS  │ ~42% (good) │
    # └─────────┴────────┴──────┴─────────┴──────────┴─────────────┘
    #
    # For PRODUCTION: Use 6 encoder + 6 decoder layers
    # For LEARNING/TESTING: Use 1-2 layers (current setting)
    # ============================================================

    # Create model (1 layer for demo)
    print_message("\n[bold cyan]Creating DETR model...[/bold cyan]\n")
    model = DETR(
        num_classes=3,          # Number of object classes (excluding background)
        num_queries=25,         # Number of detection slots (max objects per image)
        num_encoder_layers=1,   # Demo: 1 layer = fast but AP ~23% (low accuracy)
        num_decoder_layers=1,   # Demo: 1 layer = fast but AP ~23% (low accuracy)
        nheads=8,               # Attention heads: 8 parallel attention mechanisms
        verbose=True            # Print model architecture info
    )
    # For production/high accuracy: set num_encoder_layers=6, num_decoder_layers=6 → AP ~42%

    # Display model summary
    print_message("[bold cyan]Model Summary:[/bold cyan]\n")
    summary(model, input_size=(2, 3, 224, 224), device='cpu')

    # Test forward pass
    print_message("\n[bold cyan]Testing forward pass...[/bold cyan]\n")
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print_message(Panel(
        f"[green]Input shape:[/green] {list(dummy_input.shape)}\n"
        f"[green]Pred logits shape:[/green] {list(output['pred_logits'].shape)}\n"
        f"[green]Pred boxes shape:[/green] {list(output['pred_boxes'].shape)}",
        title="✅ Forward Pass Successful",
        border_style="green",
        box=box.ROUNDED
    ))
    print_message()
