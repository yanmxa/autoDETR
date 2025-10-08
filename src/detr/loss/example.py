"""
Example usage of DETR loss function.

This script demonstrates how to use the HungarianMatcher and DETRLoss
with sample predictions and ground truth data.
"""

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box


def main():
    """Demonstrate DETR loss computation."""

    from matcher import HungarianMatcher
    from criterion import DETRLoss, compute_total_loss

    console = Console()

    # Configuration
    num_classes = 5  # Number of object classes (excluding background)
    batch_size = 2
    num_queries = 10  # Number of object queries in DETR

    # Loss weights
    weight_dict = {
        'class_weighting': 1.0,   # Weight for classification loss
        'bbox_weighting': 5.0,    # Weight for L1 box regression loss
        'giou_weighting': 2.0     # Weight for GIoU loss
    }

    # Initialize matcher and loss criterion
    matcher = HungarianMatcher(weight_dict)
    criterion = DETRLoss(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1  # Lower weight for "no-object" class to handle class imbalance
    )

    # ============================================================
    # Create mock model predictions
    # ============================================================
    predictions = {
        # Classification logits: [batch_size, num_queries, num_classes+1]
        # Last class (index=num_classes) is the "no-object" class
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1, requires_grad=True),

        # Predicted boxes: [batch_size, num_queries, 4]
        # Format: (center_x, center_y, width, height) normalized to [0, 1]
        'pred_boxes': torch.rand(batch_size, num_queries, 4, requires_grad=True)
    }

    # ============================================================
    # Create ground truth targets
    # ============================================================
    # Each image can have different number of objects
    targets = [
        # Image 0: 3 objects
        {
            'labels': torch.tensor([1, 1, 2]),  # Class indices [0, num_classes-1]
            'boxes': torch.tensor([
                [0.5, 0.5, 0.2, 0.3],   # Object 1: center=(0.5,0.5), size=(0.2,0.3)
                [0.3, 0.7, 0.1, 0.2],   # Object 2
                [0.8, 0.2, 0.15, 0.25]  # Object 3
            ])
        },
        # Image 1: 1 object
        {
            'labels': torch.tensor([3]),
            'boxes': torch.tensor([[0.4, 0.6, 0.3, 0.4]])
        }
    ]

    # ============================================================
    # Display Configuration
    # ============================================================
    console.print()
    console.print(Panel.fit(
        "[bold cyan]DETR Loss Computation Example[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Explain what queries are
    console.print()
    console.print(Panel(
        "[bold yellow]What are \"Queries\" in DETR?[/bold yellow]\n\n"
        "In DETR (Detection Transformer), [bold cyan]queries[/bold cyan] are [bold]learnable embeddings[/bold] that represent\n"
        "\"object slots\" - potential objects that might exist in the image.\n\n"
        "[cyan]Architecture overview:[/cyan]\n"
        "  1. Image → CNN Backbone → Feature map\n"
        "  2. Feature map → Transformer Encoder → Enhanced features\n"
        "  3. [bold]N learnable queries[/bold] (e.g., 100) → Transformer Decoder\n"
        "  4. Each query outputs: [class probabilities] + [bounding box]\n\n"
        "[green]Why use queries?[/green]\n"
        "  • Traditional detectors: Use anchor boxes (hand-designed)\n"
        "  • DETR: Uses learned queries (data-driven)\n"
        "  • Each query asks: Is there an object like me in the image?\n\n"
        f"[magenta]In this example:[/magenta]\n"
        f"  • We have [bold]{num_queries} queries[/bold] (Q0 to Q{num_queries-1})\n"
        f"  • Image 0 has [bold]{len(targets[0]['labels'])} objects[/bold]\n"
        f"  • Image 1 has [bold]{len(targets[1]['labels'])} objects[/bold]\n"
        f"  • Hungarian matching determines which queries detect which objects\n"
        f"  • [dim]Unmatched queries are trained to predict no object[/dim]\n\n"
        "[bold]Key insight:[/bold] Queries are [bold red]learnable parameters[/bold red] that the model optimizes\n"
        "to become specialized detectors for different types of objects!",
        title="🔍 Understanding DETR Queries",
        border_style="magenta",
        box=box.ROUNDED
    ))

    # Configuration table
    config_table = Table(title="⚙️  Configuration", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="yellow")

    config_table.add_row("Number of classes", str(num_classes))
    config_table.add_row("Batch size", str(batch_size))
    config_table.add_row("Number of queries", str(num_queries))
    config_table.add_row("Objects in batch", str([len(t['labels']) for t in targets]))
    config_table.add_row("Total objects", str(sum(len(t['labels']) for t in targets)))

    console.print(config_table)

    # Loss weights table
    weights_table = Table(title="⚖️  Loss Weights", box=box.ROUNDED, show_header=True, header_style="bold green")
    weights_table.add_column("Loss Component", style="cyan")
    weights_table.add_column("Weight", style="yellow", justify="right")

    weights_table.add_row("Classification", f"{weight_dict['class_weighting']:.1f}")
    weights_table.add_row("BBox Regression (L1)", f"{weight_dict['bbox_weighting']:.1f}")
    weights_table.add_row("GIoU", f"{weight_dict['giou_weighting']:.1f}")

    console.print(weights_table)

    # Forward pass through loss criterion
    console.print()
    with console.status("[bold green]Computing loss...", spinner="dots"):
        loss_dict = criterion(predictions, targets)
        total_loss = compute_total_loss(loss_dict, weight_dict)

    # ============================================================
    # Display Loss Results
    # ============================================================
    console.print()

    # Loss components table
    loss_table = Table(
        title="📊 Loss Components",
        box=box.HEAVY,
        show_header=True,
        header_style="bold blue"
    )
    loss_table.add_column("Loss Type", style="cyan", no_wrap=True)
    loss_table.add_column("Raw Value", style="yellow", justify="right")
    loss_table.add_column("Weight", style="magenta", justify="right")
    loss_table.add_column("Weighted Value", style="green bold", justify="right")

    # Classification loss
    ce_loss = loss_dict['labels']['loss_ce']
    loss_table.add_row(
        "Classification (CE)",
        f"{ce_loss.item():.4f}",
        f"×{weight_dict['class_weighting']:.1f}",
        f"{ce_loss.item() * weight_dict['class_weighting']:.4f}"
    )

    # Box regression loss (L1)
    bbox_loss = loss_dict['boxes']['loss_bbox']
    loss_table.add_row(
        "Box Regression (L1)",
        f"{bbox_loss.item():.4f}",
        f"×{weight_dict['bbox_weighting']:.1f}",
        f"{bbox_loss.item() * weight_dict['bbox_weighting']:.4f}"
    )

    # GIoU loss
    giou_loss = loss_dict['boxes']['loss_giou']
    loss_table.add_row(
        "GIoU Loss",
        f"{giou_loss.item():.4f}",
        f"×{weight_dict['giou_weighting']:.1f}",
        f"{giou_loss.item() * weight_dict['giou_weighting']:.4f}"
    )

    console.print(loss_table)

    # Total loss panel
    console.print()
    console.print(Panel(
        f"[bold yellow]Total Weighted Loss: [bold red]{total_loss.item():.4f}[/bold red]",
        title="🎯 Final Loss",
        border_style="red",
        box=box.DOUBLE
    ))

    # ============================================================
    # Demonstrate matching results with cost matrix
    # ============================================================
    console.print()

    with torch.no_grad():
        indices = matcher(predictions, targets)

    # Show cost matrix for each image
    for img_idx, (pred_idx, tgt_idx) in enumerate(indices):
        console.print()
        console.print(Panel.fit(
            f"[bold yellow]Image {img_idx}[/bold yellow] - Cost Matrix Analysis",
            border_style="yellow"
        ))

        num_objects = len(targets[img_idx]['labels'])

        # Compute cost matrix for this image
        batch_logits = predictions["pred_logits"][img_idx]
        batch_boxes = predictions["pred_boxes"][img_idx]
        batch_prob = batch_logits.softmax(-1)

        tgt_labels = targets[img_idx]["labels"].to(torch.long)
        tgt_boxes = targets[img_idx]["boxes"].to(batch_boxes.dtype)

        # Compute costs
        cost_class = -batch_prob[:, tgt_labels]
        cost_bbox = torch.cdist(batch_boxes, tgt_boxes, p=1)

        # Compute GIoU cost
        cost_giou = -matcher._generalized_box_iou(
            matcher._box_cxcywh_to_xyxy(batch_boxes),
            matcher._box_cxcywh_to_xyxy(tgt_boxes)
        )

        # Total cost
        cost_matrix = (
            weight_dict['bbox_weighting'] * cost_bbox +
            weight_dict['class_weighting'] * cost_class +
            weight_dict['giou_weighting'] * cost_giou
        )

        # Create cost table showing top-5 queries for each object
        for obj_idx in range(num_objects):
            obj_class = targets[img_idx]['labels'][obj_idx].item()
            obj_box = targets[img_idx]['boxes'][obj_idx]

            console.print()
            cost_table = Table(
                title=f"📋 Object {obj_idx} (class {obj_class}) - Top 5 Query Candidates",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan"
            )

            cost_table.add_column("Rank", style="dim", width=6)
            cost_table.add_column("Query", style="cyan", justify="center")
            cost_table.add_column("Total Cost", style="yellow", justify="right")
            cost_table.add_column("Class", style="magenta", justify="right")
            cost_table.add_column("BBox", style="green", justify="right")
            cost_table.add_column("GIoU", style="blue", justify="right")
            cost_table.add_column("Selected", style="bold", justify="center")

            # Get costs for this object across all queries
            obj_costs = cost_matrix[:, obj_idx]
            sorted_indices = torch.argsort(obj_costs)[:5]  # Top 5 lowest costs

            for rank, query_idx in enumerate(sorted_indices, 1):
                total_cost = cost_matrix[query_idx, obj_idx].item()
                class_cost = cost_class[query_idx, obj_idx].item()
                bbox_cost = cost_bbox[query_idx, obj_idx].item()
                giou_cost = cost_giou[query_idx, obj_idx].item()

                # Check if this query was selected
                is_selected = query_idx.item() in pred_idx.tolist() and obj_idx in tgt_idx[pred_idx.tolist().index(query_idx.item())]
                selected_mark = "[bold green]✓ SELECTED[/bold green]" if is_selected else ""

                # Color code the row if selected
                row_style = "bold green" if is_selected else ""

                cost_table.add_row(
                    f"#{rank}",
                    f"Q{query_idx.item()}",
                    f"{total_cost:.3f}",
                    f"{class_cost:.3f}",
                    f"{bbox_cost:.3f}",
                    f"{giou_cost:.3f}",
                    selected_mark,
                    style=row_style
                )

            console.print(cost_table)

    # Show final matching summary with explanation
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Hungarian Algorithm Selected:[/bold cyan]\n\n" +
        "\n".join([
            f"Image {img_idx}: " + ", ".join([
                f"[green]Q{p}→Obj{t}[/green]"
                for p, t in zip(pred_idx.tolist(), tgt_idx.tolist())
            ])
            for img_idx, (pred_idx, tgt_idx) in enumerate(indices)
        ]) +
        f"\n\n[dim]These assignments minimize the [bold]total cost[/bold] across all objects.[/dim]",
        title="🎯 Final Matching Result",
        border_style="green",
        box=box.DOUBLE
    ))

    # Add explanation for Image 0's interesting case
    if len(indices) > 0:
        console.print()
        console.print(Panel(
            "[bold yellow]Why Q8→Obj0 instead of Q9→Obj0?[/bold yellow]\n\n"
            "[cyan]Greedy approach (wrong):[/cyan]\n"
            "  • Q9→Obj0: cost = 2.944 (lowest for Obj0)\n"
            "  • Q5→Obj1: cost = 5.489 (lowest for Obj1)\n"
            "  • Q8→Obj2: cost = 2.844 (2nd lowest for Obj2)\n"
            "  [red]Total: 2.944 + 5.489 + 2.844 = 11.277[/red]\n\n"
            "[green]Hungarian approach (optimal):[/green]\n"
            "  • Q8→Obj0: cost = 3.353 (2nd lowest for Obj0)\n"
            "  • Q5→Obj1: cost = 5.489 (lowest for Obj1)\n"
            "  • Q9→Obj2: cost = 1.420 (lowest for Obj2) ✨\n"
            "  [green bold]Total: 3.353 + 5.489 + 1.420 = 10.262[/green bold] ✓\n\n"
            "[bold]Key insight:[/bold]\n"
            "Q9 is [bold red]better at matching Obj2[/bold red] (cost 1.420) than Obj0 (cost 2.944).\n"
            "By sacrificing Obj0's best match (+0.409), we gain a much better Obj2 match (-1.424).\n"
            "Net benefit: -1.424 + 0.409 = [bold green]-1.015[/bold green] (lower total cost!)\n\n"
            "[dim]This is why Hungarian algorithm finds [bold]globally optimal[/bold] assignments,\n"
            "while greedy selection would be stuck in a [bold]local optimum[/bold].[/dim]",
            title="💡 Global vs Local Optimization",
            border_style="blue",
            box=box.ROUNDED
        ))

    console.print()

    return total_loss


if __name__ == "__main__":
    console = Console()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run example
    loss = main()

    # Show that loss is differentiable
    console.print()
    with console.status("[bold green]Verifying gradient computation...", spinner="dots"):
        loss.backward()

    console.print(Panel(
        "[bold green]✓[/bold green] Gradients computed successfully!\n"
        "[bold cyan]Loss computation complete. Ready for training![/bold cyan]",
        title="✅ Verification Complete",
        border_style="green",
        box=box.ROUNDED
    ))
    console.print()
