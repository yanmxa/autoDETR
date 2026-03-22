"""
Plot experiment history from results.tsv

Usage:
    autodetr-plot                    # reads results.tsv, saves experiment_history.png
"""

import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def read_results(tsv_path='results.tsv'):
    """Read results.tsv and return list of experiment dicts."""
    path = Path(tsv_path)
    if not path.exists():
        print(f"Error: {tsv_path} not found. Run some experiments first.")
        sys.exit(1)

    experiments = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                row['val_loss'] = float(row['val_loss'])
                row['accuracy'] = float(row['accuracy'])
                row['mean_iou'] = float(row['mean_iou'])
            except (ValueError, KeyError):
                continue
            experiments.append(row)

    if not experiments:
        print("No valid experiments found in results.tsv")
        sys.exit(1)

    return experiments


def plot_history(experiments, save_path='experiment_history.png'):
    """Plot accuracy, val_loss, and mean_iou across experiments."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    indices = list(range(len(experiments)))
    descriptions = [e.get('description', '')[:30] for e in experiments]
    statuses = [e.get('status', 'unknown') for e in experiments]

    # Color by status
    colors = []
    for s in statuses:
        if s == 'keep':
            colors.append('#2ecc71')
        elif s == 'discard':
            colors.append('#e74c3c')
        elif s == 'crash':
            colors.append('#95a5a6')
        else:
            colors.append('#3498db')

    # Accuracy
    acc = [e['accuracy'] for e in experiments]
    ax1.bar(indices, acc, color=colors, alpha=0.7, edgecolor='white')
    ax1.plot(indices, acc, 'ko-', markersize=4, linewidth=1)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Experiment History', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Highlight best accuracy
    if acc:
        best_idx = max(range(len(acc)), key=lambda i: acc[i])
        ax1.bar(best_idx, acc[best_idx], color='gold', edgecolor='black', linewidth=1.5)
        ax1.annotate(f'Best: {acc[best_idx]:.3f}', xy=(best_idx, acc[best_idx]),
                    xytext=(0, 10), textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    # Val Loss
    val_losses = [e['val_loss'] for e in experiments]
    ax2.bar(indices, val_losses, color=colors, alpha=0.7, edgecolor='white')
    ax2.plot(indices, val_losses, 'ko-', markersize=4, linewidth=1)
    ax2.set_ylabel('Val Loss', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Highlight best val_loss (lowest non-zero)
    valid_losses = [(i, v) for i, v in enumerate(val_losses) if v > 0]
    if valid_losses:
        best_idx, best_val = min(valid_losses, key=lambda x: x[1])
        ax2.bar(best_idx, best_val, color='gold', edgecolor='black', linewidth=1.5)
        ax2.annotate(f'Best: {best_val:.3f}', xy=(best_idx, best_val),
                    xytext=(0, 10), textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    # Mean IoU
    ious = [e['mean_iou'] for e in experiments]
    ax3.bar(indices, ious, color=colors, alpha=0.7, edgecolor='white')
    ax3.plot(indices, ious, 'ko-', markersize=4, linewidth=1)
    ax3.set_ylabel('Mean IoU', fontsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # X-axis labels
    ax3.set_xticks(indices)
    ax3.set_xticklabels(descriptions, rotation=45, ha='right', fontsize=8)
    ax3.set_xlabel('Experiments', fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='keep'),
        Patch(facecolor='#e74c3c', label='discard'),
        Patch(facecolor='#95a5a6', label='crash'),
        Patch(facecolor='gold', label='best'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Experiment history saved to: {save_path}")


def main():
    experiments = read_results()
    plot_history(experiments)


if __name__ == '__main__':
    main()
