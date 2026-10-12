"""Renders a visit-count grid as an annotated percentage heatmap image tensor."""
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def generate_annotated_heatmap(heatmap_counts):
    """Returns a [1, H, W, 4] PNG tensor of the heatmap with per-cell percentages."""
    total = np.sum(heatmap_counts)
    percentage_grid = (heatmap_counts / total) * 100 if total > 0 else heatmap_counts

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(percentage_grid, cmap='Blues', vmin=0, vmax=100)
    for row in range(percentage_grid.shape[0]):
        for column in range(percentage_grid.shape[1]):
            value = percentage_grid[row, column]
            ax.text(
                column, row, f'{value:.1f}%',
                va='center', ha='center',
                color='white' if value > 45.0 else 'black',
                fontsize=9, weight='bold',
            )
    size = percentage_grid.shape[0]
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    image_tensor = tf.image.decode_png(buffer.getvalue(), channels=4)
    buffer.close()
    return tf.expand_dims(image_tensor, 0)
