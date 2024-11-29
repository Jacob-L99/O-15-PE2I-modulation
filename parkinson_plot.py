# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:11:38 2024

@author: jacke
"""
import matplotlib
cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.1, 0.5, 0.5),
                  (0.2, 0.0, 0.0),
                  (0.4, 0.2, 0.2),
                  (0.6, 0.0, 0.0),
                  (0.8, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                  (0.1, 0.0, 0.0),
                  (0.2, 0.0, 0.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 1.0, 1.0),
                  (0.8, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                  (0.1, 0.5, 0.5),
                  (0.2, 1.0, 1.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

import numpy as np
import matplotlib.pyplot as plt

def plot_specific_slices(volume, axis=0, slice_indices=None, vmin=0, vmax=None, title=None):

    # Check if volume is a 3D array
    if len(volume.shape) != 3:
        raise ValueError("Input volume must be a 3D numpy array")

    # Check if slice_indices is provided and is a list
    if slice_indices is None or not isinstance(slice_indices, list):
        raise ValueError("slice_indices must be provided as a list of integers")


    # Create subplots for displaying the slices
    num_slices = len(slice_indices)
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))

    # Plot each slice
    im = None  # Variable to hold the last image for the colorbar
    middle_idx = num_slices // 2  # Calculate middle slice index
    for i, idx in enumerate(slice_indices):
        if axis == 0:
            slice_img = volume[idx, :, :]
        elif axis == 1:
            slice_img = volume[:, idx, :]
        else:
            slice_img = volume[:, :, idx]

        # Display the slice with user-defined or default vmin and vmax
        im = axes[i].imshow(np.rot90(slice_img), cmap=my_cmap, vmin=vmin, vmax=vmax)
        axes[i].axis('off')  # Turn off axis labels for clarity

        # Add the provided title to the middle slice
        if i == middle_idx and title is not None:
            axes[i].set_title(title, fontsize=14)

    # Add a colorbar to the last image
    fig.colorbar(im, ax=axes[-1], orientation='vertical')

    plt.tight_layout()
    plt.show()
