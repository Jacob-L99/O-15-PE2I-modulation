# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:37:37 2024

@author: jacke
"""


import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import ants
import time




def flow_regions_z_score(Z_brain):
    means=Z_brain
    means[means == 0] = np.nan
    # print(np.nanmax(means))
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1 ,1 ,1), (1, 0, 0), (0.5, 0, 0)]  # R, G, B
    n_bins = 10  # Number of bins in the colormap
    
    # Create the colormap
    cmap_name = 'custom_blue_white_red'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    def view_angels(image_array):
        x_dim, y_dim, z_dim = image_array.shape
        first_values_xz_pos_y = np.full((x_dim, z_dim), np.nan)
        first_values_xz_neg_y = np.full((x_dim, z_dim), np.nan)
        
        # Arrays for the x-axis values
        first_values_yz_pos_x = np.full((y_dim, z_dim), np.nan)
        first_values_yz_neg_x = np.full((y_dim, z_dim), np.nan)
        
        # Find first non-NaN value along positive x-axis
        for y in range(y_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[:, y, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_yz_pos_x[y, z] = image_array[first_valid_idx[0], y, z]
        
        # Find first non-NaN value along negative x-axis
        for y in range(y_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[::-1, y, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_yz_neg_x[y, z] = image_array[-(first_valid_idx[0] + 1), y, z]
        
        # Find first non-NaN value along positive y-axis
        for x in range(x_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[x, :, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_xz_pos_y[x, z] = image_array[x, first_valid_idx[0], z]
                    
        # Find first non-NaN value along negative y-axis
        for x in range(x_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[x, ::-1, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_xz_neg_y[x, z] = image_array[x, -(first_valid_idx[0] + 1), z]
       
        return first_values_yz_pos_x, first_values_yz_neg_x, first_values_xz_pos_y, first_values_xz_neg_y
    
    means_right=means.copy()
    means_left=means.copy()
    for z in range(128):
        for x in range(63,128):
            for y in range(153):
                means_right[x,y,z]=np.nan
    for z in range(128):
        for x in range(64):
            for y in range(153):
                means_left[x,y,z]=np.nan
    first_values_yz_pos_x0, first_values_yz_neg_x0, first_values_xz_pos_y0, first_values_xz_neg_y0= view_angels(means_right)
    first_values_yz_pos_x1, first_values_yz_neg_x1, first_values_xz_pos_y1, first_values_xz_neg_y1= view_angels(means_left)
    first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2= view_angels(means)
    
    
    def plot_first_values(vmin=None, vmax=None):
        fig, axes = plt.subplots(1, 6, figsize=(20, 6))
        
        # Plot for the first non-NaN value along positive x-axis
        im1 = axes[0].imshow(np.rot90(np.rot90(np.rot90(first_values_yz_neg_x0))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Dex inside')
        axes[0].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative x-axis
        im2 = axes[1].imshow(np.flip(np.rot90(np.rot90(np.rot90(first_values_yz_pos_x1))), axis=1), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('Sin inside)')
        axes[1].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im3 = axes[2].imshow(np.flip(np.rot90(np.rot90(np.rot90(first_values_yz_pos_x2))), axis=1), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title('Dex')
        axes[2].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im4 = axes[3].imshow(np.rot90(np.rot90(np.rot90(first_values_yz_neg_x2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[3].set_title('Sin')
        axes[3].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im5 = axes[4].imshow(np.rot90(np.rot90(np.rot90(first_values_xz_pos_y2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[4].set_title('Back')
        axes[4].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im6 = axes[5].imshow(np.rot90(np.rot90(np.rot90(first_values_xz_neg_y2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[5].set_title('Front')
        axes[5].axis('off')  # Remove axis
    
        # Add one shared colorbar for all subplots
        fig.subplots_adjust(right=0.85)  # Adjust to leave space for colorbar
        cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax, label='ml/cm3/min')
    
        plt.tight_layout()
        plt.show()
    
    # Example usage:
    plot_first_values(vmin=-5, vmax=5)
    return first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2

# R_I=np.load('Z_brain.npy')
# first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2 = flow_regions_z_score(R_I)
# #%%
# from Z_score_brain_surface_parkinson import fig_get_flow_regions
# fig_get_flow_regions(first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2)