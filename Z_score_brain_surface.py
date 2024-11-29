# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:16:43 2024

@author: jacke
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import SimpleITK as sitk
import matplotlib
import os
import sys
import numpy as np


# Function to determine the correct file path
def get_file_path(filename):
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, look in the temporary directory
        return os.path.join(sys._MEIPASS, filename)
    else:
        # Otherwise, look in the current working directory
        return os.path.join(os.getcwd(), filename)
    
def shift_right_zero_fill(matrix, x):
    """
    Shifts the 2D NumPy array `matrix` to the right by `x` steps, filling with zeros.
    
    Parameters:
        matrix (np.ndarray): The input 2D array.
        x (int): Number of steps to shift.
        
    Returns:
        np.ndarray: The shifted array with zeros filled in.
    """
    num_cols = matrix.shape[1]
    x = min(x, num_cols)  # Prevent shifting more than the number of columns
    # Create a zero array of the same type and appropriate shape
    zeros = np.zeros((matrix.shape[0], x), dtype=matrix.dtype)
    # Slice the matrix to exclude the last x columns
    shifted_part = matrix[:, :-x] if x != 0 else matrix
    # Concatenate zeros to the beginning of each row
    shifted_matrix = np.concatenate((zeros, shifted_part), axis=1)
    return shifted_matrix

def shift_left_zero_fill(matrix, x):
    """
    Shifts the 2D NumPy array `matrix` to the left by `x` steps, filling with zeros.
    
    Parameters:
        matrix (np.ndarray): The input 2D array.
        x (int): Number of steps to shift.
        
    Returns:
        np.ndarray: The shifted array with zeros filled in.
    """
    num_cols = matrix.shape[1]
    x = min(x, num_cols)  # Prevent shifting more than the number of columns
    # Slice the matrix to exclude the first x columns
    shifted_part = matrix[:, x:] if x != 0 else matrix
    # Create a zero array of the same type and appropriate shape
    zeros = np.zeros((matrix.shape[0], x), dtype=matrix.dtype)
    # Concatenate the shifted part with zeros at the end of each row
    shifted_matrix = np.concatenate((shifted_part, zeros), axis=1)
    return shifted_matrix 
    
plt.style.use('dark_background')

def blend_images(fixed_image_np, transformed_image_colored_np, alpha=0.5):
    """
    Alpha blend the fixed grayscale image and the colored registered image.
    Args:
        fixed_image_np: The numpy array of the fixed (template) grayscale image.
        transformed_image_colored_np: The numpy array of the colored registered image (RGB format).
        alpha: The blending factor for the fixed image. The transformed image is always shown at full intensity.
    Returns:
        A blended numpy array of the two images.
    """
    # Normalize the fixed grayscale image to [0, 1]
    fixed_image_norm = (fixed_image_np - fixed_image_np.min()) / (fixed_image_np.max() - fixed_image_np.min())

    # Convert the fixed image to RGB format by stacking it
    fixed_image_rgb = np.stack([fixed_image_norm]*3, axis=-1)

    # Blend the images using the formula: blended = alpha * fixed + (1 - alpha) * transformed
    blended_image = alpha * fixed_image_rgb + (1 - alpha) * transformed_image_colored_np
    
    # Clip the values to the [0, 1] range to avoid overflow
    blended_image = np.clip(blended_image, 0, 1)
    
    return blended_image



def mr_angel(project90, pos_x2):
    target_shape = np.rot90(np.rot90(np.rot90(pos_x2))).shape
    
    # Get the current shape of projection90
    current_shape = project90.shape
    
    # Calculate the padding required for each dimension
    padding = []
    for dim, (current, target) in enumerate(zip(current_shape, target_shape)):
        if dim == 0:  # For the first dimension (rows), pad only at the top
            total_pad = max(target - current, 0)
            pad_before = total_pad  # All padding at the top
            pad_after = 0
        else:  # For the second dimension (columns), pad symmetrically
            total_pad = max(target - current, 0)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        padding.append((pad_before, pad_after))
    
    # Pad the projection90 array with zeros to match the target shape
    projection90_padded = np.pad(project90, padding, mode='constant', constant_values=0)
    projection90_padded = np.flip(projection90_padded, axis=1)
    
    # Remove 20 layers from the bottom and add them to the top of the fixed_slice
    fixed_slice = projection90_padded
    # if fixed_slice.shape[0] > 9:
    #     fixed_slice = np.vstack((fixed_slice[-9:], fixed_slice[:-9]))
    return fixed_slice

def fig_get(neg_x0, pos_x1, pos_x2, neg_x2, pos_y2, neg_y2):
    plt.style.use('dark_background')
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1, 1, 1), (1, 0, 0), (0.5, 0, 0)]  # R, G, B
    n_bins = 10  # Number of bins in the colormap
    cmap_name = 'custom_blue_white_red'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    def temp_name(pet_image, fixed_slice):
        # Create the colormap
        
        
        # Existing code
        alpha_value = 0.6  # Fixed alpha value
        transformed_slice = pet_image
        transformed_slice = np.nan_to_num(transformed_slice, nan=0)
        
        # Clip and normalize the transformed slice to the range [-5, 5]
        transformed_slice_clipped = np.clip(transformed_slice, -5, 5)
        normalized_transformed_slice = (transformed_slice_clipped + 5) / 10  # Normalizes to [0, 1] range for the colormap
        
        # Apply colormap to the normalized transformed image
        transformed_colored = custom_cmap(normalized_transformed_slice)  # Apply custom colormap
        # transformed_colored is an RGBA array (height, width, 4)
        
        # Create a mask for values between -1 and 1 and set alpha to 0 for those pixels
        alpha_channel = transformed_colored[:, :, 3]  # Extract the alpha channel
        
        # Create a mask where the original values are between -1 and 1
        transparent_mask = (transformed_slice_clipped >= -1) & (transformed_slice_clipped <= 1)
        
        # Set alpha to 0 for values within the range [-1, 1]
        alpha_channel[transparent_mask] = 0  # Fully transparent
        # Set alpha to 1 for values outside the range [-1, 1]
        alpha_channel[~transparent_mask] = 1  # Fully opaque
        
        # Update the alpha channel in transformed_colored
        transformed_colored[:, :, 3] = alpha_channel
        
        # Plot the transformed image with transparency applied
        plt.imshow(transformed_colored)
        plt.axis('off')
        
        # Add a colorbar for the transformed_colored plot with fixed limits
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
        cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.04)
        cbar.set_label('Color Scale')
        cbar.set_ticks(np.arange(-5, 6, 1))  # Set colorbar range from -5 to 5 in steps of 1
        
        plt.show()
        plt.close()
        
        # Generate the mask and expand dimensions
        mask_1 = np.where(transformed_slice != 0, 1, np.nan)
        mask_1_expanded = mask_1[:, :, np.newaxis]  # Shape becomes (128, 158, 1)
        
        # Blend images
        blended_image_np = blend_images(fixed_slice, transformed_colored[:, :, :3], alpha_value)
        blended_image_np = np.where(blended_image_np == 0, np.nan, blended_image_np) * mask_1_expanded
        
        # Plotting the blended image with fixed color limits
        fig_blended, ax_blended = plt.subplots(figsize=(8, 8), dpi=50)  # Increased size and set DPI
        ax_blended.axis('off')  # Turn off axes
        im = ax_blended.imshow(blended_image_np)
        
        # Add a colorbar for the blended image with fixed limits
        cbar_blended = fig_blended.colorbar(sm, ax=ax_blended, fraction=0.03, pad=0.04)
        cbar_blended.set_label('Color Scale')
        cbar_blended.set_ticks(np.arange(-5, 6, 1))  # Set colorbar range from -5 to 5 in steps of 1
        
        plt.show()
        plt.close()
        
        # New Plot: MR image as base, color regions for parametric PET values between [-5, -1] and [1, 5]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=50)
        
        # Plot the MR image in grayscale
        ax.imshow(fixed_slice, cmap='gray')
        
        # Create a mask for the PET values between [-5, -1] and [1, 5]
        highlight_mask = ((transformed_slice_clipped >= -5) & (transformed_slice_clipped <= -1)) | \
                         ((transformed_slice_clipped >= 1) & (transformed_slice_clipped <= 5))
        
        # Apply the colormap only to the regions within the highlight_mask
        highlighted_region = np.zeros_like(transformed_colored) * mask_1_expanded
        highlighted_region[highlight_mask] = transformed_colored[highlight_mask]
        
        # Plot the highlighted regions on top of the MR image
        ax.imshow(highlighted_region)
        ax.axis('off')
        
        # Add a colorbar for the highlighted regions with fixed limits
        cbar_highlight = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar_highlight.set_label('Color Scale')
        cbar_highlight.set_ticks(np.arange(-5, 6, 1))
        
        plt.show()
        plt.close()
        
        # Return the last figure
        return fig
    
    project_dex_medial = np.load(get_file_path('projection_90.npy'))
    fixed_slice_dex_medial = mr_angel(project_dex_medial, neg_x2)
    
    last_figure_4 = temp_name(np.rot90(np.rot90(np.rot90(neg_x2))), fixed_slice_dex_medial)
    
    # Now, you can use loaded_figure as you would use last_figure_4
    
    
    # print(type(last_figure))
    # plt.imshow(np.rot90(np.rot90(np.rot90(neg_x2))))
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    
    project_sin_medial = np.load(get_file_path('projection_270.npy'))
    fixed_slice_sin_medial=mr_angel(project_sin_medial, pos_x2)
    
    last_figure_3 = temp_name(np.flip(np.rot90(np.rot90(np.rot90(pos_x2))),axis=1), fixed_slice_sin_medial)
    
    # plt.imshow(np.flip(np.rot90(np.rot90(np.rot90(pos_x2))),axis=1))
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    
    project_dex = np.load(get_file_path('projection_0.npy'))
    
    x = 2
    project_dex = shift_right_zero_fill(project_dex, x)
    fixed_slice_dex=mr_angel(project_dex, neg_y2)
    last_figure_6 = temp_name(np.rot90(np.rot90(np.rot90(neg_y2))), fixed_slice_dex)
    # plt.imshow(np.rot90(np.rot90(np.rot90(neg_y2))))
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    
    project_sin = np.load(get_file_path('projection_180.npy'))
    fixed_slice_sin=mr_angel(project_sin, pos_y2)
    
    last_figure_5 = temp_name(np.rot90(np.rot90(np.rot90(pos_y2))), fixed_slice_sin)
    # plt.imshow(np.rot90(np.rot90(np.rot90(pos_y2))))
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    
    project_back = np.load(get_file_path('projection_inside_sin.npy'))
    fixed_slice_back=mr_angel(project_back, neg_x0)
    
    last_figure_1 = temp_name(np.rot90(np.rot90(np.rot90(neg_x0))), fixed_slice_back)
    # plt.imshow(np.rot90(np.rot90(np.rot90(neg_x0))))
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    
    project_front = np.load(get_file_path('projection_inside_dex.npy'))
    fixed_slice_front=mr_angel(project_front, pos_x1)
    
    last_figure_2 = temp_name(np.flip(np.rot90(np.rot90(np.rot90(pos_x1))), axis=1), fixed_slice_front)
    # plt.imshow(np.flip(np.rot90(np.rot90(np.rot90(pos_x1))), axis=1))
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    return last_figure_1, last_figure_2, last_figure_3, last_figure_4, last_figure_5, last_figure_6


def fig_get_regions(neg_x0, pos_x1, pos_x2, neg_x2, pos_y2, neg_y2):
    plt.style.use('dark_background')
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1, 1, 1), (1, 0, 0), (0.5, 0, 0)]  # R, G, B
    n_bins = 10  # Number of bins in the colormap
    cmap_name = 'custom_blue_white_red'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    def temp_name(pet_image, fixed_slice):
        # Create the colormap
        
        
        # Existing code
        alpha_value = 0.6  # Fixed alpha value
        transformed_slice = pet_image
        transformed_slice = np.nan_to_num(transformed_slice, nan=0)
        
        # Clip and normalize the transformed slice to the range [-5, 5]
        transformed_slice_clipped = np.clip(transformed_slice, -5, 5)
        normalized_transformed_slice = (transformed_slice_clipped + 5) / 10  # Normalizes to [0, 1] range for the colormap
        
        # Apply colormap to the normalized transformed image
        transformed_colored = custom_cmap(normalized_transformed_slice)  # Apply custom colormap
        # transformed_colored is an RGBA array (height, width, 4)
        
        # Create a mask for values between -1 and 1 and set alpha to 0 for those pixels
        alpha_channel = transformed_colored[:, :, 3]  # Extract the alpha channel
        
        # Create a mask where the original values are between -1 and 1
        transparent_mask = (transformed_slice_clipped >= -1) & (transformed_slice_clipped <= 1)
        
        # Set alpha to 0 for values within the range [-1, 1]
        alpha_channel[transparent_mask] = 0  # Fully transparent
        # Set alpha to 1 for values outside the range [-1, 1]
        alpha_channel[~transparent_mask] = 1  # Fully opaque
        
        # Update the alpha channel in transformed_colored
        transformed_colored[:, :, 3] = alpha_channel
        
        # Plot the transformed image with transparency applied
        plt.imshow(transformed_colored)
        plt.axis('off')
        
        # Add a colorbar for the transformed_colored plot with fixed limits
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
        cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.04)
        cbar.set_label('Color Scale')
        cbar.set_ticks(np.arange(-5, 6, 1))  # Set colorbar range from -5 to 5 in steps of 1
        
        plt.show()
        plt.close()
        
        # Generate the mask and expand dimensions
        mask_1 = np.where(transformed_slice != 0, 1, np.nan)
        mask_1_expanded = mask_1[:, :, np.newaxis]  # Shape becomes (128, 158, 1)
        
        # Blend images
        blended_image_np = blend_images(fixed_slice, transformed_colored[:, :, :3], alpha_value)
        blended_image_np = np.where(blended_image_np == 0, np.nan, blended_image_np) * mask_1_expanded
        
        # Plotting the blended image with fixed color limits
        fig_blended, ax_blended = plt.subplots(figsize=(8, 8), dpi=50)  # Increased size and set DPI
        ax_blended.axis('off')  # Turn off axes
        im = ax_blended.imshow(blended_image_np)
        
        # Add a colorbar for the blended image with fixed limits
        cbar_blended = fig_blended.colorbar(sm, ax=ax_blended, fraction=0.03, pad=0.04)
        cbar_blended.set_label('Color Scale')
        cbar_blended.set_ticks(np.arange(-5, 6, 1))  # Set colorbar range from -5 to 5 in steps of 1
        
        plt.show()
        plt.close()
        
        # New Plot: MR image as base, color regions for parametric PET values between [-5, -1] and [1, 5]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=50)
        
        # Plot the MR image in grayscale
        ax.imshow(fixed_slice, cmap='gray')
        
        # Create a mask for the PET values between [-5, -1] and [1, 5]
        highlight_mask = ((transformed_slice_clipped >= -5) & (transformed_slice_clipped <= -1)) | \
                         ((transformed_slice_clipped >= 1) & (transformed_slice_clipped <= 5))
        
        # Apply the colormap only to the regions within the highlight_mask
        highlighted_region = np.zeros_like(transformed_colored) * mask_1_expanded
        highlighted_region[highlight_mask] = transformed_colored[highlight_mask]
        
        # Plot the highlighted regions on top of the MR image
        ax.imshow(highlighted_region)
        ax.axis('off')
        
        # Add a colorbar for the highlighted regions with fixed limits
        cbar_highlight = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar_highlight.set_label('Color Scale')
        cbar_highlight.set_ticks(np.arange(-5, 6, 1))
        
        plt.show()
        plt.close()
        
        # Return the last figure
        return fig
    
    
    
    project_dex_medial = np.load(get_file_path('projection_90.npy'))
    x = 5
    project_dex_medial = shift_right_zero_fill(project_dex_medial, x)
    fixed_slice_dex_medial = mr_angel(project_dex_medial, neg_x2)
    
    last_figure_4 = temp_name(np.rot90(np.rot90(np.rot90(neg_x2))), fixed_slice_dex_medial)
    
    # # Now, you can use loaded_figure as you would use last_figure_4
    
    
    # # print(type(last_figure))
    # # plt.imshow(np.rot90(np.rot90(np.rot90(neg_x2))))
    # # plt.axis('off')
    # # plt.colorbar()
    # # plt.show()
    
    project_sin_medial = np.load(get_file_path('projection_270.npy'))
    x = 1
    project_sin_medial = shift_left_zero_fill(project_sin_medial, x)
    fixed_slice_sin_medial=mr_angel(project_sin_medial, pos_x2)
    
    last_figure_3 = temp_name(np.flip(np.rot90(np.rot90(np.rot90(pos_x2))),axis=1), fixed_slice_sin_medial)
    
    # # plt.imshow(np.flip(np.rot90(np.rot90(np.rot90(pos_x2))),axis=1))
    # # plt.axis('off')
    # # plt.colorbar()
    # # plt.show()
    
    project_dex = np.load(get_file_path('projection_0.npy'))
    x = 2
    project_dex = shift_right_zero_fill(project_dex, x)
    fixed_slice_dex=mr_angel(project_dex, neg_y2)
    
    last_figure_6 = temp_name(np.rot90(np.rot90(np.rot90(neg_y2))), fixed_slice_dex)
    # # plt.imshow(np.rot90(np.rot90(np.rot90(neg_y2))))
    # # plt.axis('off')
    # # plt.colorbar()
    # # plt.show()
    
    project_sin = np.load(get_file_path('projection_180.npy'))
    x = 2
    project_sin = shift_right_zero_fill(project_sin, x)
    fixed_slice_sin=mr_angel(project_sin, pos_y2)
    
    last_figure_5 = temp_name(np.rot90(np.rot90(np.rot90(pos_y2))), fixed_slice_sin)
    # # plt.imshow(np.rot90(np.rot90(np.rot90(pos_y2))))
    # # plt.axis('off')
    # # plt.colorbar()
    # # plt.show()
    
    project_back = np.load(get_file_path('projection_inside_sin.npy'))
    x = 5
    project_back = shift_right_zero_fill(project_back, x)
    fixed_slice_back=mr_angel(project_back, neg_x0)
    
    last_figure_1 = temp_name(np.rot90(np.rot90(np.rot90(neg_x0))), fixed_slice_back)
    # # plt.imshow(np.rot90(np.rot90(np.rot90(neg_x0))))
    # # plt.axis('off')
    # # plt.colorbar()
    # # plt.show()
    
    project_front = np.load(get_file_path('projection_inside_dex.npy'))
    x = 4
    project_front = shift_left_zero_fill(project_front, x)
    fixed_slice_front=mr_angel(project_front, pos_x1)
    
    last_figure_2 = temp_name(np.flip(np.rot90(np.rot90(np.rot90(pos_x1))), axis=1), fixed_slice_front)
    # # plt.imshow(np.flip(np.rot90(np.rot90(np.rot90(pos_x1))), axis=1))
    # # plt.axis('off')
    # # plt.colorbar()
    # # plt.show()
    return last_figure_1, last_figure_2, last_figure_3, last_figure_4, last_figure_5, last_figure_6