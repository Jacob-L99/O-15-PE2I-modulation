# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:12:40 2024

@author: jacke
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
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

from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1 ,1 ,1), (1, 0, 0), (0.5, 0, 0)]  # R, G, B
n_bins = 10  # Number of bins in the colormap

# Create the colormap
cmap_name = 'custom_blue_white_red'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def SSP_Z(first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2, frame, wat, normal_data):
    # Load the data
    if frame=="real" and wat=='wat':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1.npy'))
    elif frame=='ref' and wat=='wat':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1_ref.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1_ref.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1_ref.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1_ref.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1_ref.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1_ref.npy'))
    elif frame=='real' and wat=='wat1':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1_1.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1_1.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1_1.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1_1.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1_1.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1_1.npy'))
    elif frame=='ref' and wat=='wat1':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1_1_ref.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1_1_ref.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1_1_ref.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1_1_ref.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1_1_ref.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1_1_ref.npy'))
    elif frame=='real' and wat=='wat2':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1_2.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1_2.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1_2.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1_2.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1_2.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1_2.npy'))
    elif frame=='ref' and wat=='wat2':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1_2_ref.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1_2_ref.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1_2_ref.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1_2_ref.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1_2_ref.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1_2_ref.npy'))
    elif frame=='real' and wat=='wat1_2':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1_3.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1_3.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1_3.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1_3.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1_3.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1_3.npy'))
    elif frame=='ref' and wat=='wat1_2':
        mean_and_std_first_values_yz_neg_x0_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x0_k1_3_ref.npy'))
        mean_and_std_first_values_yz_pos_x1_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x1_k1_3_ref.npy'))
        mean_and_std_first_values_yz_pos_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_pos_x2_k1_3_ref.npy'))
        mean_and_std_first_values_yz_neg_x2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_yz_neg_x2_k1_3_ref.npy'))
        mean_and_std_first_values_xz_pos_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_pos_y2_k1_3_ref.npy'))
        mean_and_std_first_values_xz_neg_y2_k1=np.load(os.path.join(normal_data,'mean_and_std_first_values_xz_neg_y2_k1_3_ref.npy'))
    
    
    mean_first_values_yz_neg_x0, std_first_values_yz_neg_x0=mean_and_std_first_values_yz_neg_x0_k1[0, :, :], mean_and_std_first_values_yz_neg_x0_k1[1, :, :]
    mean_first_values_yz_pos_x1, std_first_values_yz_pos_x1=mean_and_std_first_values_yz_pos_x1_k1[0, :, :], mean_and_std_first_values_yz_pos_x1_k1[1, :, :]
    mean_first_values_yz_pos_x2, std_first_values_yz_pos_x2=mean_and_std_first_values_yz_pos_x2_k1[0, :, :], mean_and_std_first_values_yz_pos_x2_k1[1, :, :]
    mean_first_values_yz_neg_x2, std_first_values_yz_neg_x2=mean_and_std_first_values_yz_neg_x2_k1[0, :, :], mean_and_std_first_values_yz_neg_x2_k1[1, :, :]
    mean_first_values_xz_pos_y2, std_first_values_xz_pos_y2=mean_and_std_first_values_xz_pos_y2_k1[0, :, :], mean_and_std_first_values_xz_pos_y2_k1[1, :, :]
    mean_first_values_xz_neg_y2, std_first_values_xz_neg_y2=mean_and_std_first_values_xz_neg_y2_k1[0, :, :], mean_and_std_first_values_xz_neg_y2_k1[1, :, :]
    
    neg_x0=(first_values_yz_neg_x0-mean_first_values_yz_neg_x0)/std_first_values_yz_neg_x0
    pos_x1=(first_values_yz_pos_x1-mean_first_values_yz_pos_x1)/std_first_values_yz_pos_x1
    pos_x2=(first_values_yz_pos_x2-mean_first_values_yz_pos_x2)/std_first_values_yz_pos_x2
    neg_x2=(first_values_yz_neg_x2-mean_first_values_yz_neg_x2)/std_first_values_yz_neg_x2
    pos_y2=(first_values_xz_pos_y2-mean_first_values_xz_pos_y2)/std_first_values_xz_pos_y2
    neg_y2=(first_values_xz_neg_y2-mean_first_values_xz_neg_y2)/std_first_values_xz_neg_y2
    
    def plot_first_values(vmin=None, vmax=None):
        fig, axes = plt.subplots(1, 6, figsize=(20, 6))
        
        # Plot for the first non-NaN value along positive x-axis
        im1 = axes[0].imshow(np.rot90(np.rot90(np.rot90(neg_x0))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Dex inside')
        axes[0].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative x-axis
        im2 = axes[1].imshow(np.flip(np.rot90(np.rot90(np.rot90(pos_x1))), axis=1), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('Sin inside)')
        axes[1].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im3 = axes[2].imshow(np.flip(np.rot90(np.rot90(np.rot90(pos_x2))), axis=1), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title('Dex')
        axes[2].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im4 = axes[3].imshow(np.rot90(np.rot90(np.rot90(neg_x2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[3].set_title('Sin')
        axes[3].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im5 = axes[4].imshow(np.rot90(np.rot90(np.rot90(pos_y2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[4].set_title('Back')
        axes[4].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im6 = axes[5].imshow(np.rot90(np.rot90(np.rot90(neg_y2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
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
    plt.imshow(pos_x2)
    return neg_x0, pos_x1, pos_x2, neg_x2, pos_y2, neg_y2