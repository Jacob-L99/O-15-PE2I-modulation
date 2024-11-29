# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:25:14 2024

@author: jacke
"""

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

def z_score_BP_R_I(R_I_data, BP_data):
    # Load the mean and standard deviation arrays from the saved .npy file
    loaded_arrays = np.load(get_file_path('R_I_mean_and_std.npy'), allow_pickle=True)
    
    # Access the mean and standard deviation arrays
    mean_array = loaded_arrays[0]
    std_array = loaded_arrays[1]
    
    
    
    
    # Create a mask for non-zero elements in the BP data
    mask = R_I_data >=0.1
    
    # Calculate the Z-score only for non-zero elements
    z_score_R_I = np.zeros_like(R_I_data, dtype=np.float64)
    z_score_R_I[~mask] = np.nan
    z_score_R_I[mask] = (R_I_data[mask] - mean_array[mask]) / std_array[mask]
    
    
    
    from parkinson_plot import plot_specific_slices
    # plot_specific_slices(R_I_data, axis=2, slice_indices=[60, 80, 100], vmin=0, vmax=2.5, title="R_I mean")
    # # plot_specific_slices(mean_array, axis=2, slice_indices=[60, 80, 100], vmin=0, vmax=2.5, title="R_I mean")
    # # plot_specific_slices(std_array, axis=2, slice_indices=[60, 80, 100], vmin=0, vmax=2, title="R_I STD")
    plot_specific_slices(z_score_R_I, axis=2, slice_indices=[60, 80, 100], vmin=-5, vmax=5, title="R_I STD")
    
    # # Load the mean and standard deviation arrays from the saved .npy file
    # loaded_arrays = np.load('BP_mean_and_std.npy', allow_pickle=True)
    
    # # Access the mean and standard deviation arrays
    # mean_array = loaded_arrays[0]
    # std_array = loaded_arrays[1]
    
    
    
    
    # # Create a mask for non-zero elements in the BP data
    # mask = BP_data >= 0.75
    
    # # Calculate the Z-score only for non-zero elements
    # z_score_BP = np.zeros_like(BP_data, dtype=np.float64)
    # z_score_BP[~mask] = np.nan
    # z_score_BP[mask] = (BP_data[mask] - mean_array[mask]) / std_array[mask]
    
    
    
    # # from parkinson_plot import plot_specific_slices
    # # plot_specific_slices(BP_data, axis=2, slice_indices=[60, 80, 100], vmin=0, vmax=7.5, title="BP mean")
    # # # plot_specific_slices(mean_array, axis=2, slice_indices=[60, 80, 100], vmin=0, vmax=7.5, title="BP mean")
    # # # plot_specific_slices(std_array, axis=2, slice_indices=[60, 80, 100], vmin=0, vmax=2, title="BP STD")
    # plot_specific_slices(z_score_BP, axis=2, slice_indices=[60, 80, 100], vmin=-5, vmax=5, title="BP STD")
    return z_score_R_I

# R_I_data=np.load('FE-PE2I-01-R_I.npy')
# BP_data=np.load('FE-PE2I-01-BP.npy')
# z_score_BP_R_I(R_I_data, BP_data)