# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:44:06 2024

@author: jacke
"""


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
from scipy.signal import convolve

def convexvarc(t_out, k2, t, f):
    b = k2

    
    # Reshape t and f as flattened 1D arrays
    t = np.reshape(t, (np.prod(t.shape),))
    f = np.reshape(f, (np.prod(f.shape),))
    
    
    
    # Ensure time and function vectors start from zero
    if np.min(t) > 0:
        t = np.concatenate(([0], t))
        f = np.concatenate(([0], f))
    
   
    # Interpolated time vector with finer resolution
    t0 = np.arange(0, np.max(t), 0.01)
    
    # Interpolate function values at finer time points
    f_interp = interp1d(t, f, kind='nearest', fill_value='extrapolate')
    f0 = f_interp(t0)
    
    # plt.plot(t0,f_interp(t0))
    # plt.plot(t,f, ".")
    # # # plt.title('AIF')
    # plt.show()

    
    # Perform convolution with exponential decay
    exp_decay = np.exp(-b * t0)
    # print(np.exp(-b*t))
    
    # print(f)
    fout_conv = convolve(f0, exp_decay, mode='full')
    fout_conv = fout_conv[:len(f0)]  # Truncate to the original size of f0
    
    # Interpolate back to the output time points (t_out)
    fout_interp = interp1d(t0, fout_conv, kind='linear', fill_value='extrapolate')
    fout = fout_interp(t_out) / 100.0  # Divide by 100 to adjust for step size
    
    # plt.plot(t0,exp_decay)
    # plt.title('decay funktionen')
    # plt.show()
    
    # plt.plot(t0,fout_interp(t0)/sum(fout_interp(t0)))
    # plt.title('Faltningen')
    return fout

# Define or import the convexvarc function
# Make sure to define convexvarc before using it
# For example:
# def convexvarc(t1, k2, t2, Ca):
#     # Your implementation here
#     return result


def beräkningar_3(data_4d, t, Ca):
    # Get the shape of the 4D data
    x1, y1, z1, t1 = data_4d.shape
    
    # Reshape the 4D data to 2D: (t1, x1*y1*z1)
    reshaped_data_4d = data_4d.reshape(-1, t1).T  # Shape: (t1, num_voxels)
    
    # Compute the sum over time for each voxel
    voxel_sums = reshaped_data_4d.sum(axis=0)  # Shape: (num_voxels,)
    
    # Identify voxels with sum > 50
    selected_voxel_indices = np.where(voxel_sums > 100)[0]  # Indices of selected voxels
    num_selected = len(selected_voxel_indices)
    print(f"Number of selected voxels: {num_selected} out of {reshaped_data_4d.shape[1]}")
    
    # Select only the voxels that meet the criterion
    CP_selected = reshaped_data_4d[:, selected_voxel_indices]  # Shape: (t1, num_selected)
    
    # Initialize residuals (wsse) for selected voxels with a large initial value
    wsse = np.ones(num_selected) * 1e5
    
    # Initialize arrays to store parameters for selected voxels
    K_2_selected = np.zeros(num_selected)
    K_1_selected = np.zeros(num_selected)
    V_a_selected = np.zeros(num_selected)
    
    # Define the range and step for k2
    k2_values = np.arange(np.log(0.0005), np.log(2), 0.1)
    
    tid_start = time.time()
    
    # Loop over each k2 value
    for k2 in k2_values:
        k2=np.exp(k2)
        # Compute the convolution using convexvarc
        conv = convexvarc(t, k2, t, Ca)  # Shape: (n_time_points,)
        
        # Normalize the convolution for plotting
        normalized_conv = conv / np.sum(conv)
        plt.plot(t, normalized_conv, alpha=1)  # Adjust alpha for better visibility if needed
        
        # Stack conv and Ca to form the design matrix X
        # Each row corresponds to a time point, and there are two predictors: conv and Ca
        X = np.vstack((conv, Ca)).T       # Shape: (n_time_points, 2)
        
        # Perform least squares fitting for all selected voxels simultaneously
        # np.linalg.lstsq can handle multiple right-hand sides
        # The result will contain coefficients and residuals for each voxel
        results = np.linalg.lstsq(X, CP_selected, rcond=None)
        coeffs = results[0]               # Shape: (2, num_selected)
        residuals = results[1]            # Shape: (num_selected,) or (num_selected, 1)
        
        # Some numpy versions return residuals as (num_selected, 1), so flatten if necessary
        residuals = residuals.flatten()
        
        # Identify which voxels have improved residuals
        better = residuals < wsse
        
        # Update residuals where improvement is found
        wsse[better] = residuals[better]
        
        # Update parameters for the improved fits
        # coeffs[0, :] corresponds to the first predictor (conv)
        # coeffs[1, :] corresponds to the second predictor (Ca)
        # Assuming K_2 relates to k2, K_1 and V_a are derived from coefficients
        K_2_selected[better] = k2
        V_a_selected[better] = coeffs[1, better]
        
        # Avoid division by zero by ensuring (1 - V_a) is not zero
        with np.errstate(divide='ignore', invalid='ignore'):
            K_1_selected[better] = np.where(
                (1 - V_a_selected[better]) != 0,
                coeffs[0, better] / (1 - V_a_selected[better]),
                0  # Assign a default value or handle as needed
            )
        
        # Optional: If tracking second best parameters, implement logic here
        # For example, store the second smallest residuals and corresponding parameters
    
    # After the loop, plot the final normalized convolution (optional)
    plt.title('Normalized Convolutions for All k2 Values')
    plt.xlabel('Time')
    plt.ylabel('Normalized Convex Convolution')
    plt.show()
    
    print('Beräknings tid:', int(time.time() - tid_start), 's')
    
    # Reconstruct the 3D volumes for K_1, K_2, and V_a
    # Initialize empty 3D arrays with the original spatial dimensions
    K_1_volume = np.zeros((x1, y1, z1))  # Using NaN for unprocessed voxels
    K_2_volume = np.zeros((x1, y1, z1))
    V_a_volume = np.zeros((x1, y1, z1))
    
    # Map the processed parameters back to their original voxel positions
    # Compute the original voxel indices from the flattened indices
    # Assuming the flattening was done in C order (row-major)
    for idx, voxel_idx in enumerate(selected_voxel_indices):
        # Convert the flat index back to 3D indices
        x, y, z = np.unravel_index(voxel_idx, (x1, y1, z1))
        
        # Assign the computed parameters to the 3D volumes
        K_1_volume[x, y, z] = K_1_selected[idx]
        K_2_volume[x, y, z] = K_2_selected[idx]
        V_a_volume[x, y, z] = V_a_selected[idx]

    K_1_volume=np.transpose(K_1_volume, (2,0,1))
    K_2_volume=np.transpose(K_2_volume, (2,0,1))
    V_a_volume=np.transpose(V_a_volume, (2,0,1))
    
    K_1_volume[K_1_volume > 2] = 2
    K_1_volume[K_1_volume < 0] = 0
    
    K_2_volume[K_2_volume > 2] = 2
    K_2_volume[K_2_volume < 0] = 0
    
    return K_1_volume, K_2_volume, V_a_volume
    
# # # Load data
# Ca = np.load('AIF_s309.npy')          # Shape: (n_time_points,)
# t = np.load('AIF_time_s309.npy')      # Shape: (n_time_points,)
# # Cp = np.load('means_s309.npy')        # Shape: (n_time_points,)
# data_4d = np.load('data_4d_s309.npy') # Shape: (x1, y1, z1, t1)
# beräkningar_3(data_4d, t, Ca)