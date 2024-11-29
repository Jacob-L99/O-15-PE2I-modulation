# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:51:32 2024

@author: jacke
"""

import numpy as np
from scipy.ndimage import zoom
import time

def Downsample(data_4d, first_image_shape):
    downsample_factors = (128/first_image_shape[0], 128/first_image_shape[0], 1, 1)  # Keep time dimension unchanged

    if 128/first_image_shape[0]!=1:
        
        def downsample_4d_image_adjust_z(image_4d, factors):
            """
            Downsample a 4D image in its spatial dimensions (x, y, z) while leaving the time dimension unchanged.
            Adjusts the downsampled 'z' dimension size by +1 if the original 'z' size was odd.
            
            :param image_4d: The 4D numpy array to downsample.
            :param factors: A tuple of downsampling factors for each dimension (x, y, z, time).
            :return: The downsampled 4D image with adjusted 'z' dimension if necessary.
            """
            # Calculate the expected downsampled sizes
            downsampled_size_x = int(image_4d.shape[0] * factors[0])
            downsampled_size_y = int(image_4d.shape[1] * factors[1])
            downsampled_size_z = int(image_4d.shape[2] * factors[2])
    
            # Adjust downsampled_size_z if original z size is odd
            if image_4d.shape[2] % 2 != 0:
                downsampled_size_z += 1
            
            # Initialize the downsampled image array
            downsampled_image_4d = np.zeros((downsampled_size_x, downsampled_size_y, downsampled_size_z, image_4d.shape[3]),
                                            dtype=image_4d.dtype)
    
            for t in range(image_4d.shape[3]):
                # Downsample the current 3D volume
                downsampled_volume = zoom(image_4d[:, :, :, t], factors[:-1], order=3)  # Use cubic interpolation
                
                # If z dimension was adjusted, handle potential size mismatch
                if downsampled_volume.shape[2] != downsampled_size_z:
                    # Initialize adjusted volume with the new z size
                    adjusted_volume = np.zeros((downsampled_size_x, downsampled_size_y, downsampled_size_z),
                                               dtype=downsampled_volume.dtype)
                    # Copy the downsampled volume into the adjusted volume
                    adjusted_volume[:, :, :downsampled_volume.shape[2]] = downsampled_volume
                    downsampled_volume = adjusted_volume
                
                downsampled_image_4d[:, :, :, t] = downsampled_volume
    
            return downsampled_image_4d
        
        # Downsample your 4D image with adjustment for odd z dimension
        st=time.time()
    
        image_4d_downsampled = downsample_4d_image_adjust_z(data_4d, downsample_factors)
    
        data_4d=image_4d_downsampled
        print("down sample:", time.time()-st)
        return data_4d
    else:
        return data_4d