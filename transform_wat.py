# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:09:11 2024

@author: jacke
"""

import ants
import numpy as np

def transform_wat(reg_3, data_4d):
    """
    Apply a registration transform (fwdtransforms) to every frame of a 4D numpy dataset.
    
    Parameters:
        reg_3: list
            List of forward transforms from ANTs registration.
        data_4d: np.ndarray
            The 4D numpy dataset (shape: [x, y, z, t]).
    
    Returns:
        np.ndarray
            The transformed 4D numpy dataset.
    """
    # Validate inputs
    if data_4d.ndim != 4:
        raise ValueError("The input dataset must be a 4D numpy array.")
    
    # Extract shape
    x, y, z, t = data_4d.shape

    # Prepare an array to store transformed frames
    transformed_data = np.zeros_like(data_4d)

    # Convert each 3D frame to ANTsImage, apply the transform, and store the result
    for i in range(t):
        # Extract the i-th frame
        frame = data_4d[..., i]
        
        # Convert the frame to an ANTs image
        ants_frame = ants.from_numpy(frame)
        
        # Apply the transforms
        transformed_frame = ants.apply_transforms(
            fixed=ants_frame,  # Reference frame space
            moving=ants_frame,
            transformlist=reg_3
        )
        
        # Convert back to numpy and store
        transformed_data[..., i] = transformed_frame.numpy()

    return transformed_data
