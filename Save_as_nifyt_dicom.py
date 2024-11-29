# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:25:24 2024

@author: jacke
"""

import numpy as np
import nibabel as nib
import os

def construct_affine(pixel_spacing, slice_thickness, coordinate_position):
    """
    Constructs an affine transformation matrix for NIfTI images.

    Parameters:
    - pixel_spacing (tuple, list, or np.ndarray of float): The (x, y) pixel spacing in millimeters.
    - slice_thickness (float): The thickness of each slice in millimeters.
    - coordinate_position (tuple, list, or np.ndarray of float): The (x, y, z) coordinate position (origin) in millimeters.

    Returns:
    - np.ndarray: A 4x4 affine transformation matrix.
    """
    x_spacing, y_spacing = pixel_spacing
    z_spacing = slice_thickness
    x_origin, y_origin, z_origin = coordinate_position[0], coordinate_position[1], coordinate_position[2]

    # Affine matrix with no rotation (identity orientation)
    affine = np.array([
        [x_spacing,        0,        0, x_origin],
        [       0, y_spacing,        0, y_origin],
        [       0,        0, z_spacing, z_origin],
        [       0,        0,        0,        1]
    ])

    return affine



def save_as_nifti(directory ,arrays, filenames, arrays_pat, filenames_pat, pixel_spacing, slice_thickness, image_positions):
    """
    Save multiple 3D NumPy arrays as separate NIfTI files.

    Parameters:
    - arrays (list or tuple of np.ndarray): 3D NumPy arrays to be saved.
    - filenames (list or tuple of str): Output filenames or full paths for the NIfTI files.
    - affine (np.ndarray): 4x4 affine transformation matrix. Defaults to identity matrix.

    Raises:
    - TypeError: If any input is not a NumPy array or filenames are not strings.
    - ValueError: If any input array is not 3D or if the number of arrays and filenames do not match.
    - OSError: If the directory for a filename does not exist and cannot be created.
    """
    if not isinstance(arrays, (list, tuple)):
        raise TypeError("arrays must be a list or tuple of NumPy arrays.")
    if not isinstance(filenames, (list, tuple)):
        raise TypeError("filenames must be a list or tuple of strings.")
    if len(arrays) != len(filenames):
        raise ValueError("The number of arrays and filenames must be the same.")
    
    for idx, (array, fname) in enumerate(zip(arrays, filenames), start=1):
        # Validate that the input is a NumPy array
        nib_save=os.path.basename(directory)+ '_' + fname
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input {idx} ({fname}) is not a NumPy array.")
        
        # Validate that the array is 3D
        if array.ndim != 3:
            raise ValueError(f"Input {idx} ({fname}) is not a 3D array. It has {array.ndim} dimensions.")
        
        # Ensure the directory exists

        dir_name = os.path.dirname(nib_save)
        if dir_name and not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")
            except OSError as e:
                raise OSError(f"Could not create directory {dir_name}: {e}")
        
        affine = np.array([
            [-1.421875,          0,          0,  90],
            [          0, 1.4248366,          0, -126],
            [          0,          0, 1.421875,  -72],
            [          0,          0,          0,    1]
        ])

        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(array, affine)
        
        # Save the NIfTI file
        nib.save(nifti_img, nib_save)
        print(f"Saved {nib_save} successfully.")

        """
        Saves a list of 3D NumPy arrays as NIfTI images with specified spatial metadata.
    
        Parameters:
        - arrays (list or tuple of np.ndarray): List of 3D NumPy arrays to save.
        - filenames (list or tuple of str): Corresponding filenames for the NIfTI images.
        - pixel_spacing (tuple of float): The (x, y) pixel spacing in millimeters.
        - slice_thickness (float): The thickness of each slice in millimeters.
        - coordinate_position (tuple of float): The (x, y, z) coordinate position (origin) in millimeters.
    
        Raises:
        - TypeError: If inputs are not of the expected type.
        - ValueError: If inputs have mismatched lengths or incorrect dimensions.
        - OSError: If directories cannot be created.
        """
    
    pixel_spacing=list(pixel_spacing)
    slice_thickness=float(slice_thickness)
    coordinate_position= tuple(image_positions[0])
    
    print(type(pixel_spacing))
    print(pixel_spacing)
    print(type(slice_thickness))
    print(slice_thickness)
    print(type(image_positions))
    print(coordinate_position)
    # Validate that 'arrays' is a list or tuple
    if not isinstance(arrays_pat, (list, tuple)):
        raise TypeError("arrays must be a list or tuple of NumPy arrays.")

    # Validate that 'filenames' is a list or tuple
    if not isinstance(filenames_pat, (list, tuple)):
        raise TypeError("filenames must be a list or tuple of strings.")

    # Validate that 'pixel_spacing' is a tuple or list of two floats
    if not (isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) == 2 and 
            all(isinstance(ps, (int, float)) for ps in pixel_spacing)):
        raise TypeError("pixel_spacing must be a list or tuple of two numbers (x, y).")

    # Validate that 'slice_thickness' is a float or int
    if not isinstance(slice_thickness, (int, float)):
        raise TypeError("slice_thickness must be a number (float or int).")

    # Validate that 'coordinate_position' is a tuple or list of three floats
    if not (isinstance(coordinate_position, (list, tuple)) and len(coordinate_position) == 3 and 
            all(isinstance(cp, (int, float)) for cp in coordinate_position)):
        raise TypeError("coordinate_position must be a list or tuple of three numbers (x, y, z).")

    # Check that the number of arrays and filenames match
    if len(arrays_pat) != len(filenames_pat):
        raise ValueError("The number of arrays and filenames must be the same.")

    for idx, (array, fname) in enumerate(zip(arrays_pat, filenames_pat), start=1):
        nib_save=os.path.basename(directory)+ '_' + fname
        # Validate that the input is a NumPy array
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input {idx} ({fname}) is not a NumPy array.")

        # Validate that the array is 3D
        if array.ndim != 3:
            raise ValueError(f"Input {idx} ({fname}) is not a 3D array. It has {array.ndim} dimensions.")

        # Ensure the directory exists
        dir_name = os.path.dirname(nib_save)
        if dir_name and not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(f"Created directory: {dir_name}")
            except OSError as e:
                raise OSError(f"Could not create directory {dir_name}: {e}")

        # Construct the affine matrix
        affine = construct_affine(pixel_spacing, slice_thickness, coordinate_position)

        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(array, affine)

        # Save the NIfTI file
        nib.save(nifti_img, nib_save)
        print(f"Saved {nib_save} successfully.")
    



