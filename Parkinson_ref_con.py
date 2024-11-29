# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:13:56 2024

@author: jacke
"""
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

import nibabel as nib


def nii_gz_to_numpy(file_path):
    """
    Loads a .nii.gz file and converts it to a NumPy array.

    Parameters:
    - file_path (str): The path to the .nii.gz file.

    Returns:
    - data (np.ndarray): The image data as a NumPy array.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        # Load the NIfTI file
        nii_img = nib.load(file_path)
    except Exception as e:
        raise IOError(f"An error occurred while loading the NIfTI file: {e}")

    # Get the image data as a NumPy array
    data = nii_img.get_fdata()

    return data

# Define the list of mask names in the order they appear in the NIfTI file
mask_names = [
    "R_Cingulate_Ant",
    "L_Cingulate_Ant",
    "R_Cingulate_Post",
    "L_Cingulate_Post",
    "R_Insula",
    "L_Insula",
    "R_Brainstem",
    "L_Brainstem",
    "R_Thalamus",
    "L_Thalamus",
    "R_Caudate",
    "L_Caudate",
    "R_Putamen",
    "L_Putamen",
    "R_Pallidum",
    "L_Pallidum",
    "R_Substantia_nigra",
    "L_Substantia_nigra",
    "R_Frontal_Lat",
    "L_Frontal_Lat",
    "R_Orbital",
    "L_Orbital",
    "R_Frontal_Med_Sup",
    "L_Frontal_Med_Sup",
    "R_Precentral",
    "L_Precentral",
    "R_Parietal_Inf",
    "L_Parietal_Inf",
    "R_Postcentral",
    "L_Postcentral",
    "R_Precuneus",
    "L_Precuneus",
    "R_Parietal_Sup",
    "L_Parietal_Sup",
    "R_Temporal_Mesial",
    "L_Temporal_Mesial",
    "R_Temporal_Basal",
    "L_Temporal_Basal",
    "R_Temporal_Lat_Ant",
    "L_Temporal_Lat_Ant",
    "R_Occipital_Med",
    "L_Occipital_Med",
    "R_Occipital_Lat",
    "L_Occipital_Lat",
    "R_Cerebellum",
    "L_Cerebellum",
    "R_Vermis",
    "L_Vermis"
]

# Path to the NIfTI file containing all masks
nifti_file_path = get_file_path(f'brain_masks.nii')  # Update this path if necessary


# Load the NIfTI file using nibabel
try:
    nifti_data = nii_gz_to_numpy(nifti_file_path)
    print(f"NIfTI file '{nifti_file_path}' loaded successfully.")
except FileNotFoundError as fnf_error:
    print(fnf_error)
    nifti_data = None
except IOError as io_error:
    print(io_error)
    nifti_data = None

def ref_con(motion_corrected_data):
    
    motion_corrected_data=np.array(motion_corrected_data)

    nifti_data_dex = nifti_data.copy()
    nifti_data_sin = nifti_data.copy()
    nifti_data_dex[nifti_data<0.5]=np.nan
    nifti_data_sin[nifti_data<0.5]=np.nan
    
    Cerebellum_mean_dex_ref=np.nanmean(nifti_data_dex[..., 44]*motion_corrected_data)
    Cerebellum_mean_sin_ref=np.nanmean(nifti_data_sin[..., 45]*motion_corrected_data)
    Cerebellum_mean=(Cerebellum_mean_dex_ref+Cerebellum_mean_sin_ref)/2
    

    Ref_concentration=motion_corrected_data*(Cerebellum_mean)
    
    Ref_TAC = []
    
    # Loop over the first dimension (t)
    for frame in Ref_concentration:
        # Get all non-zero values in the frame
        non_zero_values = frame[frame != 0]
        
        # Calculate the mean of non-zero values
        if non_zero_values.size > 0:
            mean_value = np.nanmean(non_zero_values)
        else:
            mean_value = 0  # Handle case where there are no non-zero values
        
        # Store the mean value
        Ref_TAC.append(mean_value)
    
    Ref_TAC = np.array(Ref_TAC)

    
    return Ref_TAC