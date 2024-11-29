# # -*- coding: utf-8 -*- 
# """
# Created on Thu Sep 19 09:44:37 2024

# @author: jacke
# """
# import numpy as np
# import os
# import pydicom
# import time


# def läsa_in(dicom_name, inf_name):
#     def apply_dicom_slope_intercept(pixel_array, slope, intercept):
#         """Apply DICOM slope and intercept to pixel values."""
#         return (pixel_array * slope + intercept) / 1000

#     def load_and_sort_dicom_images(directory, frames):
#         global first_image_shape, pixel_spacing, slice_thickness, image_positions
#         # List all DICOM files in the directory
#         dicom_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]

#         # Initialize metadata variables
#         pixel_spacing = None
#         slice_thickness = None
#         image_positions = []

#         # Read each file and extract slice number, acquisition time, slope, and intercept
#         dicom_files_with_info = []
#         for filename in dicom_files:
#             file_path = os.path.join(directory, filename)
#             dicom_image = pydicom.dcmread(file_path)
#             slice_number = dicom_image.get('InstanceNumber', float('inf'))
#             acquisition_time = dicom_image.get('AcquisitionTime', 'Unknown')
#             slope = dicom_image.get('RescaleSlope', 1)
#             intercept = dicom_image.get('RescaleIntercept', 0)

#             # Extract metadata from the first DICOM file
#             if pixel_spacing is None:
#                 pixel_spacing = dicom_image.get('PixelSpacing', [1.0, 1.0])  # Default to 1.0 mm if not found
#             else:
#                 # Verify consistency across all files
#                 current_pixel_spacing = dicom_image.get('PixelSpacing', [1.0, 1.0])
#                 if current_pixel_spacing != pixel_spacing:
#                     print(f"Warning: Inconsistent PixelSpacing in file {filename}")

#             if slice_thickness is None:
#                 slice_thickness = dicom_image.get('SliceThickness', 1.0)  # Default to 1.0 mm if not found
#             else:
#                 # Verify consistency across all files
#                 current_slice_thickness = dicom_image.get('SliceThickness', 1.0)
#                 if current_slice_thickness != slice_thickness:
#                     print(f"Warning: Inconsistent SliceThickness in file {filename}")

#             # Extract Image Position Patient
#             image_position = dicom_image.get('ImagePositionPatient', [0.0, 0.0, 0.0])
#             image_positions.append(image_position)

#             dicom_files_with_info.append(
#                 (file_path, slice_number, acquisition_time, slope, intercept))

#         # Sort the list by slice number
#         dicom_files_sorted = sorted(dicom_files_with_info, key=lambda x: x[1])

#         images = []
#         for i in range(len(dicom_files_sorted)):
#             file_path, slice_number, acquisition_time, slope, intercept = dicom_files_sorted[i]
#             dicom_image = pydicom.dcmread(file_path)
#             pixel_array = apply_dicom_slope_intercept(dicom_image.pixel_array, slope, intercept)
#             images.append(pixel_array)

#         # Assuming all images have the same dimensions, extract the shape from the first image
#         first_image_shape = images[0].shape

#         slices = int(len(dicom_files_sorted) / frames)
#         # Reshape according to the read dimensions and the number of frames and slices
#         data_4d = np.array(images).reshape(frames, slices, *first_image_shape)
#         # Transpose the array to fit the desired shape (y, x, slices, time)
#         data_4d_reshaped = np.transpose(data_4d, (3, 2, 1, 0))
#         return data_4d_reshaped, pixel_spacing, slice_thickness, image_positions

#     # Read AIF curve and times from .inp file
#     start_tid_0 = time.time()
#     inp_file_path = inf_name

#     # Initialize lists to store the data from each column
#     AIF_time = []
#     AIF = []

#     # Open and read the .inp file
#     with open(inp_file_path, 'r') as file:
#         for line in file:
#             # Split each line into parts based on whitespace
#             parts = line.strip().split()
#             if len(parts) >= 2:
#                 # Append data to respective lists
#                 AIF_time.append(float(parts[0]))
#                 AIF.append(float(parts[1]))

#     AIF_time = np.array(AIF_time)
#     AIF = np.array(AIF)
#     frames = len(AIF_time)

#     dicom_directory = dicom_name
#     data_4d, pixel_spacing, slice_thickness, image_positions = load_and_sort_dicom_images(dicom_directory, frames)
#     slut_tid_0 = time.time()
#     sort_tid = slut_tid_0 - start_tid_0
#     print("läsa in och sortera tid:", int(sort_tid), "s")

#     return data_4d, AIF, AIF_time, first_image_shape, pixel_spacing, slice_thickness, image_positions


# -*- coding: utf-8 -*- 
"""
Created on Thu Sep 19 09:44:37 2024

@author: jacke
"""
import numpy as np
import os
import pydicom
import time
import nibabel as nib  # New import for NIfTI handling

def läsa_in(dicom_name, inf_name):
    def apply_dicom_slope_intercept(pixel_array, slope, intercept):
        """Apply DICOM slope and intercept to pixel values."""
        return (pixel_array * slope + intercept) / 1000

    def load_and_sort_dicom_images(directory, frames):
        global first_image_shape, pixel_spacing, slice_thickness, image_positions
        # List all DICOM files in the directory
        dicom_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]

        if not dicom_files:
            raise ValueError(f"No DICOM files found in directory: {directory}")

        # Initialize metadata variables
        pixel_spacing = None
        slice_thickness = None
        image_positions = []

        # Read each file and extract slice number, acquisition time, slope, and intercept
        dicom_files_with_info = []
        for filename in dicom_files:
            file_path = os.path.join(directory, filename)
            dicom_image = pydicom.dcmread(file_path)
            slice_number = dicom_image.get('InstanceNumber', float('inf'))
            acquisition_time = dicom_image.get('AcquisitionTime', 'Unknown')
            slope = dicom_image.get('RescaleSlope', 1)
            intercept = dicom_image.get('RescaleIntercept', 0)

            # Extract metadata from the first DICOM file
            if pixel_spacing is None:
                pixel_spacing = dicom_image.get('PixelSpacing', [1.0, 1.0])  # Default to 1.0 mm if not found
            else:
                # Verify consistency across all files
                current_pixel_spacing = dicom_image.get('PixelSpacing', [1.0, 1.0])
                if current_pixel_spacing != pixel_spacing:
                    print(f"Warning: Inconsistent PixelSpacing in file {filename}")

            if slice_thickness is None:
                slice_thickness = dicom_image.get('SliceThickness', 1.0)  # Default to 1.0 mm if not found
            else:
                # Verify consistency across all files
                current_slice_thickness = dicom_image.get('SliceThickness', 1.0)
                if current_slice_thickness != slice_thickness:
                    print(f"Warning: Inconsistent SliceThickness in file {filename}")

            # Extract Image Position Patient
            image_position = dicom_image.get('ImagePositionPatient', [0.0, 0.0, 0.0])
            image_positions.append(image_position)

            dicom_files_with_info.append(
                (file_path, slice_number, acquisition_time, slope, intercept))

        # Sort the list by slice number
        dicom_files_sorted = sorted(dicom_files_with_info, key=lambda x: x[1])

        images = []
        for i in range(len(dicom_files_sorted)):
            file_path, slice_number, acquisition_time, slope, intercept = dicom_files_sorted[i]
            dicom_image = pydicom.dcmread(file_path)
            pixel_array = apply_dicom_slope_intercept(dicom_image.pixel_array, slope, intercept)
            images.append(pixel_array)

        # Assuming all images have the same dimensions, extract the shape from the first image
        first_image_shape = images[0].shape

        slices = int(len(dicom_files_sorted) / frames)
        if slices == 0:
            raise ValueError("Number of slices is zero. Check the 'frames' parameter and DICOM files.")

        # Reshape according to the read dimensions and the number of frames and slices
        data_4d = np.array(images).reshape(frames, slices, *first_image_shape)
        # Transpose the array to fit the desired shape (y, x, slices, time)
        data_4d_reshaped = np.transpose(data_4d, (3, 2, 1, 0))
        return data_4d_reshaped, pixel_spacing, slice_thickness, image_positions

    def load_nifti_file(nifti_path, frames):
        global first_image_shape, pixel_spacing, slice_thickness, image_positions

        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()

        # Assuming the NIfTI data is 4D: (x, y, z, time)
        if data.ndim != 4:
            raise ValueError(f"NIfTI file must be 4D. Provided file has {data.ndim} dimensions.")

        first_image_shape = data.shape[:3]
        first_image_shape = first_image_shape[::-1]  # Reverse to (y, x, z)

        # Extract voxel dimensions from the affine
        affine = nifti_img.affine
        voxel_sizes = nib.affines.voxel_sizes(affine)
        pixel_spacing = voxel_sizes[:2]  # (x, y)
        slice_thickness = voxel_sizes[2]  # z

        # Placeholder for image_positions since NIfTI might not have this information
        image_positions = [affine[:3, 3]] * data.shape[2]

        # Transpose data to match (y, x, slices, time)
        data_4d_reshaped = np.transpose(data, (0, 1, 2, 3))  # (y, x, z, time)
        data_4d_reshaped = np.rot90(np.rot90(data_4d_reshaped, axes=(1,2)), axes=(1,2))

        return data_4d_reshaped, pixel_spacing, slice_thickness, image_positions

    # Read AIF curve and times from .inp file
    start_tid_0 = time.time()
    inp_file_path = inf_name

    # Initialize lists to store the data from each column
    AIF_time = []
    AIF = []

    # Open and read the .inp file
    with open(inp_file_path, 'r') as file:
        for line in file:
            # Split each line into parts based on whitespace
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # Append data to respective lists
                    AIF_time.append(float(parts[0]))
                    AIF.append(float(parts[1]))
                except ValueError:
                    # Skip lines that cannot be converted to float
                    continue

    AIF_time = np.array(AIF_time)
    AIF = np.array(AIF)
    frames = len(AIF_time)

    # Determine if dicom_name is a directory or a NIfTI file
    if os.path.isdir(dicom_name):
        dicom_directory = dicom_name
        data_4d, pixel_spacing, slice_thickness, image_positions = load_and_sort_dicom_images(dicom_directory, frames)
    elif os.path.isfile(dicom_name):
        _, file_extension = os.path.splitext(dicom_name)
        if file_extension.lower() in ['.nii', '.nii.gz']:
            data_4d, pixel_spacing, slice_thickness, image_positions = load_nifti_file(dicom_name, frames)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions are directories of DICOM files or NIfTI files (.nii, .nii.gz).")
    else:
        raise ValueError(f"dicom_name must be a directory or a NIfTI file. Provided: {dicom_name}")

    slut_tid_0 = time.time()
    sort_tid = slut_tid_0 - start_tid_0
    print("läsa in och sortera tid:", int(sort_tid), "s")

    return data_4d, AIF, AIF_time, first_image_shape, pixel_spacing, slice_thickness, image_positions
