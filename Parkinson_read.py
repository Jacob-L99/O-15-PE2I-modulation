# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:41:27 2024

@author: jacke
"""

import numpy as np
import os
import pydicom
import time
from datetime import datetime

def läsa_in_parkinson(dicom_name):
    def apply_dicom_slope_intercept(pixel_array, slope, intercept):
        """Apply DICOM slope and intercept to pixel values."""
        return (pixel_array * slope + intercept) / 1000
    
    def convert_time_to_minutes_midpoints(acquisition_times):
        """Convert DICOM acquisition times (HHMMSS) to minutes and return the midpoints of each frame."""
        time_in_minutes = []
        first_time = datetime.strptime(acquisition_times[0], '%H%M%S.%f') if '.' in acquisition_times[0] else datetime.strptime(acquisition_times[0], '%H%M%S')
        
        # Convert acquisition times into datetime objects
        acquisition_datetime = []
        for time_str in acquisition_times:
            current_time = datetime.strptime(time_str, '%H%M%S.%f') if '.' in time_str else datetime.strptime(time_str, '%H%M%S')
            acquisition_datetime.append(current_time)
        
        # Calculate the midpoint of each frame
        for i in range(len(acquisition_datetime)):
            if i == len(acquisition_datetime) - 1:  # Last frame
                # Use the time difference between the last two frames
                time_diff = (acquisition_datetime[i] - acquisition_datetime[i - 1]).total_seconds() / 60
            else:
                # Calculate the time difference to the next frame
                time_diff = (acquisition_datetime[i + 1] - acquisition_datetime[i]).total_seconds() / 60
            
            # Midpoint is the start time + half of the difference to the next frame
            midpoint_minutes = (acquisition_datetime[i] - first_time).total_seconds() / 60 + (time_diff / 2)
            time_in_minutes.append(midpoint_minutes)
        
        return time_in_minutes
    
    def load_and_sort_dicom_images(directory):
        global first_image_shape, acquisition_times
        # List all DICOM files in the directory
        dicom_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]

        if not dicom_files:
            raise FileNotFoundError("No DICOM files found in the specified directory.")

        # Read each file and extract slice number, acquisition time, slope, intercept, and metadata
        dicom_files_with_info = []
        acquisition_times = []
        pixel_spacing = None
        slice_thickness = None
        image_positions = []

        for filename in dicom_files:
            file_path = os.path.join(directory, filename)
            dicom_image = pydicom.dcmread(file_path)
            slice_number = dicom_image.get('InstanceNumber', float('inf'))
            acquisition_time = dicom_image.get('AcquisitionTime', 'Unknown')
            acquisition_times.append(acquisition_time)  # Store the raw acquisition time as a string
            slope = dicom_image.get('RescaleSlope', 1)
            intercept = dicom_image.get('RescaleIntercept', 0)
            dicom_files_with_info.append(
                (file_path, slice_number, acquisition_time, slope, intercept))
            
            # Extract metadata from the first DICOM file
            if pixel_spacing is None:
                pixel_spacing = dicom_image.get('PixelSpacing', [1.0, 1.0])  # Default to 1.0 mm if not found
            else:
                # Optional: Verify consistency across all files
                current_pixel_spacing = dicom_image.get('PixelSpacing', [1.0, 1.0])
                if current_pixel_spacing != pixel_spacing:
                    print(f"Warning: Inconsistent PixelSpacing in file {filename}")
            
            if slice_thickness is None:
                slice_thickness = dicom_image.get('SliceThickness', 1.0)  # Default to 1.0 mm if not found
            else:
                # Optional: Verify consistency across all files
                current_slice_thickness = dicom_image.get('SliceThickness', 1.0)
                if current_slice_thickness != slice_thickness:
                    print(f"Warning: Inconsistent SliceThickness in file {filename}")
            
            # Extract Image Position Patient
            image_position = dicom_image.get('ImagePositionPatient', [0.0, 0.0, 0.0])
            image_positions.append(image_position)

        # Sort the list by slice number
        dicom_files_sorted = sorted(dicom_files_with_info, key=lambda x: x[1])
        
        # Sort acquisition times and ensure it's unique (per frame)
        acquisition_times = sorted(list(set(acquisition_times)))
        
        # The number of frames corresponds to the unique acquisition times
        frames = len(acquisition_times)
        
        images = []
        for i in range(len(dicom_files_sorted)):
            file_path, slice_number, acquisition_time, slope, intercept = dicom_files_sorted[i]
            dicom_image = pydicom.dcmread(file_path)
            pixel_array = apply_dicom_slope_intercept(dicom_image.pixel_array, slope, intercept)
            images.append(pixel_array)

        # Assuming all images have the same dimensions, extract the shape from the first image
        first_image_shape = images[0].shape

        slices = int(len(dicom_files_sorted) / frames)
        
        # Reshape according to the read dimensions and the number of frames and slices
        data_4d = np.array(images).reshape(frames, slices, *first_image_shape)
        # Transpose the array to fit the desired shape (y, x, slices, time) or (x, y, slices, time)
        data_4d_reshaped = np.transpose(data_4d, (3, 2, 1, 0))
        
        # Convert acquisition times to minutes, calculating the midpoints of each frame
        acquisition_times_in_minutes = convert_time_to_minutes_midpoints(acquisition_times)
        
        return (data_4d_reshaped, 
                frames, 
                np.array(acquisition_times_in_minutes), 
                pixel_spacing, 
                slice_thickness, 
                image_positions)
    
    # Start timing the reading and sorting process
    start_tid_0 = time.time()

    # Load the DICOM images and extract 4D data along with metadata
    dicom_directory = dicom_name
    (data_4d, 
     frames, 
     acquisition_times_in_minutes, 
     pixel_spacing, 
     slice_thickness, 
     image_positions) = load_and_sort_dicom_images(dicom_directory)
    
    # End timing
    slut_tid_0 = time.time()
    sort_tid = slut_tid_0 - start_tid_0
    print("Läsa in och sortera tid:", int(sort_tid), "s")
    
    # Output number of frames and acquisition times in minutes for each frame
    # print(f"Antal frames: {frames}")
    # print(f"Acquisition times per frame (in minutes, midpoints): {acquisition_times_in_minutes}")
    
    return data_4d, first_image_shape, np.array(acquisition_times_in_minutes), pixel_spacing, slice_thickness, image_positions