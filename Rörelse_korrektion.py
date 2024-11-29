# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:04:20 2024

@author: jacke
"""
import ants
import numpy as np
import time

def rörelse_korrektion(data_4d):
    def motion_correct_4d_array(data, reference_time_point=10):
        """
        Perform motion correction on a 4D numpy array (x, y, z, time)
        to align each 3D volume to the volume at the specified time point.
        
        Parameters:
        - data: A 4D numpy array of shape (x, y, z, time).
        - reference_time_point: The time point of the reference volume.
        
        Returns:
        - A 4D numpy array with motion-corrected volumes.
        """
        # Ensure the reference time point is within the bounds of the data's time dimension
        if reference_time_point < 0 or reference_time_point >= data.shape[3]:
            raise ValueError("Reference time point is out of bounds.")
        
        # Convert the reference volume to an ANTs image
        reference_volume = ants.from_numpy(data[..., reference_time_point])
        
        # Initialize the array to hold the motion-corrected volumes
        corrected_data = np.zeros_like(data)
        
        # # Process each time point
        # for t in range(data.shape[3]):
        #     # print(t)
        #     # Skip the reference volume since it doesn't need registration
        #     if t == reference_time_point:
        #         corrected_data[..., t] = data[..., t]
        #         continue
        #     #skippar första volymen för den kan vara noll
        #     if t == 0:
        #         corrected_data[..., t] = data[..., t]
        #         continue
            
        #     # Convert the current volume to an ANTs image
        #     moving_volume = ants.from_numpy(data[..., t])
        #     random_seed = 42
        #     np.random.seed(random_seed)
        #     seed=2
        #     # Perform registration (align the moving volume to the reference volume)
        #     registration = ants.registration(fixed=reference_volume, moving=moving_volume, type_of_transform='QuickRigid', random_seed=seed)
            

        #     # Apply the transformation to the moving volume and convert back to numpy array
        #     corrected_volume = ants.apply_transforms(fixed=reference_volume, moving=moving_volume, transformlist=registration['fwdtransforms'], interpolator='linear')
            
        #     # Store the corrected volume
        #     corrected_data[..., t] = corrected_volume.numpy()

        
        # Open a text file to store motion parameters

        
        # Open a text file to store motion parameters

        
        # Open a text file to store motion parameters
       # Initialize an empty list to store motion parameters for each time point
        motion_parameters = []
        
        for t in range(data.shape[3]):
            print('summan:', np.sum(data[..., t]))
            
            # Skip the first volume if it may be zero
            if np.sum(data[..., t]) <= 500000 or t == reference_time_point:
                corrected_data[..., t] = data[..., t]
                noll = 0.000
                motion_parameters.append([t, noll, noll, noll])
                continue
        
            # Convert the current volume to an ANTs image
            moving_volume = ants.from_numpy(data[..., t])
            random_seed = 42
            np.random.seed(random_seed)
            seed = 2
        
            # Perform registration
            registration = ants.registration(
                fixed=reference_volume,
                moving=moving_volume,
                type_of_transform='QuickRigid',
                reg_iterations=[10, 5, 2],
                random_seed=seed
            )
        
            # Apply the transformation to the moving volume
            corrected_volume = ants.apply_transforms(
                fixed=reference_volume,
                moving=moving_volume,
                transformlist=registration['fwdtransforms'],
                interpolator='linear'
            )
        
            # Store the corrected volume
            corrected_data[..., t] = corrected_volume.numpy()
        
            # Voxel dimensions (pixel spacing and slice thickness)
            voxel_spacing = np.array([1.9, 1.9, 2.8])  # x, y, z dimensions in mm
            
            # Extract the affine matrix from the transformation
            affine_transform = ants.read_transform(registration['fwdtransforms'][0])
            affine_matrix = np.array(affine_transform.parameters).reshape(3, 4)
            affine_matrix = np.vstack([affine_matrix, [0, 0, 0, 1]])  # Convert to 4x4 matrix
            
            # Scale the affine matrix by voxel dimensions
            # Create a diagonal scaling matrix to convert voxel space to physical space
            voxel_scaling_matrix = np.diag(np.append(voxel_spacing, 1.0))  # Add 1 for homogeneous coordinates
            physical_affine_matrix = affine_matrix @ voxel_scaling_matrix  # Adjust both rotation and translation
            
            # Extract the translation vector
            scaled_translation = physical_affine_matrix[:3, 3]  # Tx, Ty, Tz in millimeters

            
            # Append results for both translation and rotation to the array
            motion_parameters.append([
                t,
                scaled_translation[0], scaled_translation[1], scaled_translation[2],  # Tx, Ty, Tz
            ])
        
        # Convert the list to a numpy array for easier manipulation
        motion_parameters_array = np.array(motion_parameters)
        # np.save('motion_parameters_array.npy', motion_parameters_array)
        
        return corrected_data, motion_parameters_array
    
    # Example usage:
    # Assuming `data` is your 4D numpy array with shape (x, y, z, time)
    start=time.time()
    corrected_data, motion_parameters_array = motion_correct_4d_array(data_4d)
    print("rörelse korrektion tid:", int(time.time()-start))
    return corrected_data, motion_parameters_array