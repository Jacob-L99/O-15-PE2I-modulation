# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:06:19 2024

@author: jacke
"""
import ants
import numpy as np

def transform_park(corrected_data, registration_1, registration_2, template_3d):
    transformed_volumes = []
    
    # Ensure corrected_data has the expected shape
    if len(np.shape(corrected_data)) != 4:
        raise ValueError("corrected_data should have 4 dimensions: (x, y, z, time).")
    
    # Loop through the 3D volumes (along the 4th dimension, typically time or different scans)
    for i in range(corrected_data.shape[3]):
        # Extract the i-th 3D volume
        K_1_reshape_list = corrected_data[:, :, :, i]
        
        try:
            # Adjust dimensions to match expected order for ANTs (adjust as per your needs)
            # Typically, ANTs expects (x, y, z), so ensure you're passing correct dims.
            img_3d_K_1 = ants.from_numpy(K_1_reshape_list)
            transformed_img = ants.apply_transforms(fixed=template_3d, moving=img_3d_K_1, 
                                                    transformlist=registration_1['fwdtransforms'])
            # Apply the transformation using the provided forward transforms
            transformed_img_2 = ants.apply_transforms(fixed=template_3d, moving=transformed_img, 
                                                    transformlist=registration_2['fwdtransforms'])
            
            # Convert transformed image back to numpy and store in the list
            transformed_volumes.append(transformed_img_2.numpy())
            
        except Exception as e:
            print(f"Error transforming volume {i}: {e}")
            transformed_volumes.append(None)  # You can handle failures differently if required
    
    # Convert the list of transformed volumes back to a numpy array if none failed
    transformed_array = np.array(transformed_volumes)
    # transformed_array=np.transpose(transformed_array, (1,2,3,0))
    
    # Output the final shape of the transformed data for verification
    # print("Transformed data shape:", transformed_array.shape)
    
    return transformed_array
