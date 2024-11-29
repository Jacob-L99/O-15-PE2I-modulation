# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:39:56 2024

@author: jacke
"""

import ants
import numpy as np

def transform_pat_space(K_1_reshape_list, K_2_reshape_list, registration_1,  template_3d):
    print(np.shape(np.array(K_1_reshape_list)))
    try:
        K_1 = np.rot90(np.transpose(np.array(K_1_reshape_list), (1,2,0)), axes=(0, 1))
        img_3d_K_1 = ants.from_numpy(K_1)
        transformed_K_1 = ants.apply_transforms(fixed=template_3d, moving=img_3d_K_1, transformlist=registration_1['fwdtransforms']).numpy()
    except Exception as e:
        print(f"Error transforming K_1: {e}")
        transformed_K_1 = None  # You can choose to return None or some other value if transformation fails
    
    try:
        K_2 = np.rot90(np.transpose(np.array(K_2_reshape_list), (1,2,0)), axes=(0, 1))
        img_3d_K_2 = ants.from_numpy(K_2)
        transformed_K_2 = ants.apply_transforms(fixed=template_3d, moving=img_3d_K_2, transformlist=registration_1['fwdtransforms']).numpy()
    except Exception as e:
        print(f"Error transforming K_2: {e}")
        transformed_K_2 = None  # Handle failure for K_2 similarly
    


    return transformed_K_1, transformed_K_2