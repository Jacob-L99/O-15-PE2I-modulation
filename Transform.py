# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:11:40 2024

@author: jacke
"""
import ants
import numpy as np


import ants
import numpy as np

def transform(K_1_reshape_list, K_2_reshape_list, V_a_reshape_list, registration_1, registration_2, template_3d):
    print(np.shape(np.array(K_1_reshape_list)))
    try:
        K_1 = np.transpose(np.array(K_1_reshape_list), (1,2,0))
        img_3d_K_1 = ants.from_numpy(K_1)
        transformed_K_1_1 = ants.apply_transforms(fixed=template_3d, moving=img_3d_K_1, transformlist=registration_1['fwdtransforms'])
        transformed_K_1 = ants.apply_transforms(fixed=template_3d, moving=transformed_K_1_1, transformlist=registration_2['fwdtransforms']).numpy()
    except Exception as e:
        print(f"Error transforming K_1: {e}")
        transformed_K_1 = None  # You can choose to return None or some other value if transformation fails
    
    try:
        K_2 = np.transpose(np.array(K_2_reshape_list), (1,2,0))
        img_3d_K_2 = ants.from_numpy(K_2)
        transformed_K_2_1 = ants.apply_transforms(fixed=template_3d, moving=img_3d_K_2, transformlist=registration_1['fwdtransforms'])
        transformed_K_2 = ants.apply_transforms(fixed=template_3d, moving=transformed_K_2_1, transformlist=registration_2['fwdtransforms']).numpy()
    except Exception as e:
        print(f"Error transforming K_2: {e}")
        transformed_K_2 = None  # Handle failure for K_2 similarly
    
    try:
        V_a = np.transpose(np.array(V_a_reshape_list), (1,2,0))
        img_3d_V_a = ants.from_numpy(V_a)
        transformed_V_a_1 = ants.apply_transforms(fixed=template_3d, moving=img_3d_V_a, transformlist=registration_1['fwdtransforms'])
        transformed_V_a = ants.apply_transforms(fixed=template_3d, moving=transformed_V_a_1, transformlist=registration_2['fwdtransforms']).numpy()
    except Exception as e:
        print(f"Error transforming V_a: {e}")
        transformed_V_a = None  # Handle failure for V_a similarly

    return transformed_K_1, transformed_K_2, transformed_V_a
