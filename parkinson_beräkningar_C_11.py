# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:29:38 2024

@author: jacke
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:21:03 2024

@author: jacke
"""


import numpy as np
from scipy import interpolate
import time
import matplotlib.pyplot as plt



def beräkning_park_C_11(Ref_concentration, Ref_concentration_time, corrected_data):
    corrected_data=np.transpose(corrected_data, (1,2,3,0))
    steps = int(100*((Ref_concentration_time[-1]+(Ref_concentration_time[-1]-Ref_concentration_time[-2])/2)-(Ref_concentration_time[0]-(Ref_concentration_time[1]-Ref_concentration_time[0])/2)))

    start_tid_3=time.time()

    print("steps =",steps)
    def perform_convolution(Ref_concentration, Ref_concentration_time, t_3, steps):
        """Perform convolution of Ref_concentration with exponential decay basis function."""
        decay_function = np.exp(-t_3 * Ref_concentration_time)
        decay_inter = interpolate.interp1d(Ref_concentration_time, decay_function, kind='nearest', fill_value="extrapolate")
        decay_list = decay_inter(np.linspace(Ref_concentration_time[0]-(Ref_concentration_time[1]-Ref_concentration_time[0])/2, Ref_concentration_time[-1]+(Ref_concentration_time[-1]-Ref_concentration_time[-2])/2, steps))
        # plt.plot(Ref_concentration_time, decay_function, "rx")
        # plt.plot(np.linspace(0,6,steps), decay_list)
        # plt.show()
    
        Ref_concentration_inter = interpolate.interp1d(Ref_concentration_time, Ref_concentration, kind='nearest', fill_value="extrapolate")
        Ref_concentration_list = Ref_concentration_inter(np.linspace(Ref_concentration_time[0]-(Ref_concentration_time[1]-Ref_concentration_time[0])/2, Ref_concentration_time[-1]+(Ref_concentration_time[-1]-Ref_concentration_time[-2])/2, steps))
        # plt.plot(Ref_concentration_time, Ref_concentration, "rx")
        # plt.plot(np.linspace(0,6,steps), Ref_concentration_list)
        # plt.show()
    
        curv = np.convolve(Ref_concentration_list, decay_list)[:len(Ref_concentration_list)] / 100
        curv_inter = interpolate.interp1d(np.linspace(Ref_concentration_time[0]-(Ref_concentration_time[1]-Ref_concentration_time[0])/2, Ref_concentration_time[-1]+(Ref_concentration_time[-1]-Ref_concentration_time[-2])/2, steps), curv, fill_value="extrapolate")
        curv_list = curv_inter(Ref_concentration_time)
    
        return curv_list
    
    curv_lists = []  # To store the convolved lists
    t_3_values = np.linspace(0.01, 0.5, 100)  # Adjust as necessary
    
    # Precompute convolutions
    import matplotlib.pyplot as plt
    for t_3 in t_3_values:
        curv_list = perform_convolution(Ref_concentration, Ref_concentration_time, t_3, steps)
        # plt.plot(Ref_concentration_time, curv_list)
        # plt.legend()
        curv_lists.append(curv_list)
    # plt.show()
    
    # lambda_F=0.000631 #min^-1

    # R_I_reshape_list=[]
    # K_2_reshape_list=[]
    # K_2_p_reshape_list=[]
    # BP_reshape_list=[]
    # for j in range(corrected_data.shape[2]):

    #     y = corrected_data[:, :, j, :].reshape(-1, corrected_data.shape[3])
    #     pixel_values_list = y.tolist() 

    #     y=np.array(pixel_values_list).transpose()

    #     x_size, y_size= corrected_data.shape[0], corrected_data.shape[1]
    
    #     k_2=np.zeros(x_size*y_size)
    #     k_2_p=np.zeros(x_size*y_size) 
    #     R_I=np.zeros(x_size*y_size)
    #     BP=np.zeros(x_size*y_size)
    #     wsse_min=np.ones(x_size*y_size)*10**5

    #     wsse_min[np.sum(y, axis=0) < 100] = 0

    #     for i, BF_i in enumerate(curv_lists):
    #         X = np.column_stack((Ref_concentration, BF_i))  # Efficiently create X
        
    #         # Solve for b using least squares instead of manual pseudo-inverse and matrix multiplication
    #         b, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
    #         # Calculate wsse if residuals are not directly provided
    #         wsse = residuals if residuals.size > 0 else np.sum((X @ b - y) ** 2)
        
    #         indices = wsse < wsse_min
        
    #         # # Update wsse_min and other variables where wsse is smaller
    #         # wsse_min[indices] = wsse[indices]
    #         # R_I[indices] = b[0][indices]   
    #         # BP[indices] = b[1][indices] / (t_3_values[i] - lambda_F) + b[0][indices] - 1
    #         # k_2[indices] = b[1][indices] + b[0][indices] * (t_3_values[i] - lambda_F) 
            
    #         # Update wsse_min and other variables where wsse is smaller
    #         wsse_min[indices] = wsse[indices]
    #         R_I[indices] = b[0][indices]   
    #         BP[indices] = (b[1][indices] / t_3_values[i]) + b[0][indices] - 1
    #         k_2[indices] = t_3_values[i]
    #         k_2_p[indices] = (b[1][indices] / b[0][indices]) + t_3_values[i] 
    
        
    #     BP_reshape=BP.reshape(corrected_data.shape[0], corrected_data.shape[1])
    #     BP_reshape_list.append(BP_reshape)
        
    #     K_2_reshape=k_2.reshape(corrected_data.shape[0], corrected_data.shape[1])
    #     K_2_reshape_list.append(K_2_reshape)
        
    #     K_2_p_reshape=k_2_p.reshape(corrected_data.shape[0], corrected_data.shape[1])
    #     K_2_p_reshape_list.append(K_2_p_reshape)
    
    #     R_I_reshape=R_I.reshape(corrected_data.shape[0], corrected_data.shape[1])
    #     R_I_reshape_list.append(R_I_reshape)


    R_I_reshape_list=[]
    K_2_reshape_list=[]
    K_2_p_reshape_list=[]
    BP_reshape_list=[]
    

    sum_over_time = corrected_data.sum(axis=3)
    valid_coords=np.ones((1,1,1))
    min_counts=100
    while len(valid_coords[0])<=300000:
        valid_coords = np.where(sum_over_time >= min_counts)
        min_counts=min_counts-10

    print(f'number of valid datapoints: {len(valid_coords[0])}')
    num_valid_coords = len(valid_coords[0])
    new_list = np.zeros((num_valid_coords, 1, 1, corrected_data.shape[3]))
    
    # Step 4: Populate the new_list array with the pixel values from corrected_data
    for idx in range(num_valid_coords):
        x, y, z = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        # For each valid (x, y, z), copy the pixel values across all time steps (t)
        new_list[idx, 0, 0, :] = corrected_data[x, y, z, :]
    



    for j in range(new_list.shape[2]):

        y = new_list[:, :, j, :].reshape(-1, new_list.shape[3])
        pixel_values_list = y.tolist() 

        y=np.array(pixel_values_list).transpose()

        x_size, y_size= new_list.shape[0], new_list.shape[1]
    
        k_2=np.zeros(x_size*y_size)
        k_2_p=np.zeros(x_size*y_size) 
        R_I=np.zeros(x_size*y_size)
        BP=np.zeros(x_size*y_size)
        wsse_min=np.ones(x_size*y_size)*10**5


        for i, BF_i in enumerate(curv_lists):
            X = np.column_stack((Ref_concentration, BF_i))  # Efficiently create X
        
            # Solve for b using least squares instead of manual pseudo-inverse and matrix multiplication
            b, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
            # Calculate wsse if residuals are not directly provided
            wsse = residuals if residuals.size > 0 else np.sum((X @ b - y) ** 2)
        
            indices = wsse < wsse_min
        
            # # Update wsse_min and other variables where wsse is smaller
            # wsse_min[indices] = wsse[indices]
            # R_I[indices] = b[0][indices]   
            # BP[indices] = b[1][indices] / (t_3_values[i] - lambda_F) + b[0][indices] - 1
            # k_2[indices] = b[1][indices] + b[0][indices] * (t_3_values[i] - lambda_F) 
            
            # Update wsse_min and other variables where wsse is smaller
            wsse_min[indices] = wsse[indices]
            R_I[indices] = b[0][indices]   
            BP[indices] = (b[1][indices] / t_3_values[i]) + b[0][indices] - 1
            k_2[indices] = t_3_values[i]
            k_2_p[indices] = (b[1][indices] / b[0][indices]) + t_3_values[i] 
    
        
        BP_reshape=BP.reshape(new_list.shape[0], new_list.shape[1])
        BP_reshape_list.append(BP_reshape)
        
        K_2_reshape=k_2.reshape(new_list.shape[0], new_list.shape[1])
        K_2_reshape_list.append(K_2_reshape)
        
        K_2_p_reshape=k_2_p.reshape(new_list.shape[0], new_list.shape[1])
        K_2_p_reshape_list.append(K_2_p_reshape)
    
        R_I_reshape=R_I.reshape(new_list.shape[0], new_list.shape[1])
        R_I_reshape_list.append(R_I_reshape)
        
    index = np.searchsorted(Ref_concentration_time, 30, side='right')
    
    for i in range(index,np.shape(Ref_concentration_time)[0]):
        B=np.sum(y, axis=0)/np.sum(Ref_concentration)
    print(B.shape)  
  #-----------   
    B = B.reshape((1, B.shape[0], 1))
    BP_reshape_list=B
    print(BP_reshape_list.shape)      
    num_valid_coords = len(valid_coords[0])
    assert num_valid_coords == BP_reshape_list.shape[1], "Mismatch in the number of valid coordinates and reshaped list length"
    
    restored_array_BP = np.zeros((corrected_data.shape[0], corrected_data.shape[1], corrected_data.shape[2]))
    
    for idx in range(num_valid_coords):
        x, y, z = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        restored_array_BP[x, y, z] = BP_reshape_list[0, idx, 0]
#---------------       
    R_I_reshape_list=np.array(R_I_reshape_list)
            
    num_valid_coords = len(valid_coords[0])
    assert num_valid_coords == R_I_reshape_list.shape[1], "Mismatch in the number of valid coordinates and reshaped list length"
    
    restored_array_R_i = np.zeros((corrected_data.shape[0], corrected_data.shape[1], corrected_data.shape[2]))
    
    for idx in range(num_valid_coords):
        x, y, z = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        restored_array_R_i[x, y, z] = R_I_reshape_list[0, idx, 0]
#-----------        
    K_2_reshape_list=np.array(K_2_reshape_list)
            
    num_valid_coords = len(valid_coords[0])
    assert num_valid_coords == K_2_reshape_list.shape[1], "Mismatch in the number of valid coordinates and reshaped list length"
    
    restored_array_k_2 = np.zeros((corrected_data.shape[0], corrected_data.shape[1], corrected_data.shape[2]))
    
    for idx in range(num_valid_coords):
        x, y, z = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        restored_array_k_2[x, y, z] = K_2_reshape_list[0, idx, 0]
#---------
    K_2_p_reshape_list=np.array(K_2_p_reshape_list)
            
    num_valid_coords = len(valid_coords[0])
    assert num_valid_coords == K_2_p_reshape_list.shape[1], "Mismatch in the number of valid coordinates and reshaped list length"
    
    restored_array_k_2_p = np.zeros((corrected_data.shape[0], corrected_data.shape[1], corrected_data.shape[2]))
    
    for idx in range(num_valid_coords):
        x, y, z = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        restored_array_k_2_p[x, y, z] = K_2_p_reshape_list[0, idx, 0]
    




    slut_tid_3=time.time()
    ber_tid=slut_tid_3-start_tid_3
    print("beräkning tid:", int(ber_tid),"s")
    
    restored_array_R_i[restored_array_R_i > 5] = 5
    restored_array_R_i[restored_array_R_i < 0] = np.nan
    
    # restored_array_BP = np.clip(K_2_volume, 0.1, 2)
    # restored_array_BP[restored_array_BP < 0] = np.nan
    
    # print(np.array(R_I_reshape_list).shape)
    # return BP_reshape_list, K_2_reshape_list, R_I_reshape_list , K_2_p_reshape_list
    return restored_array_BP, restored_array_k_2, restored_array_R_i , restored_array_k_2_p
