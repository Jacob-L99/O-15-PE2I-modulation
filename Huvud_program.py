import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from screeninfo import get_monitors
# import gc
#removes all the plots 
import matplotlib
matplotlib.use('Agg')


plt.ioff()

import tempfile
import traceback
import shutil
import sys
# Create a temporary file
temp_output = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.log')
temp_output_path = temp_output.name

# Redirect stdout and stder
sys.stdout = temp_output
sys.stderr = temp_output

# Initialize global variables for selected paths and the number of selections
monitors = get_monitors()
monitor = monitors[0]  # Use the first monitor
original_screen_width = monitor.width
original_screen_height = monitor.height
scale_factor = 1.5 * (original_screen_width / 1920)

selected_directory1 = None
selected_file1 = None
selected_directory2 = None
selected_file2 = None
num_selections = 1


def proceed():
    global num_selections
    num_selections = 2 if two_files_var.get() else 1
    if selected_directory1 and selected_file1:
        if num_selections == 2 and (not selected_directory2 or not selected_file2):
            messagebox.showinfo("Info", "Välj både andra mappen och andra filen för att fortsätta.")
            return
        update_window_for_tasks()
    else:
        messagebox.showinfo("Info", "Välj åtminstone en mapp och en fil för att fortsätta.")

def def_global(n):
    global Wx, dicom_name, inf_name
    if n == 1:
        Wx = "wat1"
        dicom_name = selected_directory1
        inf_name = selected_file1
    elif n == 2:
        Wx = "wat2"
        dicom_name = selected_directory2
        inf_name = selected_file2

from Läsa_in_och_sortera import läsa_in
from Nersampling import Downsample
from Registrering_med_GUI import Registrering
from Rörelse_korrektion import rörelse_korrektion
from Beräkningar_3 import beräkningar_3
from Transform import transform
from SSP_2d import SSP_2D
from Z_score import SD_corrected, SD_corrected_park
from Z_score_SSP import SSP_Z
from Z_score_brain_surface import fig_get
from Z_score_brain_surface import fig_get_regions
from Z_score_brain_surface_parkinson import fig_get_flow_regions
from Justera_c_bar import open_app_window
from kolla_normaldata import check_normaldata
from plot_flow_regioner_mean import flow_regions_z_score
from plot_regioner_mean import regions_z_score
from Justera_c_bar_wat2 import open_app_window_wat2
from Transform_pat import transform_pat_space

def flow_regions():
    import sys
    import traceback

    try:
        normal_data=load_direction()
        if num_selections==2:
            wat='wat1'
            norm=check_normaldata('wat_1_2', normal_data)
        else:
            wat='wat'
            norm=check_normaldata('wat', normal_data)
        
        if not norm:
            # Raise an exception to trigger the except block
            raise FileNotFoundError("All nomrmaldata hittades inte")
        
        try:
            data_4d_1, AIF_1, AIF_time_1, first_image_shape_1, pixel_spacing, slice_thickness, image_positions = läsa_in(dicom_name, inf_name)
        except Exception as e:
            raise FileNotFoundError("Kunde inte läsa in, kontrollera att du valde rätt mapp/fil")
        try:
            loading_label.config(text="Omskalar bild")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            data_4d_1 = Downsample(data_4d_1, first_image_shape_1)
            loading_label.config(text="Registrerar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            registration_1_1, registration_2_1, template_3d_1=Registrering(data_4d_1, initial_window, loading_label, original_screen_width, seed=1)
            for widget in initial_window.winfo_children():
                widget.pack_forget()
            plt.close('all')
            loading_label.config(text="Rörelsekorrektion")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            corrected_data_1, motion_parameters_array_1=rörelse_korrektion(data_4d_1)
            loading_label.config(text="Beräknar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            K_1_reshape_list_1, K_2_reshape_list_1, V_a_reshape_list_1 = beräkningar_3(corrected_data_1, AIF_time_1, AIF_1)
            loading_label.config(text="Transformerar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()      
            transformed_K_1_1, transformed_K_2_1, transformed_V_a_1 = transform(K_1_reshape_list_1, K_2_reshape_list_1, V_a_reshape_list_1, registration_1_1, registration_2_1, template_3d_1)
            loading_label.config(text="Genererar SSP")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2=SSP_2D(transformed_K_1_1)
            Z_brain, z_scores_flow, Cerebellum_mean_k1, means_list_k1 = SD_corrected(transformed_K_1_1, 'real', wat, normal_data)
            Z_brain_k2, z_scores_flow_k2, Cerebellum_mean_k2, means_list_k2 = SD_corrected(transformed_K_2_1, 'real', wat, normal_data)
            
            
            flow_regions_first_values_yz_neg_x0, flow_regions_first_values_yz_pos_x1, flow_regions_first_values_yz_pos_x2, flow_regions_first_values_yz_neg_x2, flow_regions_first_values_xz_pos_y2, flow_regions_first_values_xz_neg_y2 = flow_regions_z_score(Z_brain)
            flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6 = fig_get_flow_regions(flow_regions_first_values_yz_neg_x0, flow_regions_first_values_yz_pos_x1, flow_regions_first_values_yz_pos_x2, flow_regions_first_values_yz_neg_x2, flow_regions_first_values_xz_pos_y2, flow_regions_first_values_xz_neg_y2)

            plt.close('all')
            # gc.collect()
            
            loading_label.config(text="Beräknar z-score")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            neg_x0, pos_x1, pos_x2, neg_x2, pos_y2, neg_y2 = SSP_Z(first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2, 'real', wat, normal_data)
            last_figure_1, last_figure_2, last_figure_3, last_figure_4, last_figure_5, last_figure_6 = fig_get(neg_x0, pos_x1, pos_x2, neg_x2, pos_y2, neg_y2)
 
            regions_first_values_yz_neg_x0, regions_first_values_yz_pos_x1, regions_first_values_yz_pos_x2, regions_first_values_yz_neg_x2, regions_first_values_xz_pos_y2, regions_first_values_xz_neg_y2, z_score, z_brain_regions, mean_values_k1 = regions_z_score(transformed_K_1_1, 'real', wat, normal_data)
            region_last_figure_1, region_last_figure_2, region_last_figure_3, region_last_figure_4, region_last_figure_5, region_last_figure_6 = fig_get_regions(regions_first_values_yz_neg_x0, regions_first_values_yz_pos_x1, regions_first_values_yz_pos_x2, regions_first_values_yz_neg_x2, regions_first_values_xz_pos_y2, regions_first_values_xz_neg_y2)
        
    
            V_a_reshape_list_1, K_2_reshape_list_1, K_1_reshape_list_1=np.transpose(V_a_reshape_list_1, (1,2,0)), np.transpose(K_2_reshape_list_1, (1,2,0)), np.transpose(K_1_reshape_list_1, (1,2,0))
            
            plt.close('all')
            # gc.collect()
            
            #relative--------------------------
            loading_label.config(text="Relativa beräkningar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            Z_brain_rel, z_scores_flow_rel, Cerebellum_mean_k1_ref, means_list_k1_ref = SD_corrected(transformed_K_1_1/Cerebellum_mean_k1, 'ref', wat, normal_data)
            first_values_yz_neg_x0_rel, first_values_yz_pos_x1_rel, first_values_yz_pos_x2_rel, first_values_yz_neg_x2_rel, first_values_xz_pos_y2_rel, first_values_xz_neg_y2_rel=first_values_yz_neg_x0/Cerebellum_mean_k1, first_values_yz_pos_x1/Cerebellum_mean_k1, first_values_yz_pos_x2/Cerebellum_mean_k1, first_values_yz_neg_x2/Cerebellum_mean_k1, first_values_xz_pos_y2/Cerebellum_mean_k1, first_values_xz_neg_y2/Cerebellum_mean_k1
            flow_regions_first_values_yz_neg_x0_rel, flow_regions_first_values_yz_pos_x1_rel, flow_regions_first_values_yz_pos_x2_rel, flow_regions_first_values_yz_neg_x2_rel, flow_regions_first_values_xz_pos_y2_rel, flow_regions_first_values_xz_neg_y2_rel = flow_regions_z_score(Z_brain_rel)
            flow_region_last_figure_1_rel, flow_region_last_figure_2_rel, flow_region_last_figure_3_rel, flow_region_last_figure_4_rel, flow_region_last_figure_5_rel, flow_region_last_figure_6_rel = fig_get_flow_regions(flow_regions_first_values_yz_neg_x0_rel, flow_regions_first_values_yz_pos_x1_rel, flow_regions_first_values_yz_pos_x2_rel, flow_regions_first_values_yz_neg_x2_rel, flow_regions_first_values_xz_pos_y2_rel, flow_regions_first_values_xz_neg_y2_rel)
        
            loading_label.config(text="Genererar relativa SSP")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            neg_x0_rel, pos_x1_rel, pos_x2_rel, neg_x2_rel, pos_y2_rel, neg_y2_rel = SSP_Z(first_values_yz_neg_x0_rel, first_values_yz_pos_x1_rel, first_values_yz_pos_x2_rel, first_values_yz_neg_x2_rel, first_values_xz_pos_y2_rel, first_values_xz_neg_y2_rel, 'ref', wat, normal_data)
            last_figure_1_rel, last_figure_2_rel, last_figure_3_rel, last_figure_4_rel, last_figure_5_rel, last_figure_6_rel = fig_get(neg_x0_rel, pos_x1_rel, pos_x2_rel, neg_x2_rel, pos_y2_rel, neg_y2_rel)
            
            regions_first_values_yz_neg_x0_rel, regions_first_values_yz_pos_x1_rel, regions_first_values_yz_pos_x2_rel, regions_first_values_yz_neg_x2_rel, regions_first_values_xz_pos_y2_rel, regions_first_values_xz_neg_y2_rel, z_score_rel, z_brain_regions_rel, mean_values_k1_ref = regions_z_score(transformed_K_1_1/Cerebellum_mean_k1, 'ref', wat, normal_data)
            region_last_figure_1_rel, region_last_figure_2_rel, region_last_figure_3_rel, region_last_figure_4_rel, region_last_figure_5_rel, region_last_figure_6_rel = fig_get_regions(regions_first_values_yz_neg_x0_rel, regions_first_values_yz_pos_x1_rel, regions_first_values_yz_pos_x2_rel, regions_first_values_yz_neg_x2_rel, regions_first_values_xz_pos_y2_rel, regions_first_values_xz_neg_y2_rel)
            
            plt.close('all')
            # gc.collect()
            
            K_1_reshape_list_1_pat=K_1_reshape_list_1.copy()
            K_2_reshape_list_1_pat=K_2_reshape_list_1.copy()

            K_1_reshape_list_1=np.transpose(K_1_reshape_list_1_pat, (2,1,0))
            K_2_reshape_list_1=np.transpose(K_2_reshape_list_1_pat, (2,1,0))
            
            K_1_reshape_list_1, K_2_reshape_list_1=transform_pat_space(K_1_reshape_list_1, K_2_reshape_list_1, registration_1_1,  template_3d_1)
            
            plt.close('all')
            # gc.collect()

            
            if num_selections==2:
                def_global(2)
                
                try:
                    data_4d_2, AIF_2, AIF_time_2, first_image_shape_2, pixel_spacing, slice_thickness, image_positions = läsa_in(dicom_name, inf_name)
                except Exception as e:
                    raise FileNotFoundError("Kunde inte läsa in (stress), kontrollera att du valde rätt mapp/fil")
                
                loading_label.config(text="Omskalar bild (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                data_4d_2 = Downsample(data_4d_2, first_image_shape_2)
                loading_label.config(text="Registrerar (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                registration_1_2, registration_2_2, template_3d_2=Registrering(data_4d_2, initial_window, loading_label, original_screen_width, seed=1)
                for widget in initial_window.winfo_children():
                    widget.pack_forget()
                plt.close('all')
                loading_label.config(text="Rörelsekorrektion (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                corrected_data_2, motion_parameters_array_2=rörelse_korrektion(data_4d_2)
                
                loading_label.config(text="Registreras stress till baseline")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                from reg_wat import register_wat
                reg_3=register_wat(data_4d_1, data_4d_2)
                from transform_wat import transform_wat
                corrected_data_2=transform_wat(reg_3, corrected_data_2)
                
                loading_label.config(text="Beräknar (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                K_1_reshape_list_2, K_2_reshape_list_2, V_a_reshape_list_2 = beräkningar_3(corrected_data_2, AIF_time_2, AIF_2)
                loading_label.config(text="Transformerar (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()          
                transformed_K_1_2, transformed_K_2_2, transformed_V_a_2 = transform(K_1_reshape_list_2, K_2_reshape_list_2, V_a_reshape_list_2, registration_1_2, registration_2_2, template_3d_2)
                loading_label.config(text="Genererar SSP (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                first_values_yz_neg_x0_2, first_values_yz_pos_x1_2, first_values_yz_pos_x2_2, first_values_yz_neg_x2_2, first_values_xz_pos_y2_2, first_values_xz_neg_y2_2=SSP_2D(transformed_K_1_2)    
                Z_brain_2, z_scores_flow_2, Cerebellum_mean_k1_2, means_list_k1_2 = SD_corrected(transformed_K_1_2, 'real', "wat2", normal_data)
                Z_brain_k2_2, z_scores_flow_k2_2, Cerebellum_mean_k2_2, means_list_k2_2 = SD_corrected(transformed_K_2_2, 'real', 'wat2', normal_data)
                flow_regions_first_values_yz_neg_x0_2, flow_regions_first_values_yz_pos_x1_2, flow_regions_first_values_yz_pos_x2_2, flow_regions_first_values_yz_neg_x2_2, flow_regions_first_values_xz_pos_y2_2, flow_regions_first_values_xz_neg_y2_2 = flow_regions_z_score(Z_brain_2)
                flow_region_last_figure_1_2, flow_region_last_figure_2_2, flow_region_last_figure_3_2, flow_region_last_figure_4_2, flow_region_last_figure_5_2, flow_region_last_figure_6_2 = fig_get_flow_regions(flow_regions_first_values_yz_neg_x0_2, flow_regions_first_values_yz_pos_x1_2, flow_regions_first_values_yz_pos_x2_2, flow_regions_first_values_yz_neg_x2_2, flow_regions_first_values_xz_pos_y2_2, flow_regions_first_values_xz_neg_y2_2)
            
                plt.close('all')
                # gc.collect()
            
                loading_label.config(text="Beräknar z-score (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                neg_x0_2, pos_x1_2, pos_x2_2, neg_x2_2, pos_y2_2, neg_y2_2 = SSP_Z(first_values_yz_neg_x0_2, first_values_yz_pos_x1_2, first_values_yz_pos_x2_2, first_values_yz_neg_x2_2, first_values_xz_pos_y2_2, first_values_xz_neg_y2_2, 'real', 'wat2', normal_data)
                last_figure_1_2, last_figure_2_2, last_figure_3_2, last_figure_4_2, last_figure_5_2, last_figure_6_2 = fig_get(neg_x0_2, pos_x1_2, pos_x2_2, neg_x2_2, pos_y2_2, neg_y2_2)
                
                regions_first_values_yz_neg_x0_2, regions_first_values_yz_pos_x1_2, regions_first_values_yz_pos_x2_2, regions_first_values_yz_neg_x2_2, regions_first_values_xz_pos_y2_2, regions_first_values_xz_neg_y2_2, z_score_2, z_brain_regions_2, mean_values_k1_2 = regions_z_score(transformed_K_1_2, 'real', 'wat2', normal_data)
                region_last_figure_1_2, region_last_figure_2_2, region_last_figure_3_2, region_last_figure_4_2, region_last_figure_5_2, region_last_figure_6_2 = fig_get_regions(regions_first_values_yz_neg_x0_2, regions_first_values_yz_pos_x1_2, regions_first_values_yz_pos_x2_2, regions_first_values_yz_neg_x2_2, regions_first_values_xz_pos_y2_2, regions_first_values_xz_neg_y2_2)
            
        
                V_a_reshape_list_2, K_2_reshape_list_2, K_1_reshape_list_2=np.transpose(V_a_reshape_list_2, (1,2,0)), np.transpose(K_2_reshape_list_2, (1,2,0)), np.transpose(K_1_reshape_list_2, (1,2,0))
                
                plt.close('all')
                # gc.collect()
                
                #relative--------------------------
                loading_label.config(text="Relativa beräkningar (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                Z_brain_rel_2, z_scores_flow_rel_2, Cerebellum_mean_k1_ref_2, means_list_k1_ref_2 = SD_corrected(transformed_K_1_2/Cerebellum_mean_k1_2, 'ref', 'wat2', normal_data)
                
                first_values_yz_neg_x0_rel_2, first_values_yz_pos_x1_rel_2, first_values_yz_pos_x2_rel_2, first_values_yz_neg_x2_rel_2, first_values_xz_pos_y2_rel_2, first_values_xz_neg_y2_rel_2=first_values_yz_neg_x0_2/Cerebellum_mean_k1_2, first_values_yz_pos_x1_2/Cerebellum_mean_k1_2, first_values_yz_pos_x2_2/Cerebellum_mean_k1_2, first_values_yz_neg_x2_2/Cerebellum_mean_k1_2, first_values_xz_pos_y2_2/Cerebellum_mean_k1_2, first_values_xz_neg_y2_2/Cerebellum_mean_k1_2
                
                
                
                flow_regions_first_values_yz_neg_x0_rel_2, flow_regions_first_values_yz_pos_x1_rel_2, flow_regions_first_values_yz_pos_x2_rel_2, flow_regions_first_values_yz_neg_x2_rel_2, flow_regions_first_values_xz_pos_y2_rel_2, flow_regions_first_values_xz_neg_y2_rel_2 = flow_regions_z_score(Z_brain_rel_2)
                flow_region_last_figure_1_rel_2, flow_region_last_figure_2_rel_2, flow_region_last_figure_3_rel_2, flow_region_last_figure_4_rel_2, flow_region_last_figure_5_rel_2, flow_region_last_figure_6_rel_2 = fig_get_flow_regions(flow_regions_first_values_yz_neg_x0_rel_2, flow_regions_first_values_yz_pos_x1_rel_2, flow_regions_first_values_yz_pos_x2_rel_2, flow_regions_first_values_yz_neg_x2_rel_2, flow_regions_first_values_xz_pos_y2_rel_2, flow_regions_first_values_xz_neg_y2_rel_2)
            
                loading_label.config(text="Genererar relativa SSP (stress)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                neg_x0_rel_2, pos_x1_rel_2, pos_x2_rel_2, neg_x2_rel_2, pos_y2_rel_2, neg_y2_rel_2 = SSP_Z(first_values_yz_neg_x0_rel_2, first_values_yz_pos_x1_rel_2, first_values_yz_pos_x2_rel_2, first_values_yz_neg_x2_rel_2, first_values_xz_pos_y2_rel_2, first_values_xz_neg_y2_rel_2, 'ref', 'wat2', normal_data)
                last_figure_1_rel_2, last_figure_2_rel_2, last_figure_3_rel_2, last_figure_4_rel_2, last_figure_5_rel_2, last_figure_6_rel_2 = fig_get(neg_x0_rel_2, pos_x1_rel_2, pos_x2_rel_2, neg_x2_rel_2, pos_y2_rel_2, neg_y2_rel_2)
                
                plt.close('all')
                # gc.collect()
                
                regions_first_values_yz_neg_x0_rel_2, regions_first_values_yz_pos_x1_rel_2, regions_first_values_yz_pos_x2_rel_2, regions_first_values_yz_neg_x2_rel_2, regions_first_values_xz_pos_y2_rel_2, regions_first_values_xz_neg_y2_rel_2, z_score_rel_2, z_brain_regions_rel_2, mean_values_k1_ref_2 = regions_z_score(transformed_K_1_2/Cerebellum_mean_k1_2, 'ref', 'wat2', normal_data)
                region_last_figure_1_rel_2, region_last_figure_2_rel_2, region_last_figure_3_rel_2, region_last_figure_4_rel_2, region_last_figure_5_rel_2, region_last_figure_6_rel_2 = fig_get_regions(regions_first_values_yz_neg_x0_rel_2, regions_first_values_yz_pos_x1_rel_2, regions_first_values_yz_pos_x2_rel_2, regions_first_values_yz_neg_x2_rel_2, regions_first_values_xz_pos_y2_rel_2, regions_first_values_xz_neg_y2_rel_2)
                
                K_1_reshape_list_2_pat=K_1_reshape_list_2.copy()
                K_2_reshape_list_2_pat=K_2_reshape_list_2.copy()

                K_1_reshape_list_2=np.transpose(K_1_reshape_list_2_pat, (2,1,0))
                K_2_reshape_list_2=np.transpose(K_2_reshape_list_2_pat, (2,1,0))
                
                K_1_reshape_list_2, K_2_reshape_list_2=transform_pat_space(K_1_reshape_list_2, K_2_reshape_list_2, registration_1_2,  template_3d_1)
                
                plt.close('all')
                # gc.collect()

                
                #wat1 wat2 
                loading_label.config(text="Beräknar (flödes reserv)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                first_values_yz_neg_x0_3, first_values_yz_pos_x1_3, first_values_yz_pos_x2_3, first_values_yz_neg_x2_3, first_values_xz_pos_y2_3, first_values_xz_neg_y2_3=first_values_yz_neg_x0_2/first_values_yz_neg_x0, first_values_yz_pos_x1_2/first_values_yz_pos_x1, first_values_yz_pos_x2_2/first_values_yz_pos_x2, first_values_yz_neg_x2_2/first_values_yz_neg_x2, first_values_xz_pos_y2_2/first_values_xz_pos_y2, first_values_xz_neg_y2_2/first_values_xz_neg_y2

    
                Z_brain_3, z_scores_flow_3, Cerebellum_mean_k1_3, means_list_k1_3 = SD_corrected(transformed_K_1_2/transformed_K_1_1, 'real', 'wat1_2', normal_data)
                Z_brain_k2_3, z_scores_flow_k2_3, Cerebellum_mean_k2_3, means_list_k2_3 = SD_corrected(transformed_K_2_2/transformed_K_2_1, 'real', 'wat1_2', normal_data)
                
                flow_regions_first_values_yz_neg_x0_3, flow_regions_first_values_yz_pos_x1_3, flow_regions_first_values_yz_pos_x2_3, flow_regions_first_values_yz_neg_x2_3, flow_regions_first_values_xz_pos_y2_3, flow_regions_first_values_xz_neg_y2_3 = flow_regions_z_score(Z_brain_3)
                flow_region_last_figure_1_3, flow_region_last_figure_2_3, flow_region_last_figure_3_3, flow_region_last_figure_4_3, flow_region_last_figure_5_3, flow_region_last_figure_6_3 = fig_get_flow_regions(flow_regions_first_values_yz_neg_x0_3, flow_regions_first_values_yz_pos_x1_3, flow_regions_first_values_yz_pos_x2_3, flow_regions_first_values_yz_neg_x2_3, flow_regions_first_values_xz_pos_y2_3, flow_regions_first_values_xz_neg_y2_3)
            
                plt.close('all')
                # gc.collect()
            
                loading_label.config(text="Genererar SSP (flödes reserv)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                neg_x0_3, pos_x1_3, pos_x2_3, neg_x2_3, pos_y2_3, neg_y2_3 = SSP_Z(first_values_yz_neg_x0_3, first_values_yz_pos_x1_3, first_values_yz_pos_x2_3, first_values_yz_neg_x2_3, first_values_xz_pos_y2_3, first_values_xz_neg_y2_3, 'real', 'wat1_2', normal_data)
                last_figure_1_3, last_figure_2_3, last_figure_3_3, last_figure_4_3, last_figure_5_3, last_figure_6_3 = fig_get(neg_x0_3, pos_x1_3, pos_x2_3, neg_x2_3, pos_y2_3, neg_y2_3)
                
                regions_first_values_yz_neg_x0_3, regions_first_values_yz_pos_x1_3, regions_first_values_yz_pos_x2_3, regions_first_values_yz_neg_x2_3, regions_first_values_xz_pos_y2_3, regions_first_values_xz_neg_y2_3, z_score_3, z_brain_regions_3, mean_values_k1_3 = regions_z_score(transformed_K_1_2/transformed_K_1_1, 'real', 'wat1_2', normal_data)
                region_last_figure_1_3, region_last_figure_2_3, region_last_figure_3_3, region_last_figure_4_3, region_last_figure_5_3, region_last_figure_6_3 = fig_get_regions(regions_first_values_yz_neg_x0_3, regions_first_values_yz_pos_x1_3, regions_first_values_yz_pos_x2_3, regions_first_values_yz_neg_x2_3, regions_first_values_xz_pos_y2_3, regions_first_values_xz_neg_y2_3)
            
                Z_brain_rel_3, z_scores_flow_rel_3, Cerebellum_mean_k1_ref_3, means_list_k1_ref_3 = SD_corrected((transformed_K_1_2/Cerebellum_mean_k1_2)/(transformed_K_1_1/Cerebellum_mean_k1), 'ref', 'wat1_2', normal_data)
                loading_label.config(text="Relativa beräknar (flödes reserv)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                initial_window.update_idletasks()  # Force update of the label text
                first_values_yz_neg_x0_rel_3, first_values_yz_pos_x1_rel_3, first_values_yz_pos_x2_rel_3, first_values_yz_neg_x2_rel_3, first_values_xz_pos_y2_rel_3, first_values_xz_neg_y2_rel_3=first_values_yz_neg_x0_3/Cerebellum_mean_k1_3, first_values_yz_pos_x1_3/Cerebellum_mean_k1_3, first_values_yz_pos_x2_3/Cerebellum_mean_k1_3, first_values_yz_neg_x2_3/Cerebellum_mean_k1_3, first_values_xz_pos_y2_3/Cerebellum_mean_k1_3, first_values_xz_neg_y2_3/Cerebellum_mean_k1_3
                
                plt.close('all')
                # gc.collect()
                
                flow_regions_first_values_yz_neg_x0_rel_3, flow_regions_first_values_yz_pos_x1_rel_3, flow_regions_first_values_yz_pos_x2_rel_3, flow_regions_first_values_yz_neg_x2_rel_3, flow_regions_first_values_xz_pos_y2_rel_3, flow_regions_first_values_xz_neg_y2_rel_3 = flow_regions_z_score(Z_brain_rel_3)
                flow_region_last_figure_1_rel_3, flow_region_last_figure_2_rel_3, flow_region_last_figure_3_rel_3, flow_region_last_figure_4_rel_3, flow_region_last_figure_5_rel_3, flow_region_last_figure_6_rel_3 = fig_get_flow_regions(flow_regions_first_values_yz_neg_x0_rel_3, flow_regions_first_values_yz_pos_x1_rel_3, flow_regions_first_values_yz_pos_x2_rel_3, flow_regions_first_values_yz_neg_x2_rel_3, flow_regions_first_values_xz_pos_y2_rel_3, flow_regions_first_values_xz_neg_y2_rel_3)
            
                loading_label.config(text="Genererar relativa SSP (flödes reserv)")
                loading_label.pack(expand=True)
                initial_window.update_idletasks()  # Force update of the label text
                initial_window.update()
                neg_x0_rel_3, pos_x1_rel_3, pos_x2_rel_3, neg_x2_rel_3, pos_y2_rel_3, neg_y2_rel_3 = SSP_Z(first_values_yz_neg_x0_rel_3, first_values_yz_pos_x1_rel_3, first_values_yz_pos_x2_rel_3, first_values_yz_neg_x2_rel_3, first_values_xz_pos_y2_rel_3, first_values_xz_neg_y2_rel_3, 'ref', 'wat1_2', normal_data)
                last_figure_1_rel_3, last_figure_2_rel_3, last_figure_3_rel_3, last_figure_4_rel_3, last_figure_5_rel_3, last_figure_6_rel_3 = fig_get(neg_x0_rel_3, pos_x1_rel_3, pos_x2_rel_3, neg_x2_rel_3, pos_y2_rel_3, neg_y2_rel_3)
                
                regions_first_values_yz_neg_x0_rel_3, regions_first_values_yz_pos_x1_rel_3, regions_first_values_yz_pos_x2_rel_3, regions_first_values_yz_neg_x2_rel_3, regions_first_values_xz_pos_y2_rel_3, regions_first_values_xz_neg_y2_rel_3, z_score_rel_3, z_brain_regions_rel_3, mean_values_k1_ref_3 = regions_z_score((transformed_K_1_2/Cerebellum_mean_k1_2)/(transformed_K_1_1/Cerebellum_mean_k1), 'ref', 'wat1_2', normal_data)
                region_last_figure_1_rel_3, region_last_figure_2_rel_3, region_last_figure_3_rel_3, region_last_figure_4_rel_3, region_last_figure_5_rel_3, region_last_figure_6_rel_3 = fig_get_regions(regions_first_values_yz_neg_x0_rel_3, regions_first_values_yz_pos_x1_rel_3, regions_first_values_yz_pos_x2_rel_3, regions_first_values_yz_neg_x2_rel_3, regions_first_values_xz_pos_y2_rel_3, regions_first_values_xz_neg_y2_rel_3)
            
            
                K_1_reshape_list_3_pat, K_2_reshape_list_3_pat= K_1_reshape_list_2_pat/K_1_reshape_list_1_pat, K_2_reshape_list_2_pat/K_2_reshape_list_1_pat
            
                plt.close('all')
                # gc.collect()
                
            
            if num_selections==1:
                button_pressed = tk.StringVar()
                # Open the App window, passing the StringVar
                initial_window.app_window = open_app_window(initial_window, button_pressed, \
                                                            transformed_K_1_1, transformed_K_2_1, Z_brain, K_1_reshape_list_1, \
                                                            K_2_reshape_list_1, first_values_yz_neg_x0, first_values_yz_pos_x1, \
                                                            first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, \
                                                            first_values_xz_neg_y2, \
                                                            last_figure_1, last_figure_2, last_figure_3, \
                                                            last_figure_4, last_figure_5, last_figure_6, \
                                                            region_last_figure_1, region_last_figure_2, region_last_figure_3, \
                                                            region_last_figure_4, region_last_figure_5, region_last_figure_6, \
                                                            flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, \
                                                            flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6, \
                                                            z_score, z_scores_flow, z_brain_regions, Cerebellum_mean_k1, Cerebellum_mean_k2, \
                                                            first_values_yz_neg_x0_rel, first_values_yz_pos_x1_rel, 
                                                            first_values_yz_pos_x2_rel, first_values_yz_neg_x2_rel, first_values_xz_pos_y2_rel, 
                                                            first_values_xz_neg_y2_rel, 
                                                            last_figure_1_rel, last_figure_2_rel, last_figure_3_rel, 
                                                            last_figure_4_rel, last_figure_5_rel, last_figure_6_rel, 
                                                            region_last_figure_1_rel, region_last_figure_2_rel, region_last_figure_3_rel, 
                                                            region_last_figure_4_rel, region_last_figure_5_rel, region_last_figure_6_rel, 
                                                            flow_region_last_figure_1_rel, flow_region_last_figure_2_rel, flow_region_last_figure_3_rel, 
                                                            flow_region_last_figure_4_rel, flow_region_last_figure_5_rel, flow_region_last_figure_6_rel,
                                                            z_score_rel, z_scores_flow_rel,
                                                            Z_brain_rel, z_brain_regions_rel,
                                                            means_list_k1 ,mean_values_k1,
                                                            means_list_k1_ref ,mean_values_k1_ref,
                                                            motion_parameters_array_1,
                                                            dicom_name, K_1_reshape_list_1_pat, K_2_reshape_list_1_pat,
                                                            pixel_spacing, slice_thickness, image_positions)
            
                # Wait until the StringVar is set by one of the buttons
                initial_window.wait_variable(button_pressed)
            
                # Retrieve the value set by the button
                response = button_pressed.get()
                print(f"Button pressed: {response}")
                initial_window.destroy()
            else:
                #%%
                
                button_pressed = tk.StringVar()
                # Open the App window, passing the StringVar

                transformed_K_1_3=transformed_K_1_2/transformed_K_1_1
                transformed_K_2_3=transformed_K_2_2/transformed_K_2_1
                K_1_reshape_list_3=K_1_reshape_list_2/K_1_reshape_list_1
                K_2_reshape_list_3=K_2_reshape_list_2/K_2_reshape_list_1
                initial_window.app_window = open_app_window_wat2(initial_window, button_pressed, \
                                                            transformed_K_1_1, transformed_K_2_1, Z_brain, K_1_reshape_list_1, \
                                                            K_2_reshape_list_1, first_values_yz_neg_x0, first_values_yz_pos_x1, \
                                                            first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, \
                                                            first_values_xz_neg_y2, \
                                                            last_figure_1, last_figure_2, last_figure_3, \
                                                            last_figure_4, last_figure_5, last_figure_6, \
                                                            region_last_figure_1, region_last_figure_2, region_last_figure_3, \
                                                            region_last_figure_4, region_last_figure_5, region_last_figure_6, \
                                                            flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, \
                                                            flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6, \
                                                            z_score, z_scores_flow, z_brain_regions, Cerebellum_mean_k1, Cerebellum_mean_k2,  \
                                                            first_values_yz_neg_x0_rel, first_values_yz_pos_x1_rel, \
                                                            first_values_yz_pos_x2_rel, first_values_yz_neg_x2_rel, first_values_xz_pos_y2_rel, \
                                                            first_values_xz_neg_y2_rel, \
                                                            last_figure_1_rel, last_figure_2_rel, last_figure_3_rel, \
                                                            last_figure_4_rel, last_figure_5_rel, last_figure_6_rel, \
                                                            region_last_figure_1_rel, region_last_figure_2_rel, region_last_figure_3_rel, \
                                                            region_last_figure_4_rel, region_last_figure_5_rel, region_last_figure_6_rel, \
                                                            flow_region_last_figure_1_rel, flow_region_last_figure_2_rel, flow_region_last_figure_3_rel, \
                                                            flow_region_last_figure_4_rel, flow_region_last_figure_5_rel, flow_region_last_figure_6_rel, \
                                                            z_score_rel, z_scores_flow_rel, \
                                                            Z_brain_rel, z_brain_regions_rel, \
                                                            transformed_K_1_2, transformed_K_2_2, Z_brain_2, K_1_reshape_list_2, \
                                                            K_2_reshape_list_2, first_values_yz_neg_x0_2, first_values_yz_pos_x1_2, \
                                                            first_values_yz_pos_x2_2, first_values_yz_neg_x2_2, first_values_xz_pos_y2_2, \
                                                            first_values_xz_neg_y2_2, \
                                                            last_figure_1_2, last_figure_2_2, last_figure_3_2, \
                                                            last_figure_4_2, last_figure_5_2, last_figure_6_2, \
                                                            region_last_figure_1_2, region_last_figure_2_2, region_last_figure_3_2, \
                                                            region_last_figure_4_2, region_last_figure_5_2, region_last_figure_6_2, \
                                                            flow_region_last_figure_1_2, flow_region_last_figure_2_2, flow_region_last_figure_3_2, \
                                                            flow_region_last_figure_4_2, flow_region_last_figure_5_2, flow_region_last_figure_6_2, \
                                                            z_score_2, z_scores_flow_2, z_brain_regions_2, Cerebellum_mean_k1_2, Cerebellum_mean_k2_2, \
                                                            first_values_yz_neg_x0_rel_2, first_values_yz_pos_x1_rel_2, \
                                                            first_values_yz_pos_x2_rel_2, first_values_yz_neg_x2_rel_2, first_values_xz_pos_y2_rel_2, \
                                                            first_values_xz_neg_y2_rel_2, \
                                                            last_figure_1_rel_2, last_figure_2_rel_2, last_figure_3_rel_2, \
                                                            last_figure_4_rel_2, last_figure_5_rel_2, last_figure_6_rel_2, \
                                                            region_last_figure_1_rel_2, region_last_figure_2_rel_2, region_last_figure_3_rel_2, \
                                                            region_last_figure_4_rel_2, region_last_figure_5_rel_2, region_last_figure_6_rel_2, \
                                                            flow_region_last_figure_1_rel_2, flow_region_last_figure_2_rel_2, flow_region_last_figure_3_rel_2, \
                                                            flow_region_last_figure_4_rel_2, flow_region_last_figure_5_rel_2, flow_region_last_figure_6_rel_2, \
                                                            z_score_rel_2, z_scores_flow_rel_2,
                                                            Z_brain_rel_2, z_brain_regions_rel_2,
                                                            transformed_K_1_3, transformed_K_2_3, Z_brain_3, K_1_reshape_list_3, \
                                                            K_2_reshape_list_3, first_values_yz_neg_x0_3, first_values_yz_pos_x1_3, \
                                                            first_values_yz_pos_x2_3, first_values_yz_neg_x2_3, first_values_xz_pos_y2_3, \
                                                            first_values_xz_neg_y2_3, \
                                                            last_figure_1_3, last_figure_2_3, last_figure_3_3, \
                                                            last_figure_4_3, last_figure_5_3, last_figure_6_3, \
                                                            region_last_figure_1_3, region_last_figure_2_3, region_last_figure_3_3, \
                                                            region_last_figure_4_3, region_last_figure_5_3, region_last_figure_6_3, \
                                                            flow_region_last_figure_1_3, flow_region_last_figure_2_3, flow_region_last_figure_3_3, \
                                                            flow_region_last_figure_4_3, flow_region_last_figure_5_3, flow_region_last_figure_6_3, \
                                                            z_score_3, z_scores_flow_3, z_brain_regions_3, Cerebellum_mean_k1_3, Cerebellum_mean_k2_3, \
                                                            first_values_yz_neg_x0_rel_3, first_values_yz_pos_x1_rel_3, \
                                                            first_values_yz_pos_x2_rel_3, first_values_yz_neg_x2_rel_3, first_values_xz_pos_y2_rel_3, \
                                                            first_values_xz_neg_y2_rel_3, \
                                                            last_figure_1_rel_3, last_figure_2_rel_3, last_figure_3_rel_3, \
                                                            last_figure_4_rel_3, last_figure_5_rel_3, last_figure_6_rel_3, \
                                                            region_last_figure_1_rel_3, region_last_figure_2_rel_3, region_last_figure_3_rel_3, \
                                                            region_last_figure_4_rel_3, region_last_figure_5_rel_3, region_last_figure_6_rel_3, \
                                                            flow_region_last_figure_1_rel_3, flow_region_last_figure_2_rel_3, flow_region_last_figure_3_rel_3, \
                                                            flow_region_last_figure_4_rel_3, flow_region_last_figure_5_rel_3, flow_region_last_figure_6_rel_3, \
                                                            z_score_rel_3, z_scores_flow_rel_3,
                                                            Z_brain_rel_3, z_brain_regions_rel_3,
                                                            means_list_k1 ,mean_values_k1,
                                                            means_list_k1_ref ,mean_values_k1_ref,
                                                            motion_parameters_array_1,
                                                            means_list_k1_2 ,mean_values_k1_2,
                                                            means_list_k1_ref_2 ,mean_values_k1_ref_2,
                                                            motion_parameters_array_2,
                                                            means_list_k1_3 ,mean_values_k1_3,
                                                            means_list_k1_ref_3 ,mean_values_k1_ref_3,
                                                            dicom_name, K_1_reshape_list_1_pat, K_2_reshape_list_1_pat,
                                                            K_1_reshape_list_2_pat, K_2_reshape_list_2_pat,
                                                            K_1_reshape_list_3_pat, K_2_reshape_list_3_pat,
                                                            pixel_spacing, slice_thickness, image_positions
                                                            )
            
                # Wait until the StringVar is set by one of the buttons
                initial_window.wait_variable(button_pressed)
            
                # Retrieve the value set by the button
                response = button_pressed.get()
                print(f"Button pressed: {response}")
                initial_window.destroy()
        except Exception as e:
            raise FileNotFoundError("Något gick fel, prova igen")
    except Exception as e:
        # Print the traceback to the console for debugging purposes
        traceback.print_exc()
    
        # Destroy the main window if it's still open
        try:
            initial_window.destroy()
        except:
            pass  # If initial_window is already destroyed, ignore the error
    
        # Create a new window for the error message
        error_window = tk.Tk()
        error_window.title("Fel")
        error_window.geometry(f"{int(0.625 * original_screen_width)}x{int(0.32 * original_screen_width)}")
        error_window.configure(bg='black')
    
        # Use the exception message as the error text
        error_text = str(e)
    
        error_label = tk.Label(
            error_window,
            text=error_text,  # Display the exception message
            font=('Helvetica', 14),
            fg='white',
            bg='black',
            wraplength=280,
            justify='center'
        )
        error_label.pack(pady=20)
    
        def close_program():
            error_window.destroy()
            sys.exit()
    
        close_button = tk.Button(
            error_window,
            text="Stäng",
            command=close_program,
            font=('Helvetica', 12),
            bg='red',
            fg='white'
        )
        close_button.pack(pady=10)
    
        error_window.mainloop()

        
from Parkinson_read import läsa_in_parkinson
from Parkinson_transform import transform_park
from Parkinson_ref_con import ref_con
from parkinson_beräkningar import beräkning_park
from plot_regioner_mean_parkinson import regions_z_score_park
from Justera_c_bar_parkison import open_app_window_park


def Parkinson(directory, tracer):
    import sys
    import traceback

    try:
        try:
            normal_data=load_direction()
            norm=check_normaldata(tracer, normal_data)
            if not norm:
                # Raise an exception to trigger the except block
                raise FileNotFoundError("All nomrmaldata hittades inte")
            print("tracer:", tracer)
            initial_window.geometry(f"{int(0.625 * original_screen_width)}x{int(0.32 * original_screen_width)}")
            loading_label = tk.Label(initial_window, font=('Helvetica', int(16 * scale_factor)), fg='white', bg='black')
            
            # Hide all widgets but keep them in memory (don't destroy)
            for widget in initial_window.winfo_children():
                widget.pack_forget()
            
            loading_label.config(text="Läser in")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
        
            try:
                data_4d, first_image_shape, Ref_concentration_time, pixel_spacing, slice_thickness, image_positions = läsa_in_parkinson(directory)
            except Exception as e:
                raise FileNotFoundError("Kunde inte läsa in, kontrollera att du valde rätt mapp")
            
            loading_label.config(text="Omskalar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            data_4d = Downsample(data_4d, first_image_shape)
            loading_label.config(text="Registrerar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            registration_1, registration_2, template_3d = Registrering(data_4d, initial_window, loading_label, original_screen_width, seed=2)
            for widget in initial_window.winfo_children():
                widget.pack_forget()
            plt.close('all')
            loading_label.config(text="Rörelsekorrektion")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            
            corrected_data, motion_parameters_array = rörelse_korrektion(data_4d)
            loading_label.config(text="Transformerar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            small_corrected_data = transform_park(corrected_data, registration_1, registration_2, template_3d)
            loading_label.config(text="Beräknar ref. koncentration")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            Ref_TAC = ref_con(small_corrected_data)
            loading_label.config(text="Beräknar")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            
            BP_reshape_list_pat, K_2_reshape_list_pat, R_I_reshape_list_pat, K_2_p_reshape_list_pat = beräkning_park(
                Ref_TAC, Ref_concentration_time, np.transpose(corrected_data, (3,0,2,1))
            )
            
            BP_reshape_list, K_2_reshape_list, R_I_reshape_list = transform(
                np.transpose(BP_reshape_list_pat, (1,0,2)),
                K_2_reshape_list_pat,
                np.transpose(R_I_reshape_list_pat, (1,0,2)),
                registration_1, registration_2,
                template_3d
            )
            
            loading_label.config(text="Beräknar z-scores")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            regions_first_values_yz_neg_x0, regions_first_values_yz_pos_x1, regions_first_values_yz_pos_x2, regions_first_values_yz_neg_x2, regions_first_values_xz_pos_y2, regions_first_values_xz_neg_y2, z_scores, z_brain_regions, mean_values_regions = regions_z_score_park(R_I_reshape_list, tracer, normal_data)
            Z_brain_flow, z_scores_flow, means_list_flow = SD_corrected_park(R_I_reshape_list, tracer, normal_data)
            
            plt.close('all')
            # gc.collect()
            
            loading_label.config(text="Genererar SSP")
            loading_label.pack(expand=True)
            initial_window.update_idletasks()  # Force update of the label text
            initial_window.update()
            flow_regions_first_values_yz_neg_x0, flow_regions_first_values_yz_pos_x1, flow_regions_first_values_yz_pos_x2, flow_regions_first_values_yz_neg_x2, flow_regions_first_values_xz_pos_y2, flow_regions_first_values_xz_neg_y2 = flow_regions_z_score(Z_brain_flow)
            flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6 = fig_get_flow_regions(flow_regions_first_values_yz_neg_x0, flow_regions_first_values_yz_pos_x1, flow_regions_first_values_yz_pos_x2, flow_regions_first_values_yz_neg_x2, flow_regions_first_values_xz_pos_y2, flow_regions_first_values_xz_neg_y2)
    
    
            
            last_figure_1, last_figure_2, last_figure_3, last_figure_4, last_figure_5, last_figure_6 = fig_get_regions(
                regions_first_values_yz_neg_x0,
                regions_first_values_yz_pos_x1,
                regions_first_values_yz_pos_x2,
                regions_first_values_yz_neg_x2,
                regions_first_values_xz_pos_y2,
                regions_first_values_xz_neg_y2
            )
            
            plt.close('all')
            # gc.collect()
            
            BP_reshape_list_pat_space=BP_reshape_list_pat.copy()
            R_I_reshape_list_pat_space=R_I_reshape_list_pat.copy()
            # BP_reshape_list_pat=np.rot90(np.transpose(BP_reshape_list_pat_space, (2,1,0)))
            # BP_reshape_list_pat=np.transpose(np.rot90(np.rot90(BP_reshape_list_pat_space)), (2,1,0))
            # BP_reshape_list_pat=np.transpose(BP_reshape_list_pat_space, (2,1,0))
            BP_reshape_list_pat=np.transpose(BP_reshape_list_pat_space, (1,2,0))
            
            # R_I_reshape_list_pat=np.rot90(np.transpose(R_I_reshape_list_pat_space, (2,1,0)))
            # R_I_reshape_list_pat=np.transpose(np.rot90(np.rot90(R_I_reshape_list_pat_space)), (2,1,0))
            # R_I_reshape_list_pat=np.transpose(R_I_reshape_list_pat_space, (2,1,0))
            R_I_reshape_list_pat=np.transpose(R_I_reshape_list_pat_space, (1,2,0))
            
            BP_reshape_list_pat, R_I_reshape_list_pat=transform_pat_space(BP_reshape_list_pat, R_I_reshape_list_pat, registration_1,  template_3d)
            

            
            button_pressed = tk.StringVar()
            open_app_window_park(
                initial_window,
                button_pressed,
                BP_reshape_list,
                R_I_reshape_list,
                BP_reshape_list_pat,
                R_I_reshape_list_pat,
                last_figure_1, last_figure_2, last_figure_3,
                last_figure_4, last_figure_5, last_figure_6,
                z_scores, 
                z_brain_regions,
                Z_brain_flow, 
                z_scores_flow,
                flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, 
                flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6,
                Ref_concentration_time,
                tracer, motion_parameters_array,
                means_list_flow, mean_values_regions, 
                directory, 
                BP_reshape_list_pat_space, R_I_reshape_list_pat_space,
                pixel_spacing, slice_thickness, image_positions
            )
            initial_window.wait_variable(button_pressed)
        
            # Retrieve the value set by the button
            response = button_pressed.get()
        
            initial_window.destroy()
        except Exception as e:
            raise FileNotFoundError("Något gick fel, prova igen")

    except Exception as e:
        # Print the traceback to the console for debugging purposes
        traceback.print_exc()
    
        # Destroy the main window if it's still open
        try:
            initial_window.destroy()
        except:
            pass  # If initial_window is already destroyed, ignore the error
    
        # Create a new window for the error message
        error_window = tk.Tk()
        error_window.title("Fel")
        error_window.geometry(f"{int(0.625 * original_screen_width)}x{int(0.32 * original_screen_width)}")
        error_window.configure(bg='black')
    
        # Use the exception message as the error text
        error_text = str(e)
    
        error_label = tk.Label(
            error_window,
            text=error_text,  # Display the exception message
            font=('Helvetica', 14),
            fg='white',
            bg='black',
            wraplength=280,
            justify='center'
        )
        error_label.pack(pady=20)
    
        def close_program():
            error_window.destroy()
            sys.exit()
    
        close_button = tk.Button(
            error_window,
            text="Stäng",
            command=close_program,
            font=('Helvetica', 12),
            bg='red',
            fg='white'
        )
        close_button.pack(pady=10)
    
        error_window.mainloop()
        
def fort3_1():
    loading_label.config(text="klar")
    initial_window.destroy()

def update_window_for_tasks():
    for widget in initial_window.winfo_children():
        widget.pack_forget()
    loading_label.config(text="Läser in")
    loading_label.pack(expand=True)
    def_global(1)
    initial_window.after(100, flow_regions)
    


def change_template_function():
    print("byter template")
    

from parkinson_screen import parkinson_function




# # Add this line to keep the window open and responsive
# initial_window.mainloop()
import os
import json
# from tkinter import filedialog

def get_hidden_file_path():
    """Get the path for the hidden direction file."""
    file_name = ".hidden_direction.txt"
    # Save in the user's home directory for compatibility
    return os.path.join(os.path.expanduser("~"), file_name)

def save_direction(new_direction):
    """Save the selected direction to a hidden file."""
    file_path = get_hidden_file_path()
    try:
        with open(file_path, 'w') as file:
            json.dump({"direction": new_direction}, file)
        if os.name == 'nt':  # Make file hidden on Windows
            import ctypes
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(file_path, FILE_ATTRIBUTE_HIDDEN)
        print(f"Direction saved: {new_direction}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_direction():
    """Load the saved direction from the hidden file."""
    file_path = get_hidden_file_path()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get("direction", None)
        except Exception as e:
            print(f"Error reading the file: {e}")
    return None

def choose_and_save_direction():
    """Open a dialog to choose a directory and save it."""
    directory = filedialog.askdirectory(title="Select a directory")
    if directory:
        save_direction(directory)
        print(f"Saved direction: {directory}")


# Initialize the initial selection window
initial_window = tk.Tk()
initial_window.title("Perfusion and SSP")
initial_window.configure(background='black')
initial_window.geometry(f"{int(0.625 * original_screen_width)}x{int(0.32 * original_screen_width)}")

def initial():
    global loading_label
    global two_files_var
    global types
    global direction_btn

    selected_directory1 = None
    selected_file1 = None
    selected_directory2 = None
    selected_file2 = None
    num_selections = 1
    types = "dicom"

    for widget in initial_window.winfo_children():
        widget.destroy()

    initial_window.geometry(f"{int(0.625 * original_screen_width)}x{int(0.32 * original_screen_width)}")
    loading_label = tk.Label(initial_window, font=('Helvetica', int(16 * scale_factor)), fg='white', bg='black')

    def choose_directory(number):
        global selected_directory1, selected_directory2
        directory = filedialog.askdirectory(title=f"Välj mapp {number}")
        if directory:
            if number == 1:
                selected_directory1 = directory
                choose_dir_btn1.config(text=f"Mapp 1: {directory.split('/')[-1]}")
            else:
                selected_directory2 = directory
                choose_dir_btn2.config(text=f"Mapp 2: {directory.split('/')[-1]}")

    def choose_file(number):
        global selected_file1, selected_file2
        file_path = filedialog.askopenfilename(
            title=f"Välj .inp-fil {number}",
            filetypes=[("INP files", "*.inp")]
        )
        if file_path:
            if number == 1:
                selected_file1 = file_path
                choose_file_btn1.config(text=f"Fil 1: {file_path.split('/')[-1]}")
            else:
                selected_file2 = file_path
                choose_file_btn2.config(text=f"Fil 2: {file_path.split('/')[-1]}")

    def toggle_additional_options():
        state = tk.NORMAL if two_files_var.get() else tk.DISABLED
        choose_dir_btn2.config(state=state)
        choose_file_btn2.config(state=state)

    def choose_nii_file(number):
        global selected_directory1, selected_directory2
        file_path = filedialog.askopenfilename(
            title=f"Välj .nii fil {number}",
            filetypes=[("NIfTI files", "*.nii;*.nii.gz")]
        )
        if file_path:
            if number == 1:
                selected_directory1 = file_path
                choose_dir_btn1.config(text=f"Fil 1: {file_path.split('/')[-1]}")
            else:
                selected_directory2 = file_path
                choose_dir_btn2.config(text=f"Fil 2: {file_path.split('/')[-1]}")

    def change_to_nii():
        global types
        if nii_files_var.get():
            types = "nii"
            choose_dir_btn1.config(command=lambda: choose_nii_file(1))
            choose_dir_btn1.config(text="Välj första .nii filen")
            choose_dir_btn2.config(command=lambda: choose_nii_file(2))
            choose_dir_btn2.config(text="Välj andra .nii filen")
        else:
            types = "dicom"
            choose_dir_btn1.config(command=lambda: choose_directory(1))
            choose_dir_btn1.config(text="Välj första mapp")
            choose_dir_btn2.config(command=lambda: choose_directory(2))
            choose_dir_btn2.config(text="Välj andra mapp")

    # Bottom buttons defined first to anchor them
    bottom_frame = tk.Frame(initial_window, background='black')
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    # "Parkinson" button
    parkinson_btn = tk.Button(bottom_frame, text="FP2I", command=lambda: parkinson_function(initial_window, initial, Parkinson), fg='white', bg='grey')
    parkinson_btn.pack(pady=10)

    # "m/s väg" button for choosing and saving the direction
    direction_btn = tk.Button(bottom_frame, text="Normalmaterial", command=choose_and_save_direction, fg='white', bg='grey')
    direction_btn.pack(pady=10)
    
    # Other components (like directory and file selection) defined after bottom buttons
    two_files_var = tk.BooleanVar(value=False)
    two_files_check = tk.Checkbutton(initial_window, text="Välj två mappar och filer", var=two_files_var, 
                                     command=toggle_additional_options, fg='white', bg='black', selectcolor='grey')
    two_files_check.pack(pady=10)

    # Checkbox for selecting ".nii" files
    nii_files_var = tk.BooleanVar(value=False)
    nii_files_check = tk.Checkbutton(initial_window, text="Använd endast '.nii' filer", var=nii_files_var, 
                                     command=change_to_nii, fg='white', bg='black', selectcolor='grey')
    nii_files_check.pack(pady=10)

    frame_left = tk.Frame(initial_window, background='black')
    frame_right = tk.Frame(initial_window, background='black')

    choose_dir_btn1 = tk.Button(frame_left, text="Välj första mapp", command=lambda: choose_directory(1), fg='white', bg='grey')
    choose_file_btn1 = tk.Button(frame_left, text="Välj första fil", command=lambda: choose_file(1), fg='white', bg='grey')
    choose_dir_btn2 = tk.Button(frame_right, text="Välj andra mapp", command=lambda: choose_directory(2), state=tk.DISABLED, fg='white', bg='grey')
    choose_file_btn2 = tk.Button(frame_right, text="Välj andra fil", command=lambda: choose_file(2), state=tk.DISABLED, fg='white', bg='grey')

    choose_dir_btn1.pack(pady=10)
    choose_file_btn1.pack(pady=10)
    choose_dir_btn2.pack(pady=10)
    choose_file_btn2.pack(pady=10)

    continue_btn = tk.Button(initial_window, text="Fortsätt", command=proceed, fg='white', bg='gray')
    continue_btn.pack(pady=20)

    frame_left.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=20, pady=20)
    frame_right.pack(side=tk.RIGHT, fill=tk.Y, expand=True, padx=20, pady=20)



import traceback

try:
    initial()

    # Add this line to keep the window open and responsive
    initial_window.mainloop()    
except Exception as e:
    # Write the traceback to the temp file
    traceback.print_exc(file=temp_output)
    # Optionally, you can print a message
    print(f"An error occurred: {e}")
finally:
    # Close the temp file
    temp_output.close()
    # Delete the temp file if no exceptions occurred
    if 'e' not in locals():
        os.remove(temp_output_path)

# initial()

# # Add this line to keep the window open and responsive
# initial_window.mainloop() 
