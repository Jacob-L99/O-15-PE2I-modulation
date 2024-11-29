import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt

# Define the custom colormaps
cdict = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.1, 0.5, 0.5),
        (0.2, 0.0, 0.0),
        (0.4, 0.2, 0.2),
        (0.6, 0.0, 0.0),
        (0.8, 1.0, 1.0),
        (1.0, 1.0, 1.0)
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 1.0, 1.0),
        (0.6, 1.0, 1.0),
        (0.8, 1.0, 1.0),
        (1.0, 0.0, 0.0)
    ),
    'blue': (
        (0.0, 0.0, 0.0),
        (0.1, 0.5, 0.5),
        (0.2, 1.0, 1.0),
        (0.4, 1.0, 1.0),
        (0.6, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    )
}

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

# Define the custom blue-white-red colormap
colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1, 1, 1), (1, 0, 0), (0.5, 0, 0)]
n_bins = 10
cmap_name = 'custom_blue_white_red'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def create_custom_figure(first_values_yz_neg_x0, first_values_yz_pos_x1,
                          first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
                          first_values_xz_neg_y2, last_figure_1, last_figure_2, last_figure_3,
                          last_figure_4, last_figure_5, last_figure_6,
                          region_last_figure_1, region_last_figure_2, 
                          region_last_figure_3, region_last_figure_4,
                          region_last_figure_5, region_last_figure_6,
                          flow_region_last_figure_1, flow_region_last_figure_2, 
                          flow_region_last_figure_3, flow_region_last_figure_4,
                          flow_region_last_figure_5, flow_region_last_figure_6,
                          data_set, norm=None, fig=None):
    """Creates or updates the custom figure using the provided data and returns it."""
    print('-----', data_set, '-----------')
    if fig is None:
        fig = Figure(facecolor='black')
    else:
        fig.clear()

    if data_set == 'SSP':
        # Rotate the data arrays if necessary
        data_list = [
            np.rot90(np.rot90(np.rot90(first_values_yz_neg_x0))),
            np.rot90(np.rot90(np.rot90(first_values_yz_pos_x1))),
            np.rot90(np.rot90(np.rot90(first_values_yz_pos_x2))),
            np.rot90(np.rot90(np.rot90(first_values_yz_neg_x2))),
            np.rot90(np.rot90(np.rot90(first_values_xz_pos_y2))),
            np.rot90(np.rot90(np.rot90(first_values_xz_neg_y2)))
        ]

        # Create a figure with 1 row and 6 columns
        gs = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=fig)
        axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
        cbar_ax = fig.add_subplot(gs[0, 6])

        # Define the normalization if not provided
        if norm is None:
            norm = Normalize(vmin=0, vmax=2)

        # Plot each 2D array
        images = []
        for i, data in enumerate(data_list):
            ax = axs[i]
            im = ax.imshow(data, cmap=my_cmap, norm=norm, interpolation='bicubic')
            ax.axis('off')
            ax.set_aspect('equal')
            images.append(im)

        # Add a shared colorbar
        cbar = fig.colorbar(images[-1], cax=cbar_ax)
        cbar.set_label('Value', rotation=270, labelpad=15, color='white', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
        cbar.outline.set_edgecolor('white')

        # Adjust layout
        fig.tight_layout()

        return images  # Return images for updating norm

    elif data_set == 'SSP Z-score':
        # Use the existing figures
        all_figures = [last_figure_1, last_figure_2, last_figure_3, last_figure_4, last_figure_5, last_figure_6]

        # Create a figure with 1 row and 6 columns
        gs = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=fig)
        axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
        cbar_ax = fig.add_subplot(gs[0, 6])

        # Loop over all the figures and plot them in subplots
        for i, figure in enumerate(all_figures):
            ax_new = axs[i]
            for ax_old in figure.axes:
                for img in ax_old.images:
                    blended_image = img.get_array()
                    im = ax_new.imshow(blended_image, cmap=img.get_cmap(), alpha=img.get_alpha())
            ax_new.axis('off')
            ax_new.set_aspect('equal')

        # Add one shared colorbar
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Z-score', rotation=270, labelpad=15, fontsize=8, color='white')
        cbar.ax.tick_params(labelsize=6, colors='white')
        cbar.outline.set_edgecolor('white')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        # Adjust layout
        fig.tight_layout()

        return None  # No images to update norm
    
    elif data_set == 'SSP (neurodegenerativ regioner)':
        # Use the existing figures
        all_figures = [region_last_figure_1, region_last_figure_2, region_last_figure_3, region_last_figure_4, region_last_figure_5, region_last_figure_6]

        # Create a figure with 1 row and 6 columns
        gs = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=fig)
        axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
        cbar_ax = fig.add_subplot(gs[0, 6])

        # Loop over all the figures and plot them in subplots
        for i, figure in enumerate(all_figures):
            ax_new = axs[i]
            for ax_old in figure.axes:
                for img in ax_old.images:
                    blended_image = img.get_array()
                    im = ax_new.imshow(blended_image, cmap=img.get_cmap(), alpha=img.get_alpha())
            ax_new.axis('off')
            ax_new.set_aspect('equal')

        # Add one shared colorbar
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Z-score', rotation=270, labelpad=15, fontsize=8, color='white')
        cbar.ax.tick_params(labelsize=6, colors='white')
        cbar.outline.set_edgecolor('white')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        # Adjust layout
        fig.tight_layout()

        return None  # No images to update norm
    
    elif data_set == 'SSP (flödes regioner)':
        # Use the existing figures
        all_figures = [flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6]

        # Create a figure with 1 row and 6 columns
        gs = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=fig)
        axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
        cbar_ax = fig.add_subplot(gs[0, 6])

        # Loop over all the figures and plot them in subplots
        for i, figure in enumerate(all_figures):
            ax_new = axs[i]
            for ax_old in figure.axes:
                for img in ax_old.images:
                    blended_image = img.get_array()
                    im = ax_new.imshow(blended_image, cmap=img.get_cmap(), alpha=img.get_alpha())
            ax_new.axis('off')
            ax_new.set_aspect('equal')

        # Add one shared colorbar
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Z-score', rotation=270, labelpad=15, fontsize=8, color='white')
        cbar.ax.tick_params(labelsize=6, colors='white')
        cbar.outline.set_edgecolor('white')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        # Adjust layout
        fig.tight_layout()

        return None  # No images to update norm

    else:
        # Handle other datasets if any
        pass



class App(tk.Toplevel):
    def __init__(self, transformed_K_1_1=None, transformed_K_2_1=None, Z_brain=None, K_1_reshape_list_1=None,
                 K_2_reshape_list_1=None, first_values_yz_neg_x0=None, first_values_yz_pos_x1=None,
                 first_values_yz_pos_x2=None, first_values_yz_neg_x2=None, first_values_xz_pos_y2=None,
                 first_values_xz_neg_y2=None,
                 last_figure_1=None, last_figure_2=None, last_figure_3=None, last_figure_4=None,
                 last_figure_5=None, last_figure_6=None,
                 region_last_figure_1=None, region_last_figure_2=None, region_last_figure_3=None,
                 region_last_figure_4=None, region_last_figure_5=None, region_last_figure_6=None,
                 flow_region_last_figure_1=None, flow_region_last_figure_2=None, flow_region_last_figure_3=None,
                 flow_region_last_figure_4=None, flow_region_last_figure_5=None, flow_region_last_figure_6=None,
                 z_score=None, z_scores_flow=None, z_brain_regions=None, Cerebellum_mean_k1=None,
                 Cerebellum_mean_k2=None, 
                 first_values_yz_neg_x0_rel=None, first_values_yz_pos_x1_rel=None, 
                 first_values_yz_pos_x2_rel=None, first_values_yz_neg_x2_rel=None, first_values_xz_pos_y2_rel=None, 
                 first_values_xz_neg_y2_rel=None, 
                 last_figure_1_rel=None, last_figure_2_rel=None, last_figure_3_rel=None, 
                 last_figure_4_rel=None, last_figure_5_rel=None, last_figure_6_rel=None, 
                 region_last_figure_1_rel=None, region_last_figure_2_rel=None, region_last_figure_3_rel=None,
                 region_last_figure_4_rel=None, region_last_figure_5_rel=None, region_last_figure_6_rel=None, 
                 flow_region_last_figure_1_rel=None, flow_region_last_figure_2_rel=None, flow_region_last_figure_3_rel=None, 
                 flow_region_last_figure_4_rel=None, flow_region_last_figure_5_rel=None, flow_region_last_figure_6_rel=None, 
                 z_score_rel=None, z_scores_flow_rel=None,
                 Z_brain_rel=None, z_brain_regions_rel=None,
                 means_list_k1=None ,mean_values_k1=None,
                 means_list_k1_ref=None ,mean_values_k1_ref=None,
                 motion_parameters_array_1=None, dicom_name=None,
                 K_1_reshape_list_1_pat=None, K_2_reshape_list_1_pat=None,
                 pixel_spacing=None, slice_thickness=None, image_positions=None,
                 master=None, button_var=None):
        """
        Initializes the App window.

        Args:
            master (tk.Tk or tk.Toplevel): The parent window.
            button_var (tk.StringVar): Variable to capture which button is pressed.
        """
        super().__init__(master)
        self.button_var = button_var  # Store the reference to the StringVar
        self.title("Adjust Colormap Scale")
        self.configure(bg="black")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # **Step 1: Configure the grid layout for the App window**
        # Define three rows: slider_frame (row 0), main_frame (row 1), bottom_frame (row 2)
        self.grid_rowconfigure(0, weight=0)  # Non-expandable row for slider
        self.grid_rowconfigure(1, weight=1)  # Expandable row for main content
        self.grid_rowconfigure(2, weight=0)  # Non-expandable row for buttons
        self.grid_columnconfigure(0, weight=1)  # Single column that expands

        # **Step 2: Create and Position the Frames**

        # Frame for the slice slider (row 0)
        slider_frame = ttk.Frame(self, style="TFrame")
        slider_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        slider_frame.grid_columnconfigure(0, weight=0)
        slider_frame.grid_columnconfigure(1, weight=1)

        # Main frame to hold canvas and other widgets (row 1)
        self.main_frame = ttk.Frame(self, style="TFrame")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        # Adjusted column weights to redistribute space
        self.main_frame.grid_columnconfigure(0, weight=1)  # Side column (controls)
        self.main_frame.grid_columnconfigure(1, weight=4)  # Middle column (figures)
        self.main_frame.grid_columnconfigure(2, weight=1)  # Side column (sliders)
        # Rows configuration
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)

        # Bottom frame for buttons (row 2)
        bottom_frame = ttk.Frame(self, style="TFrame")
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        # **Step 3: Configure the Slider Frame**

        # Slice slider label
        ttk.Label(slider_frame, text="Slice Index", background="black", foreground="white").grid(row=0, column=0, padx=5)

        # Slice slider
        self.slice_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=100,  # Temporary value; will be updated later based on data
            orient='horizontal',
            command=self.update_slices,
            style="TScale",
            length=200
        )
        self.slice_slider.set(63)  # Initial slice index
        self.slice_slider.grid(row=0, column=1, sticky="ew", padx=5)

        # **Step 4: Store Data Arrays as Instance Variables**
        # (Your existing data handling code remains here)
        self.first_values_yz_neg_x0 = first_values_yz_neg_x0
        self.first_values_yz_pos_x1 = np.flip(first_values_yz_pos_x1, axis=0)
        self.first_values_yz_pos_x2 = np.flip(first_values_yz_pos_x2, axis=0)
        self.first_values_yz_neg_x2 = first_values_yz_neg_x2
        self.first_values_xz_pos_y2 = first_values_xz_pos_y2
        self.first_values_xz_neg_y2 = first_values_xz_neg_y2
        self.last_figure_1 = last_figure_1
        self.last_figure_2 = last_figure_2
        self.last_figure_3 = last_figure_3
        self.last_figure_4 = last_figure_4
        self.last_figure_5 = last_figure_5
        self.last_figure_6 = last_figure_6
        self.region_last_figure_1 = region_last_figure_1
        self.region_last_figure_2 = region_last_figure_2
        self.region_last_figure_3 = region_last_figure_3
        self.region_last_figure_4 = region_last_figure_4
        self.region_last_figure_5 = region_last_figure_5
        self.region_last_figure_6 = region_last_figure_6
        self.flow_region_last_figure_1 = flow_region_last_figure_1
        self.flow_region_last_figure_2 = flow_region_last_figure_2
        self.flow_region_last_figure_3 = flow_region_last_figure_3
        self.flow_region_last_figure_4 = flow_region_last_figure_4
        self.flow_region_last_figure_5 = flow_region_last_figure_5
        self.flow_region_last_figure_6 = flow_region_last_figure_6
        
        self.z_score = z_score
        self.z_scores_flow = z_scores_flow
        self.z_score_rel = z_score_rel
        self.z_scores_flow_rel = z_scores_flow_rel

        self.means_list_k1=means_list_k1
        self.mean_values_k1=mean_values_k1
        self.means_list_k1_ref=means_list_k1_ref 
        self.mean_values_k1_ref=mean_values_k1_ref
        self.motion_parameters_array_1=motion_parameters_array_1
        self.dicom_name=dicom_name
        self.pixel_spacing=pixel_spacing
        self.slice_thickness=slice_thickness
        self.image_positions=image_positions
        
        self.first_values_yz_neg_x0_rel = first_values_yz_neg_x0_rel
        self.first_values_yz_pos_x1_rel = np.flip(first_values_yz_pos_x1_rel, axis=0)
        self.first_values_yz_pos_x2_rel = first_values_yz_pos_x2_rel
        self.first_values_yz_neg_x2_rel = np.flip(first_values_yz_neg_x2_rel, axis=0)
        self.first_values_xz_pos_y2_rel = first_values_xz_pos_y2_rel
        self.first_values_xz_neg_y2_rel = first_values_xz_neg_y2_rel
        self.last_figure_1_rel = last_figure_1_rel
        self.last_figure_2_rel = last_figure_2_rel
        self.last_figure_3_rel = last_figure_3_rel
        self.last_figure_4_rel = last_figure_4_rel
        self.last_figure_5_rel = last_figure_5_rel
        self.last_figure_6_rel = last_figure_6_rel
        self.region_last_figure_1_rel = region_last_figure_1_rel
        self.region_last_figure_2_rel = region_last_figure_2_rel
        self.region_last_figure_3_rel = region_last_figure_3_rel
        self.region_last_figure_4_rel = region_last_figure_4_rel
        self.region_last_figure_5_rel = region_last_figure_5_rel
        self.region_last_figure_6_rel = region_last_figure_6_rel
        self.flow_region_last_figure_1_rel = flow_region_last_figure_1_rel
        self.flow_region_last_figure_2_rel = flow_region_last_figure_2_rel
        self.flow_region_last_figure_3_rel = flow_region_last_figure_3_rel
        self.flow_region_last_figure_4_rel = flow_region_last_figure_4_rel
        self.flow_region_last_figure_5_rel = flow_region_last_figure_5_rel
        self.flow_region_last_figure_6_rel = flow_region_last_figure_6_rel

        # **Step 5: Style Configuration for Dark Theme**
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", background="grey", foreground="white")
        style.configure("TFrame", background="black")
        style.configure("TLabel", background="black", foreground="white")
        style.configure("TCombobox", background="grey", foreground="white", fieldbackground="grey")
        style.configure("TScale", background="black", troughcolor="grey", sliderlength=30)

        # Create figures
        self.fig1 = Figure(facecolor='black')
        gs1 = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=self.fig1)
        self.ax1 = [self.fig1.add_subplot(gs1[0, i]) for i in range(6)]
        self.cbar_ax1 = self.fig1.add_subplot(gs1[0, 6])

        self.fig2 = Figure(facecolor='black')
        gs2 = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=self.fig2)
        self.ax2 = [self.fig2.add_subplot(gs2[0, i]) for i in range(6)]
        self.cbar_ax2 = self.fig2.add_subplot(gs2[0, 6])

        self.fig3 = Figure(facecolor='black')
        gs3 = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=self.fig3)
        self.ax3 = [self.fig3.add_subplot(gs3[0, i]) for i in range(6)]
        self.cbar_ax3 = self.fig3.add_subplot(gs3[0, 6])

        # Initialize the fourth figure with datasets
        self.datasets_row4 = ['SSP', 'SSP Z-score', 'SSP (neurodegenerativ regioner)', 'SSP (flödes regioner)']
        self.fig4 = Figure(facecolor='black')
        self.images_row4 = {}

        # Initialize normalization object for SSP
        self.norm_row4 = Normalize(vmin=0, vmax=2)

        # Set the default dataset
        self.current_dataset_row4 = 'SSP'  # Default dataset

        # Create the initial figure for row 4
        images = create_custom_figure(
            self.first_values_yz_neg_x0, self.first_values_yz_pos_x1, self.first_values_yz_pos_x2,
            self.first_values_yz_neg_x2, self.first_values_xz_pos_y2, self.first_values_xz_neg_y2,
            self.last_figure_1, self.last_figure_2, self.last_figure_3,
            self.last_figure_4, self.last_figure_5, self.last_figure_6,
            self.region_last_figure_1, self.region_last_figure_2,
            self.region_last_figure_3, self.region_last_figure_4,
            self.region_last_figure_5, self.region_last_figure_6,
            self.flow_region_last_figure_1, self.flow_region_last_figure_2,
            self.flow_region_last_figure_3, self.flow_region_last_figure_4,
            self.flow_region_last_figure_5, self.flow_region_last_figure_6,
            self.current_dataset_row4, norm=self.norm_row4, fig=self.fig4
        )
        self.images_row4[self.current_dataset_row4] = images

        # Load or generate test datasets for the first three rows
        try:
            self.data1 = np.rot90(transformed_K_1_1)
            self.data1_rel=self.data1/Cerebellum_mean_k1
            self.data2 = np.rot90(transformed_K_2_1)
            self.data2_rel=self.data2/Cerebellum_mean_k2 #k2
            self.data3 = self.data1 / self.data2  # For the third row
            self.data3_rel = self.data1_rel / self.data2_rel
            self.data4 = np.rot90(Z_brain)  # Z-score data
            self.data4_reg=np.rot90(z_brain_regions)
            self.data4_rel = np.rot90(Z_brain_rel)  # Z-score data
            self.data4_reg_rel=np.rot90(z_brain_regions_rel)

            # Load data5, data6, data7 (patient space)
            self.data5 = np.flip(np.rot90(K_1_reshape_list_1), axis=1)
            self.data5_rel=self.data5/Cerebellum_mean_k1
            self.data6 = np.flip(np.rot90(K_2_reshape_list_1), axis=1)
            self.data6_rel=self.data6/Cerebellum_mean_k2
            self.data7 = self.data5 / self.data6
            self.data7_rel = self.data5_rel / self.data6_rel
            
            self.data5_pat = np.rot90(np.rot90(np.rot90(K_1_reshape_list_1_pat)))
            self.data6_pat = np.rot90(np.rot90(np.rot90(K_2_reshape_list_1_pat)))

            slices_diff = self.data1.shape[-1] - self.data5.shape[-1]
            if slices_diff > 0:
                pad_width = ((0, 0), (0, 0), (0, slices_diff))
                self.data5 = np.pad(self.data5, pad_width, mode='constant', constant_values=0)
                self.data5_rel = np.pad(self.data5_rel, pad_width, mode='constant', constant_values=0)
                self.data6 = np.pad(self.data6, pad_width, mode='constant', constant_values=0)
                self.data6_rel = np.pad(self.data6_rel, pad_width, mode='constant', constant_values=0)
                self.data7 = np.pad(self.data7, pad_width, mode='constant', constant_values=0)
                self.data7_rel = np.pad(self.data7_rel, pad_width, mode='constant', constant_values=0)
            elif slices_diff < 0:
                # If data5 has more slices, truncate it
                self.data5 = self.data5[..., :self.data1.shape[-1]]
                self.data5_rel = self.data5_rel[..., :self.data1.shape[-1]]
                self.data6 = self.data6[..., :self.data1.shape[-1]]
                self.data6_rel = self.data6_rel[..., :self.data1.shape[-1]]
                self.data7 = self.data7[..., :self.data1.shape[-1]]
                self.data7_rel = self.data7_rel[..., :self.data1.shape[-1]]

        except FileNotFoundError as e:
            print(f"Data file not found: {e}")
            messagebox.showerror("Error", f"Data file not found: {e}")
            self.destroy()
            return

        # Initialize slice index and maximum slices
        self.slice_index = 63  # initial slice index
        self.max_slices = self.data1.shape[-1]

        # Update the slice slider maximum value based on data
        self.slice_slider.configure(to=self.max_slices - 1)
        self.slice_slider.set(self.slice_index)

        slice_indices = self.get_slice_indices()
        self.channel_data1 = [self.data1[..., idx] for idx in slice_indices]
        self.channel_data1_rel = [self.data1_rel[..., idx] for idx in slice_indices]
        self.channel_data2 = [self.data2[..., idx] for idx in slice_indices]
        self.channel_data2_rel = [self.data2_rel[..., idx] for idx in slice_indices]
        self.channel_data3 = [self.data3[..., idx] for idx in slice_indices]
        self.channel_data3_rel = [self.data3_rel[..., idx] for idx in slice_indices]
        self.channel_data4 = [self.data4[..., idx] for idx in slice_indices]
        self.channel_data4_reg = [self.data4_reg[..., idx] for idx in slice_indices]
        self.channel_data4_rel = [self.data4_rel[..., idx] for idx in slice_indices]
        self.channel_data4_reg_rel = [self.data4_reg_rel[..., idx] for idx in slice_indices]

        self.channel_data5 = [self.data5[..., idx] for idx in slice_indices]
        self.channel_data5_rel = [self.data5_rel[..., idx] for idx in slice_indices]
        self.channel_data6 = [self.data6[..., idx] for idx in slice_indices]
        self.channel_data6_rel = [self.data6_rel[..., idx] for idx in slice_indices]
        self.channel_data7 = [self.data7[..., idx] for idx in slice_indices]
        self.channel_data7_rel = [self.data7_rel[..., idx] for idx in slice_indices]

        # Initialize normalization objects
        self.norm = Normalize(vmin=0, vmax=2)
        self.norm3 = Normalize(vmin=0, vmax=2)
        self.norm4 = Normalize(vmin=-5, vmax=5)  # For Z-score data

        # Define colormap
        self.cmap = my_cmap
        self.cmap_BWR = custom_cmap

        # Initialize image plots with the datasets
        self.images1 = [
            self.ax1[i].imshow(self.channel_data1[i], interpolation='bicubic', norm=self.norm, cmap=self.cmap)
            for i in range(6)
        ]
        self.images2 = [
            self.ax2[i].imshow(self.channel_data2[i], interpolation='bicubic', norm=self.norm, cmap=self.cmap)
            for i in range(6)
        ]
        self.images3 = [
            self.ax3[i].imshow(self.channel_data3[i], interpolation='nearest', norm=self.norm3, cmap=my_cmap)
            for i in range(6)
        ]

        # Remove axis from each plot and set facecolor
        for ax in self.ax1 + self.ax2 + self.ax3:
            ax.set_facecolor('black')
            ax.tick_params(axis='both', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.axis('off')
            # Fix aspect ratio
            ax.set_aspect('equal')

        # Create colorbars for the first three plots using the seventh subplot
        self.cbar1 = self.fig1.colorbar(self.images1[-1], cax=self.cbar_ax1)
        self.cbar1.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
        self.cbar1.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
        self.cbar1.outline.set_edgecolor('white')

        self.cbar2 = self.fig2.colorbar(self.images2[-1], cax=self.cbar_ax2)
        self.cbar2.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
        self.cbar2.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
        self.cbar2.outline.set_edgecolor('white')

        self.cbar3 = self.fig3.colorbar(self.images3[-1], cax=self.cbar_ax3)
        self.cbar3.set_label('Value', rotation=270, labelpad=8, color='white', fontsize=8)
        self.cbar3.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
        self.cbar3.outline.set_edgecolor('white')

        # Embed the figures in the Tk window
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.main_frame)
        self.canvas_widget1 = self.canvas1.get_tk_widget()
        self.canvas_widget1.grid(row=0, column=1, sticky="nsew")

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.main_frame)
        self.canvas_widget2 = self.canvas2.get_tk_widget()
        self.canvas_widget2.grid(row=1, column=1, sticky="nsew")

        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.main_frame)
        self.canvas_widget3 = self.canvas3.get_tk_widget()
        self.canvas_widget3.grid(row=2, column=1, sticky="nsew")

        # For row 4, use the figure created based on the current dataset
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.main_frame)
        self.canvas_widget4 = self.canvas4.get_tk_widget()
        self.canvas_widget4.grid(row=3, column=1, sticky="nsew")

        # Frame for the sliders
        sliders_frame = ttk.Frame(self.main_frame, width=50, style="TFrame")
        sliders_frame.grid(row=0, column=2, rowspan=4, sticky="ns", padx=10, pady=10)
        # Increased weight of column 2 in grid configuration
        self.main_frame.grid_columnconfigure(2, weight=1)

        # Create a vertical slider for the upper limit adjustment
        self.slider_vmax = ttk.Scale(
            sliders_frame,
            from_=2,
            to=0.01,
            orient='vertical',
            command=self.update_vmax,
            style="TScale"
        )
        self.slider_vmax.set(2)
        ttk.Label(sliders_frame, text="Övre gräns", background="black", foreground="white").pack()
        self.slider_vmax.pack(expand=True, fill=tk.Y, pady=(0, 10))

        # Create frame for the first row controls
        row1_controls = ttk.Frame(self.main_frame, style="TFrame")
        row1_controls.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        # Increased weight of column 0 in grid configuration
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Variable to store the space selection
        self.space_var = tk.StringVar()
        self.space_var.set('MNI space')  # Default value

        # Dropdown menu to select the space (MNI or PAT)
        self.space_dropdown = ttk.Combobox(
            row1_controls,
            textvariable=self.space_var,
            state='readonly',
            width=25,
            style="TCombobox"
        )
        self.space_dropdown['values'] = ['MNI space', 'MNI space relative', 'PAT space', 'PAT space relative']
        self.space_dropdown.current(0)
        self.space_dropdown.grid(row=0, column=0, padx=5, pady=5)

        # Dropdown menu to select the dataset for the first row
        self.plot_var1 = tk.StringVar()
        self.plot_dropdown1 = ttk.Combobox(
            row1_controls,
            textvariable=self.plot_var1,
            state='readonly',
            width=25,
            style="TCombobox"
        )
        self.plot_dropdown1['values'] = ['Perfusion']
        self.plot_dropdown1.current(0)
        self.plot_dropdown1.grid(row=1, column=0, padx=5, pady=5)

        # Button to show the first row large
        self.show_large_button1 = ttk.Button(
            row1_controls,
            text="Visa stort",
            command=lambda: self.show_large_image(self.ax1, self.cbar1),
            style="TButton"
        )
        self.show_large_button1.grid(row=2, column=0, padx=5, pady=5)

        # Create frame for the second row controls
        row2_controls = ttk.Frame(self.main_frame, style="TFrame")
        row2_controls.grid(row=1, column=0, sticky="ns", padx=5, pady=5)

        # Dropdown menu to select the dataset for the second row
        self.plot_var2 = tk.StringVar()
        self.plot_dropdown2 = ttk.Combobox(
            row2_controls,
            textvariable=self.plot_var2,
            state='readonly',
            width=25,
            style="TCombobox"
        )
        self.plot_dropdown2['values'] = ['Flow-out rate']
        self.plot_dropdown2.current(0)
        self.plot_dropdown2.grid(row=0, column=0, padx=5, pady=5)

        # Button to show the second row large
        self.show_large_button2 = ttk.Button(
            row2_controls,
            text="Visa stort",
            command=lambda: self.show_large_image(self.ax2, self.cbar2),
            style="TButton"
        )
        self.show_large_button2.grid(row=1, column=0, padx=5, pady=5)

        # Create frame for the third row controls
        row3_controls = ttk.Frame(self.main_frame, style="TFrame")
        row3_controls.grid(row=2, column=0, sticky="ns", padx=5, pady=5)

        # Dropdown menu to select the dataset for the third row
        self.plot_var3 = tk.StringVar()
        self.plot_dropdown3 = ttk.Combobox(
            row3_controls,
            textvariable=self.plot_var3,
            state='readonly',
            width=25,
            style="TCombobox"
        )
        self.plot_dropdown3['values'] = ['Volume of Distribution', 'Z-score (flödes regioner)', 'Z-score (neurodegenerativ regioner)']
        self.plot_dropdown3.current(0)
        self.plot_dropdown3.grid(row=0, column=0, padx=5, pady=5)

        # Button to show the third row large
        self.show_large_button3 = ttk.Button(
            row3_controls,
            text="Visa stort",
            command=lambda: self.show_large_image(self.ax3, self.cbar3),
            style="TButton"
        )
        self.show_large_button3.grid(row=1, column=0, padx=5, pady=5)

        # Create frame for the fourth row controls
        row4_controls = ttk.Frame(self.main_frame, style="TFrame")
        row4_controls.grid(row=3, column=0, sticky="ns", padx=5, pady=5)

        # Dropdown menu to select the dataset for the fourth row
        self.plot_var4 = tk.StringVar()
        self.plot_dropdown4 = ttk.Combobox(
            row4_controls,
            textvariable=self.plot_var4,
            state='readonly',
            width=25,
            style="TCombobox"
        )
        self.plot_dropdown4['values'] = self.datasets_row4
        self.plot_dropdown4.current(0)
        self.plot_dropdown4.grid(row=0, column=0, padx=5, pady=5)

        # Button to show the fourth row large
        self.show_large_button4 = ttk.Button(
            row4_controls,
            text="Visa stort",
            command=self.show_large_image_fig4,
            style="TButton"
        )
        self.show_large_button4.grid(row=1, column=0, padx=5, pady=5)

        self.fig1.tight_layout()
        self.fig2.tight_layout()
        self.fig3.tight_layout()
        self.fig4.tight_layout()
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()

        # Bind dropdown change events
        self.plot_dropdown1.bind('<<ComboboxSelected>>', self.change_first_row)
        self.plot_dropdown2.bind('<<ComboboxSelected>>', self.change_second_row)
        self.plot_dropdown3.bind('<<ComboboxSelected>>', self.change_third_row)
        self.plot_dropdown4.bind('<<ComboboxSelected>>', self.change_fourth_row)
        self.space_dropdown.bind('<<ComboboxSelected>>', self.change_space)

        # Inner frame to center buttons
        button_frame = ttk.Frame(bottom_frame, style="TFrame")
        button_frame.pack(side=tk.TOP, expand=True)

        # Create buttons for the button frame
        self.button_left = ttk.Button(
            button_frame,
            text="Spara som .nii",
            command=self.left_button_action,
            style="TButton"
        )
        self.button_left.pack(side=tk.LEFT, padx=20)

        self.button_middle = ttk.Button(
            button_frame,
            text="Spara som dicom",
            command=self.middle_button_action,
            style="TButton"
        )
        self.button_middle.pack(side=tk.LEFT, padx=20)

        self.button_right = ttk.Button(
            button_frame,
            text="Skapa PDF",
            command=self.right_button_action,
            style="TButton"
        )
        self.button_right.pack(side=tk.LEFT, padx=20)

        # **Add Window Management Commands Here**
        self.deiconify()           # Ensure the window is not minimized or hidden
        self.lift()                # Bring the window to the top
        self.focus_force()         # Focus on the App window
        self.grab_set()            # Make the App window modal (optional but recommended)
        self.update_idletasks()    # Process any pending idle tasks
        self.update()              # Force an update to render all widgets

        # **Release the grab to make the window interactive**
        self.grab_release()

    def get_slice_indices(self):
        """Return the list of slice indices to display based on the current slice index."""
        idx = int(self.slice_index)
        delta = 7  # Number of slices apart
        indices = []

        # Calculate previous index
        first_idx = idx - delta*5
        if first_idx >= 0:
            indices.insert(0, first_idx)
        else:
            indices.insert(0, 0)

        second_idx = idx - delta*3
        if second_idx >= 0:
            indices.insert(1, second_idx)
        else:
            indices.insert(1, 0)

        third_idx = idx - delta
        if third_idx >= 0:
            indices.insert(2, third_idx)
        else:
            indices.insert(2, 0)

        # Calculate next index
        fourth_idx = idx + delta
        if fourth_idx < self.max_slices:
            indices.insert(3, fourth_idx)
        else:
            indices.insert(3, self.max_slices - 1)

        fifth_idx = idx + delta*3
        if fifth_idx < self.max_slices:
            indices.insert(4, fifth_idx)
        else:
            indices.insert(4, self.max_slices - 1)

        sixth_idx = idx + delta*5
        if sixth_idx < self.max_slices:
            indices.insert(5, sixth_idx)
        else:
            indices.insert(5, self.max_slices - 1)

        return indices

    def update_slices(self, value):
        """Update the displayed slices when the slider is moved."""
        self.slice_index = int(float(value))
        print(f"Updating slices to index: {self.slice_index}")
        slice_indices = self.get_slice_indices()

        # Update channel data
        self.channel_data1 = [self.data1[..., idx] for idx in slice_indices]
        self.channel_data1_rel = [self.data1_rel[..., idx] for idx in slice_indices]
        self.channel_data2 = [self.data2[..., idx] for idx in slice_indices]
        self.channel_data2_rel = [self.data2_rel[..., idx] for idx in slice_indices]
        self.channel_data3 = [self.data3[..., idx] for idx in slice_indices]
        self.channel_data3_rel = [self.data3_rel[..., idx] for idx in slice_indices]
        self.channel_data4 = [self.data4[..., idx] for idx in slice_indices]
        self.channel_data4_reg = [self.data4_reg[..., idx] for idx in slice_indices]
        self.channel_data4_rel = [self.data4_rel[..., idx] for idx in slice_indices]
        self.channel_data4_reg_rel = [self.data4_reg_rel[..., idx] for idx in slice_indices]

        self.channel_data5 = [self.data5[..., idx] for idx in slice_indices]
        self.channel_data5_rel = [self.data5_rel[..., idx] for idx in slice_indices]
        self.channel_data6 = [self.data6[..., idx] for idx in slice_indices]
        self.channel_data6_rel = [self.data6_rel[..., idx] for idx in slice_indices]
        self.channel_data7 = [self.data7[..., idx] for idx in slice_indices]
        self.channel_data7_rel = [self.data7_rel[..., idx] for idx in slice_indices]

        # Update images
        self.change_first_row(None)
        self.change_second_row(None)
        self.change_third_row(None)
        self.change_fourth_row(None)

    def update_vmax(self, value):
        """Adjust the colormap normalization based on the upper limit slider."""
        global vmax
        vmax = float(value)
        print(f"Updating vmax to: {vmax}")
        self.norm.vmax = vmax
        self.norm_row4.vmax = vmax  # Update norm for SSP figures
        self.update_plots()

    def update_plots(self):
        """Update the normalization across plots that respond to the slider."""
        print("Updating plots with new normalization.")
        # Update images that use self.norm
        for img in self.images1:
            img.set_norm(self.norm)
        selection2 = self.plot_var2.get()
        if selection2 == 'Flow-out rate':
            for img in self.images2:
                img.set_norm(self.norm)
            self.cbar2.update_normal(self.images2[-1])
        # Update colorbars
        self.cbar1.update_normal(self.images1[-1])
        self.canvas1.draw()
        self.canvas2.draw()
        # Update SSP images if applicable
        if self.current_dataset_row4 == 'SSP':
            images = self.images_row4['SSP']
            for img in images:
                img.set_norm(self.norm_row4)
            self.canvas4.draw()

    def on_close(self):
        """Handle the window close event."""
        print("App window is closing.")
        if self.button_var:
            self.button_var.set("Closed")  # Optionally set a value indicating closure
        self.destroy()
        if self.master:
            self.master.quit()  # Ensure the application exits when App window is closed

    def left_button_action(self):
        """Action for the left button."""
        from Save_as_nifyt_dicom import save_as_nifti
        array1 = np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data1, axes=(1, 2)), axes=(1, 2)), (1, 0, 2))))
        array2 = np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data2, axes=(1, 2)), axes=(1, 2)), (1, 0, 2))))
        # array3 = np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data3, axes=(1, 2)), axes=(1, 2)), (1, 0, 2))))

        affine_matrix = np.eye(4)
        arrays = [array1, array2]
        filenames = ['K_1_MNI.nii', 'K_2_MNI.nii']
        
        array_3=np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data5_pat, axes=(1, 2)), axes=(1, 2)), (1, 0, 2))))
        array_4=np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data6_pat, axes=(1, 2)), axes=(1, 2)), (1, 0, 2))))
        arrays_pat=[array_3, array_4]
        filenames_pat = ['K_1_pat.nii', 'K_2_pat.nii']
        
        
        save_as_nifti(self.dicom_name, arrays, filenames, arrays_pat, filenames_pat, self.pixel_spacing, self.slice_thickness, self.image_positions)
        # self.destroy()

    def middle_button_action(self):
        """Action for the middle button."""
        if self.button_var:
            self.button_var.set("Middle")
        # self.destroy()

    def right_button_action(self):
        """Action for the right button."""
        from PDF_park import PDF_water
        PDF_water(self.dicom_name, self.data1, self.data2, self.data3, self.data4, self.data4_reg, self.first_values_yz_neg_x0, self.first_values_yz_pos_x1,
                  self.first_values_yz_pos_x2, self.first_values_yz_neg_x2, self.first_values_xz_pos_y2,
                  self.first_values_xz_neg_y2,
                  self.last_figure_1, self.last_figure_2, self.last_figure_3,
                  self.last_figure_4, self.last_figure_5, self.last_figure_6,
                  self.region_last_figure_1, self.region_last_figure_2, self.region_last_figure_3, 
                  self.region_last_figure_4, self.region_last_figure_5, self.region_last_figure_6, 
                  self.flow_region_last_figure_1, self.flow_region_last_figure_2, self.flow_region_last_figure_3, 
                  self.flow_region_last_figure_4, self.flow_region_last_figure_5, self.flow_region_last_figure_6,
                  self.z_score, self.z_scores_flow, vmax, self.means_list_k1 ,self.mean_values_k1, 
                  self.data1_rel, self.data2_rel, self.data3_rel, self.data4_rel, self.data4_reg_rel, self.first_values_yz_neg_x0_rel, self.first_values_yz_pos_x1_rel,
                  self.first_values_yz_pos_x2_rel, self.first_values_yz_neg_x2_rel, self.first_values_xz_pos_y2_rel,
                  self.first_values_xz_neg_y2_rel,
                  self.last_figure_1_rel, self.last_figure_2_rel, self.last_figure_3_rel,
                  self.last_figure_4_rel, self.last_figure_5_rel, self.last_figure_6_rel,
                  self.region_last_figure_1_rel, self.region_last_figure_2_rel, self.region_last_figure_3_rel, 
                  self.region_last_figure_4_rel, self.region_last_figure_5_rel, self.region_last_figure_6_rel, 
                  self.flow_region_last_figure_1_rel, self.flow_region_last_figure_2_rel, self.flow_region_last_figure_3_rel, 
                  self.flow_region_last_figure_4_rel, self.flow_region_last_figure_5_rel, self.flow_region_last_figure_6_rel,
                  self.z_score_rel, self.z_scores_flow_rel, self.means_list_k1_ref ,self.mean_values_k1_ref,
                  self.motion_parameters_array_1)

    def change_first_row(self, event):
        """Change the plot based on the selection from the first dropdown menu."""
        selection = self.plot_var1.get()
        space = self.space_var.get()
        print(f"First row selection changed to: {selection} in space {space}")
        if selection == "Perfusion":
            if space == 'MNI space':
                for i in range(6):
                    self.images1[i].set_data(self.channel_data1[i])
            elif space == 'PAT space':
                for i in range(6):
                    self.images1[i].set_data(self.channel_data5[i])
            elif space == 'MNI space relative':
                for i in range(6):
                    self.images1[i].set_data(self.channel_data1_rel[i])
            elif space == 'PAT space relative':
                for i in range(6):
                    self.images1[i].set_data(self.channel_data5_rel[i])

        self.canvas1.draw()

    def change_second_row(self, event):
        """Change the plot based on the selection from the second dropdown menu."""
        selection = self.plot_var2.get()
        space = self.space_var.get()
        print(f"Second row selection changed to: {selection} in space {space}")
        if selection == 'Flow-out rate':
            if space == 'MNI space':
                for i in range(6):
                    self.images2[i].set_data(self.channel_data2[i])
                    self.images2[i].set_cmap(self.cmap)
                    self.images2[i].set_norm(self.norm)
                self.cbar2.update_normal(self.images2[-1])
                self.cbar2.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space == 'PAT space':
                for i in range(6):
                    self.images2[i].set_data(self.channel_data6[i])
                    self.images2[i].set_cmap(self.cmap)
                    self.images2[i].set_norm(self.norm)
                self.cbar2.update_normal(self.images2[-1])
                self.cbar2.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space == 'MNI space relative':
                for i in range(6):
                    self.images2[i].set_data(self.channel_data2_rel[i])
                    self.images2[i].set_cmap(self.cmap)
                    self.images2[i].set_norm(self.norm)
                self.cbar2.update_normal(self.images2[-1])
                self.cbar2.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space== 'PAT space relative':
                for i in range(6):
                    self.images2[i].set_data(self.channel_data6_rel[i])
                    self.images2[i].set_cmap(self.cmap)
                    self.images2[i].set_norm(self.norm)
                self.cbar2.update_normal(self.images2[-1])
                self.cbar2.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
                
        self.canvas2.draw()

    def change_third_row(self, event):
        """Change the plot based on the selection from the third dropdown menu."""
        selection = self.plot_var3.get()
        space = self.space_var.get()
        print(f"Third row selection changed to: {selection} in space {space}")
        if selection == 'Volume of Distribution':
            if space == 'MNI space':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data3[i])
                    self.images3[i].set_cmap(my_cmap)
                    self.images3[i].set_norm(self.norm3)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Value', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space == 'PAT space':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data7[i])
                    self.images3[i].set_cmap(my_cmap)
                    self.images3[i].set_norm(self.norm3)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Value', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space == 'MNI space relative':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data3_rel[i])
                    self.images3[i].set_cmap(my_cmap)
                    self.images3[i].set_norm(self.norm3)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Value', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space == 'PAT space relative':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data7_rel[i])
                    self.images3[i].set_cmap(my_cmap)
                    self.images3[i].set_norm(self.norm3)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Value', rotation=270, labelpad=8, color='white', fontsize=8)
        elif selection == 'Z-score (flödes regioner)':
            if space == 'MNI space' or space == 'PAT space':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data4[i])
                    self.images3[i].set_cmap(custom_cmap)
                    self.images3[i].set_norm(self.norm4)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Z-score', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space == 'MNI space relative' or space == 'PAT space relative':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data4_rel[i])
                    self.images3[i].set_cmap(custom_cmap)
                    self.images3[i].set_norm(self.norm4)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Z-score', rotation=270, labelpad=8, color='white', fontsize=8)
        elif selection == 'Z-score (neurodegenerativ regioner)':
            if space == 'MNI space' or space == 'PAT space':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data4_reg[i])
                    self.images3[i].set_cmap(custom_cmap)
                    self.images3[i].set_norm(self.norm4)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Z-score', rotation=270, labelpad=8, color='white', fontsize=8)
            elif space == 'MNI space relative' or space == 'PAT space relative':
                for i in range(6):
                    self.images3[i].set_data(self.channel_data4_reg_rel[i])
                    self.images3[i].set_cmap(custom_cmap)
                    self.images3[i].set_norm(self.norm4)
                self.cbar3.update_normal(self.images3[-1])
                self.cbar3.set_label('Z-score', rotation=270, labelpad=8, color='white', fontsize=8)
        self.canvas3.draw()

    def change_fourth_row(self, event):
        """Change the plot based on the selection from the fourth dropdown menu."""
        selection = self.plot_var4.get()
        space = self.space_var.get()
        if space == 'MNI space' or space == 'PAT space':
            print(f"Fourth row selection changed to: {selection}")
            self.current_dataset_row4 = selection
    
            if self.current_dataset_row4 == 'SSP':
                norm = self.norm_row4
            else:
                norm = None  # For other datasets, norm is not used
    
            # Update the figure in place
            images = create_custom_figure(
                self.first_values_yz_neg_x0, self.first_values_yz_pos_x1, self.first_values_yz_pos_x2,
                self.first_values_yz_neg_x2, self.first_values_xz_pos_y2, self.first_values_xz_neg_y2,
                self.last_figure_1, self.last_figure_2, self.last_figure_3,
                self.last_figure_4, self.last_figure_5, self.last_figure_6,
                self.region_last_figure_1, self.region_last_figure_2,
                self.region_last_figure_3, self.region_last_figure_4,
                self.region_last_figure_5, self.region_last_figure_6,
                self.flow_region_last_figure_1, self.flow_region_last_figure_2,
                self.flow_region_last_figure_3, self.flow_region_last_figure_4,
                self.flow_region_last_figure_5, self.flow_region_last_figure_6,
                self.current_dataset_row4, norm=norm, fig=self.fig4
            )
    
            # Update the stored images
            if images:
                self.images_row4[self.current_dataset_row4] = images
    
            # Adjust axes positions in fig4
            for ax in self.fig4.axes:
                pos = ax.get_position()
                x0 = pos.x0 + pos.width * 0.1
                y0 = pos.y0 + pos.height * 0.1
                width = pos.width * 0.8
                height = pos.height * 0.8
                ax.set_position([x0, y0, width, height])
                
        elif space == 'MNI space relative' or space == 'PAT space relative':
            print(f"Fourth row selection changed to: {selection}")
            self.current_dataset_row4 = selection
    
            if self.current_dataset_row4 == 'SSP':
                norm = self.norm_row4
            else:
                norm = None  # For other datasets, norm is not used
    
            # Update the figure in place
            images = create_custom_figure(
                self.first_values_yz_neg_x0_rel, self.first_values_yz_pos_x1_rel, self.first_values_yz_pos_x2_rel,
                self.first_values_yz_neg_x2_rel, self.first_values_xz_pos_y2_rel, self.first_values_xz_neg_y2_rel,
                self.last_figure_1_rel, self.last_figure_2_rel, self.last_figure_3_rel,
                self.last_figure_4_rel, self.last_figure_5_rel, self.last_figure_6_rel,
                self.region_last_figure_1_rel, self.region_last_figure_2_rel,
                self.region_last_figure_3_rel, self.region_last_figure_4_rel,
                self.region_last_figure_5_rel, self.region_last_figure_6_rel,
                self.flow_region_last_figure_1_rel, self.flow_region_last_figure_2_rel,
                self.flow_region_last_figure_3_rel, self.flow_region_last_figure_4_rel,
                self.flow_region_last_figure_5_rel, self.flow_region_last_figure_6_rel,
                self.current_dataset_row4, norm=norm, fig=self.fig4
            )
    
            # Update the stored images
            if images:
                self.images_row4[self.current_dataset_row4] = images
    
            # Adjust axes positions in fig4
            for ax in self.fig4.axes:
                pos = ax.get_position()
                x0 = pos.x0 + pos.width * 0.1
                y0 = pos.y0 + pos.height * 0.1
                width = pos.width * 0.8
                height = pos.height * 0.8
                ax.set_position([x0, y0, width, height])
                
        self.canvas4.draw()

    def change_space(self, event):
        """Update the plot options based on the selected space (MNI or PAT)."""
        space = self.space_var.get()
        print(f"Space selection changed to: {space}")

        # Reset the plot selection variables
        self.plot_var1.set('Perfusion')
        self.plot_var2.set('Flow-out rate')
        self.plot_var3.set('Volume of Distribution')

        # Update the plots to reflect the new space selection
        self.change_first_row(None)
        self.change_second_row(None)
        self.change_third_row(None)
        self.change_fourth_row(None)

    # ... [Other methods remain unchanged]

    def show_large_image(self, axes, colorbar=None):
        """Open a new window to display a larger version of the images in the selected row with a colorbar (if provided)."""
        print("Opening large image window...")
        # Create a new window for displaying the large images
        large_image_window = tk.Toplevel(self)
        large_image_window.title("Large Images")
        large_image_window.configure(bg="black")
        fig = Figure(figsize=(16, 5), facecolor='black')

        # Adjust GridSpec depending on whether we have a colorbar
        if colorbar:
            gs = GridSpec(1, len(axes) + 1, width_ratios=[1] * len(axes) + [0.05], figure=fig)
        else:
            gs = GridSpec(1, len(axes), width_ratios=[1] * len(axes), figure=fig)

        # Plot each image in the new figure
        large_axes = []
        for i, axis in enumerate(axes):
            # Create a subplot for each image
            ax = fig.add_subplot(gs[0, i], facecolor='black')

            # Extract the data and norm from the corresponding smaller plot
            img_data = axis.images[0].get_array()
            norm = axis.images[0].norm
            cmap = axis.images[0].get_cmap()

            # Display the data in the large subplot
            im = ax.imshow(img_data, cmap=cmap, norm=norm, interpolation='bicubic')
            ax.axis('off')
            # Fix aspect ratio
            ax.set_aspect('equal')

            large_axes.append(im)

        # Add the colorbar to the figure if provided
        if colorbar:
            cbar_ax = fig.add_subplot(gs[0, len(axes)])
            cbar = fig.colorbar(large_axes[-1], cax=cbar_ax, orientation='vertical')
            cbar.set_label(colorbar.ax.get_ylabel(), rotation=270, labelpad=15, color='white', fontsize=12)

            # Set colorbar properties to match the theme
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('white')
            cbar.ax.yaxis.set_tick_params(labelcolor='white')
            cbar.ax.tick_params(labelsize=14, colors='white')

        # Create a canvas to hold the figure and embed it in the new window
        canvas = FigureCanvasTkAgg(fig, master=large_image_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Adjust the layout and draw the canvas
        fig.tight_layout()
        canvas.draw()

    def show_large_image_fig4(self):
        """Display the large version of the fourth figure."""
        print("Opening large image for figure 4...")
        large_image_window = tk.Toplevel(self)
        large_image_window.title("Large Image")
        large_image_window.configure(bg="black")

        # Create the figure with larger size
        fig_large = Figure(figsize=(16, 5), facecolor='black')

        if self.current_dataset_row4 == 'SSP':
            norm = self.norm_row4
        else:
            norm = None  # For 'SSP Z-score', norm is not used

        # Create the figure anew
        create_custom_figure(
            self.first_values_yz_neg_x0, self.first_values_yz_pos_x1, self.first_values_yz_pos_x2,
            self.first_values_yz_neg_x2, self.first_values_xz_pos_y2, self.first_values_xz_neg_y2,
            self.last_figure_1, self.last_figure_2, self.last_figure_3,
            self.last_figure_4, self.last_figure_5, self.last_figure_6,
            self.region_last_figure_1, self.region_last_figure_2,
            self.region_last_figure_3, self.region_last_figure_4,
            self.region_last_figure_5, self.region_last_figure_6,
            self.flow_region_last_figure_1, self.flow_region_last_figure_2,
            self.flow_region_last_figure_3, self.flow_region_last_figure_4,
            self.flow_region_last_figure_5, self.flow_region_last_figure_6,
            self.current_dataset_row4, norm=norm, fig=fig_large
        )

        # Create a canvas to hold the figure and embed it in the new window
        canvas = FigureCanvasTkAgg(fig_large, master=large_image_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig_large.tight_layout()
        canvas.draw()


def open_app_window(initial_window, button_var,
                    transformed_K_1_1, transformed_K_2_1, Z_brain, K_1_reshape_list_1,
                    K_2_reshape_list_1, first_values_yz_neg_x0, first_values_yz_pos_x1,
                    first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
                    first_values_xz_neg_y2,
                    last_figure_1, last_figure_2, last_figure_3,
                    last_figure_4, last_figure_5, last_figure_6, 
                    region_last_figure_1, region_last_figure_2, region_last_figure_3, 
                    region_last_figure_4, region_last_figure_5, region_last_figure_6,
                    flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3,
                    flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6,
                    z_score, z_scores_flow, z_brain_regions, Cerebellum_mean_k1, Cerebellum_mean_k2, 
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
                    motion_parameters_array_1, dicom_name,
                    K_1_reshape_list_1_pat, K_2_reshape_list_1_pat,
                    pixel_spacing, slice_thickness, image_positions
                    ):
    """
    Hides the initial_window and opens the App window as a new top-level window.

    Args:
        initial_window (tk.Tk or tk.Toplevel): The window to be hidden.
        button_var (tk.StringVar): Variable to capture which button is pressed.

    Returns:
        App: An instance of the App window.
    """
    print("Opening App window...")
    initial_window.withdraw()  # Hide the initial window instead of destroying it
    print("Initial window hidden.")

    # Create the App window as a Toplevel instance, passing the button_var
    app = App(transformed_K_1_1, transformed_K_2_1, Z_brain, K_1_reshape_list_1,
              K_2_reshape_list_1, first_values_yz_neg_x0, first_values_yz_pos_x1,
              first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
              first_values_xz_neg_y2,
              last_figure_1, last_figure_2, last_figure_3,
              last_figure_4, last_figure_5, last_figure_6,
              region_last_figure_1, region_last_figure_2, region_last_figure_3, 
              region_last_figure_4, region_last_figure_5, region_last_figure_6,
              flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3,
              flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6,
              z_score, z_scores_flow, z_brain_regions, Cerebellum_mean_k1, Cerebellum_mean_k2, 
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
              motion_parameters_array_1, dicom_name, K_1_reshape_list_1_pat, K_2_reshape_list_1_pat, 
              pixel_spacing, slice_thickness, image_positions,
              master=initial_window, button_var=button_var)
    print("App window created.")

    return app
