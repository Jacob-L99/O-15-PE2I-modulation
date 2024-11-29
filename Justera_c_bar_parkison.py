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

def create_custom_figure(last_figure_1, last_figure_2, last_figure_3,
                          last_figure_4, last_figure_5, last_figure_6,
                          data_set, norm=None, fig=None):
    """Creates or updates the custom figure using the provided data and returns it."""
    print('-----', data_set, '-----------')
    if fig is None:
        fig = Figure(facecolor='black')
    else:
        fig.clear()

    if data_set == 'SSP Z-score':
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

        return axs, cbar_ax  # Return the axes and colorbar axis
    else:
        # Handle other datasets if any
        pass


class App(tk.Toplevel):
    def __init__(self, BP_MNI=None, R_I_MNI=None, BP_pat=None, R_I_pat=None,
                  regions_first_values_yz_neg_x0=None, regions_first_values_yz_pos_x1=None, 
                  regions_first_values_yz_pos_x2=None, regions_first_values_yz_neg_x2=None, 
                  regions_first_values_xz_pos_y2=None, regions_first_values_xz_neg_y2=None,
                  z_scores=None, z_brain_regions=None, Z_brain_flow=None, z_scores_flow=None,
                  flow_region_last_figure_1=None, flow_region_last_figure_2=None, 
                  flow_region_last_figure_3=None, flow_region_last_figure_4=None, 
                  flow_region_last_figure_5=None, flow_region_last_figure_6=None,
                  BP_type=None, tracer=None, motion_parameters_array=None, 
                  means_list_flow=None, mean_values_regions=None, directory=None, 
                  BP_reshape_list_pat_space=None, R_I_reshape_list_pat_space=None,
                  pixel_spacing=None, slice_thickness=None, image_positions=None,
                  master=None, button_var=None):
        """
        Initializes the App window.

        Args:
            master (tk.Tk or tk.Toplevel): The parent window.
            button_var (tk.StringVar): Variable to capture which button is pressed.
        """
        super().__init__(master)
        self.resizing = False  # Flag to prevent recursive resizing
        self.state('zoomed')  # Keep the window maximized
        self.button_var = button_var  # Store the reference to the StringVar
        self.title("Adjust Colormap Scale")
        self.configure(bg="black")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Style configuration for dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", background="grey", foreground="white")
        style.configure("TFrame", background="black")
        style.configure("TLabel", background="black", foreground="white")
        style.configure("TCombobox", background="grey", foreground="white", fieldbackground="grey")
        style.configure("TScale", background="black", troughcolor="grey", sliderlength=30)

        # Configure the main window's grid
        self.grid_rowconfigure(0, weight=0)  # Slider
        self.grid_rowconfigure(1, weight=1)  # Main content
        self.grid_rowconfigure(2, weight=0)  # Bottom buttons
        self.grid_columnconfigure(0, weight=1)

        # Frame for the slice slider
        slider_frame = ttk.Frame(self, style="TFrame")
        slider_frame.grid(row=0, column=0, sticky="ew")
        slider_frame.grid_columnconfigure(0, weight=0)
        slider_frame.grid_columnconfigure(1, weight=1)

        # Main frame to hold canvas and sliders
        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.grid(row=1, column=0, sticky="nsew")
        main_frame.grid_rowconfigure((0, 1, 2), weight=1)
        main_frame.grid_columnconfigure(1, weight=1)  # Column for figures

        # Create three matplotlib figures without fixed figsize
        self.fig1 = Figure(facecolor='black')
        gs1 = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.02], figure=self.fig1)  # Reduced colorbar width ratio
        self.ax1 = [self.fig1.add_subplot(gs1[0, i]) for i in range(3)]
        self.cbar_ax1 = self.fig1.add_subplot(gs1[0, 3])

        self.fig2 = Figure(facecolor='black')
        gs2 = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.02], figure=self.fig2)  # Reduced colorbar width ratio
        self.ax2 = [self.fig2.add_subplot(gs2[0, i]) for i in range(3)]
        self.cbar_ax2 = self.fig2.add_subplot(gs2[0, 3])

        self.fig3 = Figure(facecolor='black')
        gs3 = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.02], figure=self.fig3)
        self.ax3 = [self.fig3.add_subplot(gs3[0, i]) for i in range(3)]
        self.cbar_ax3 = self.fig3.add_subplot(gs3[0, 3])



        # Load or generate test datasets for the first three rows
        try:
            self.data1 = np.rot90(BP_MNI)
            self.data2 = np.rot90(R_I_MNI)
            self.data4 = np.rot90(z_brain_regions)  # Z-score data
            self.data7 = np.rot90(Z_brain_flow)


            # Load data5 and data6
            self.data5 = np.rot90(BP_pat)
            self.data6 = np.rot90(R_I_pat)
            
            self.data5_pat = BP_reshape_list_pat_space
            self.data6_pat = R_I_reshape_list_pat_space
            
            #lista med z-scores i 48 regioner
            self.z_scores=z_scores
            self.z_scores_flow=z_scores_flow

            self.BP_type=BP_type
            self.tracer=tracer
            self.motion_parameters_array=motion_parameters_array
            self.means_list_flow=means_list_flow
            self.mean_values_regions=mean_values_regions
            self.directory=directory
            self.pixel_spacing=pixel_spacing
            self.slice_thickness=slice_thickness
            self.image_positions=image_positions
            
            # Pad data5 and data6 to match the number of slices in data1
            slices_diff = self.data1.shape[-1] - self.data5.shape[-1]
            if slices_diff > 0:
                pad_width = ((0, 0), (0, 0), (0, slices_diff))
                self.data5 = np.pad(self.data5, pad_width, mode='constant', constant_values=0)
                self.data6 = np.pad(self.data6, pad_width, mode='constant', constant_values=0)
            elif slices_diff < 0:
                # If data5 has more slices, truncate it
                self.data5 = self.data5[..., :self.data1.shape[-1]]
                self.data6 = self.data6[..., :self.data1.shape[-1]]

            self.regions_first_values_yz_neg_x0=regions_first_values_yz_neg_x0
            self.regions_first_values_yz_pos_x1=regions_first_values_yz_pos_x1
            self.regions_first_values_yz_pos_x2=regions_first_values_yz_pos_x2
            self.regions_first_values_yz_neg_x2=regions_first_values_yz_neg_x2
            self.regions_first_values_xz_pos_y2=regions_first_values_xz_pos_y2
            self.regions_first_values_xz_neg_y2=regions_first_values_xz_neg_y2
            
            self.flow_region_last_figure_1 = flow_region_last_figure_1
            self.flow_region_last_figure_2 = flow_region_last_figure_2
            self.flow_region_last_figure_3 = flow_region_last_figure_3
            self.flow_region_last_figure_4 = flow_region_last_figure_4
            self.flow_region_last_figure_5 = flow_region_last_figure_5
            self.flow_region_last_figure_6 = flow_region_last_figure_6

        except FileNotFoundError as e:
            print(f"Data file not found: {e}")
            messagebox.showerror("Error", f"Data file not found: {e}")
            self.destroy()
            return

        # Initialize slice index and maximum slices
        self.slice_index = 80  # initial slice index
        self.max_slices = self.data1.shape[-1]

        # Slice slider
        ttk.Label(slider_frame, text="Slice Index", background="black", foreground="white").grid(row=0, column=0, padx=5)
        self.slice_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=self.max_slices - 1,
            orient='horizontal',
            command=self.update_slices,
            style="TScale",
            length=400
        )
        self.slice_slider.set(self.slice_index)
        self.slice_slider.grid(row=0, column=1, sticky="ew", padx=5)

        # Initialize normalization objects
        self.norm = Normalize(vmin=0, vmax=10)
        self.norm2 = Normalize(vmin=0, vmax=5)  # New Normalization for Row 2
        self.norm3 = Normalize(vmin=-5, vmax=5)
        self.norm4 = Normalize(vmin=-5, vmax=5)  # For Z-score data

        # Define colormap
        self.cmap = my_cmap
        self.cmap_BWR = custom_cmap

        # Initialize channel data
        slice_indices = self.get_slice_indices()
        self.channel_data1 = [self.data1[..., idx] for idx in slice_indices]
        self.channel_data2 = [self.data2[..., idx] for idx in slice_indices]
        # self.channel_data3 = [self.data3[..., idx] for idx in slice_indices]
        self.channel_data4 = [self.data4[..., idx] for idx in slice_indices]
        self.channel_data7 = [self.data7[..., idx] for idx in slice_indices]

        self.channel_data5 = [self.data5[..., idx] for idx in slice_indices]
        self.channel_data6 = [self.data6[..., idx] for idx in slice_indices]

        # Initialize image plots with the datasets
        self.images1 = [
            self.ax1[i].imshow(self.channel_data1[i], interpolation='bicubic', norm=self.norm, cmap=self.cmap)
            for i in range(3)
        ]
        self.images2 = [
            self.ax2[i].imshow(self.channel_data2[i], interpolation='bicubic', norm=self.norm2, cmap=self.cmap)
            for i in range(3)
        ]
        self.images3 = [
            self.ax3[i].imshow(self.channel_data4[i], interpolation='bicubic', norm=self.norm3, cmap=self.cmap_BWR)
            for i in range(3)
        ]

        # Remove axis from each plot and set facecolor
        for ax in self.ax1 + self.ax2:
            ax.set_facecolor('black')
            ax.tick_params(axis='both', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.axis('off')

        # Create colorbars for the first two plots using the fourth subplot
        self.cbar1 = self.fig1.colorbar(self.images1[-1], cax=self.cbar_ax1)
        self.cbar1.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
        self.cbar1.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
        self.cbar1.outline.set_edgecolor('white')

        self.cbar2 = self.fig2.colorbar(self.images2[-1], cax=self.cbar_ax2)
        self.cbar2.set_label('ml/min/g', rotation=270, labelpad=8, color='white', fontsize=8)
        self.cbar2.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
        self.cbar2.outline.set_edgecolor('white')

        # Embed the figures in the Tk window using grid
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=main_frame)
        self.canvas_widget1 = self.canvas1.get_tk_widget()
        self.canvas_widget1.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=main_frame)
        self.canvas_widget2 = self.canvas2.get_tk_widget()
        self.canvas_widget2.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Create the canvas for row 3
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=main_frame)
        self.canvas_widget3 = self.canvas3.get_tk_widget()
        self.canvas_widget3.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

        # Initialize the third row plot based on initial selection
        self.plot_var3 = tk.StringVar()
        self.plot_var3.set('Z-score (neurodegenerativ regioner)')  # Default value

        # Call change_third_row to initialize the plot
        self.change_third_row(None)

        # Frame for the sliders
        sliders_frame = ttk.Frame(main_frame, style="TFrame")
        sliders_frame.grid(row=0, column=2, rowspan=3, sticky="ns", padx=10, pady=10)
        sliders_frame.grid_rowconfigure((0, 1), weight=1)

        # Create a vertical slider for the upper limit adjustment (Row 1 and Row 2)
        # Existing slider for Row 1
        self.slider_vmax = ttk.Scale(
            sliders_frame,
            from_=2,
            to=0.01,
            orient='vertical',
            command=self.update_vmax,
            style="TScale"
        )
        self.slider_vmax.set(2)
        ttk.Label(sliders_frame, text="Övre gräns " + self.BP_type , background="black", foreground="white").pack(pady=(0, 5))
        self.slider_vmax.pack(expand=True, fill=tk.Y, pady=(0, 10))

        # New slider for Row 2
        self.slider_vmax2 = ttk.Scale(
            sliders_frame,
            from_=5,
            to=0.01,
            orient='vertical',
            command=self.update_vmax2,  # Callback for Row 2
            style="TScale"
        )
        self.slider_vmax2.set(5)
        ttk.Label(sliders_frame, text="Övre gräns R_I", background="black", foreground="white").pack(pady=(0, 5))
        self.slider_vmax2.pack(expand=True, fill=tk.Y, pady=(0, 10))

        # Create frame for the first row controls
        row1_controls = ttk.Frame(main_frame, style="TFrame")
        row1_controls.grid(row=0, column=0, sticky="ns", padx=5, pady=5)

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
        self.space_dropdown['values'] = ['MNI space', 'PAT space']
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
        self.plot_dropdown1['values'] = [self.BP_type]
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
        row2_controls = ttk.Frame(main_frame, style="TFrame")
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
        self.plot_dropdown2['values'] = ['R_I']
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
        row3_controls = ttk.Frame(main_frame, style="TFrame")
        row3_controls.grid(row=2, column=0, sticky="ns", padx=5, pady=5)

        # Dropdown menu to select the dataset for the third row
        self.plot_dropdown3 = ttk.Combobox(
            row3_controls,
            textvariable=self.plot_var3,
            state='readonly',
            width=25,
            style="TCombobox"
        )
        self.plot_dropdown3['values'] = ('Z-score (neurodegenerativ regioner)', 'Z-score (flödes regioner)', 'SSP (neurodegenerativ regioner)', 'SSP (flödes regioner)')  # Include 'SSP regions'
        self.plot_dropdown3.current(0)
        self.plot_dropdown3.grid(row=0, column=0, padx=5, pady=5)

        # Button to show the third row large
        self.show_large_button3 = ttk.Button(
            row3_controls,
            text="Visa stort",
            command=lambda: self.show_large_image(self.ax3, self.cbar3, sm=getattr(self, 'cbar3_mappable', None)),
            style="TButton"
        )
        self.show_large_button3.grid(row=1, column=0, padx=5, pady=5)

        # Configure grid to expand properly within main_frame
        for i in range(3):
            main_frame.grid_rowconfigure(i, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)  # Column for figures
        main_frame.grid_columnconfigure(0, weight=0)  # Column for controls
        main_frame.grid_columnconfigure(2, weight=0)  # Column for sliders

        # Remove fixed figure sizes and allow them to fit the grid
        self.fig1.tight_layout()
        self.fig2.tight_layout()
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        # Bind dropdown change events
        self.plot_dropdown1.bind('<<ComboboxSelected>>', self.change_first_row)
        self.plot_dropdown2.bind('<<ComboboxSelected>>', self.change_second_row)
        self.plot_dropdown3.bind('<<ComboboxSelected>>', self.change_third_row)
        self.space_dropdown.bind('<<ComboboxSelected>>', self.change_space)

        # Bottom frame for buttons
        bottom_frame = ttk.Frame(self, style="TFrame")
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)

        # Create a subframe to center the buttons and restrict their maximum width
        buttons_subframe = ttk.Frame(bottom_frame, style="TFrame")
        buttons_subframe.grid(row=0, column=0, columnspan=3, sticky="n")
        buttons_subframe.grid_columnconfigure((0, 1, 2), weight=1)

        # Create buttons for the buttons_subframe using grid
        self.button_left = ttk.Button(
            buttons_subframe,
            text="Spara som .nii",
            command=self.left_button_action,
            style="TButton",
            width=15  # Set a fixed width for consistency
        )
        self.button_left.grid(row=0, column=0, padx=20, pady=5, sticky="ew")

        self.button_middle = ttk.Button(
            buttons_subframe,
            text="Spara som dicom",
            command=self.middle_button_action,
            style="TButton",
            width=15  # Set a fixed width for consistency
        )
        self.button_middle.grid(row=0, column=1, padx=20, pady=5, sticky="ew")

        self.button_right = ttk.Button(
            buttons_subframe,
            text="Skapa PDF",
            command=self.right_button_action,
            style="TButton",
            width=15  # Set a fixed width for consistency
        )
        self.button_right.grid(row=0, column=2, padx=20, pady=5, sticky="ew")

        # Bind the window's configure event for dynamic resizing
        self.bind("<Configure>", self.on_resize)

        # Add Window Management Commands
        self.deiconify()           # Ensure the window is not minimized or hidden
        self.lift()                # Bring the window to the top
        self.focus_force()         # Focus on the App window
        self.grab_set()            # Make the App window modal (optional but recommended)
        self.update_idletasks()    # Process any pending idle tasks
        self.update()              # Force an update to render all widgets

        # Release the grab to make the window interactive
        self.grab_release()

    def get_slice_indices(self):
        """Return the list of slice indices to display based on the current slice index."""
        idx = int(self.slice_index)
        delta = 20  # Number of slices apart
        indices = [idx]

        # Calculate previous index
        prev_idx = idx - delta
        if prev_idx >= 0:
            indices.insert(0, prev_idx)
        else:
            indices.insert(0, 0)

        # Calculate next index
        next_idx = idx + delta
        if next_idx < self.max_slices:
            indices.append(next_idx)
        else:
            indices.append(self.max_slices - 1)

        return indices

    def update_slices(self, value):
        """Update the displayed slices when the slider is moved."""
        self.slice_index = int(float(value))
        print(f"Updating slices to index: {self.slice_index}")
        slice_indices = self.get_slice_indices()

        # Update channel data
        self.channel_data1 = [self.data1[..., idx] for idx in slice_indices]
        self.channel_data2 = [self.data2[..., idx] for idx in slice_indices]
        # self.channel_data3 = [self.data3[..., idx] for idx in slice_indices]
        self.channel_data4 = [self.data4[..., idx] for idx in slice_indices]
        self.channel_data7 = [self.data7[..., idx] for idx in slice_indices]

        self.channel_data5 = [self.data5[..., idx] for idx in slice_indices]
        self.channel_data6 = [self.data6[..., idx] for idx in slice_indices]

        # Update images
        self.change_first_row(None)
        self.change_second_row(None)
        self.change_third_row(None)

    def update_vmax(self, value):
        """Update the vmax value for Row 1 when the slider is moved."""
        global vmax1
        vmax1 = float(value)
        print(f"Updating vmax to: {vmax1}")
        self.norm.vmax = vmax1
        for img in self.images1:
            img.set_norm(self.norm)
        self.canvas1.draw()

    def update_vmax2(self, value):
        """Update the vmax value for Row 2 when the slider is moved."""
        global vmax2
        vmax2 = float(value)
        print(f"Updating vmax2 to: {vmax2}")
        self.norm2.vmax = vmax2
        for img in self.images2:
            img.set_norm(self.norm2)
        self.canvas2.draw()

    def on_resize(self, event):
        """Handle the window resize event to adjust figure sizes dynamically."""
        if self.resizing:
            return  # Prevent recursive resizing
        self.resizing = True

        try:
            # Define a helper function to resize a figure based on its container's size
            def resize_figure(canvas, figure):
                container = canvas.get_tk_widget()
                width = container.winfo_width()
                height = container.winfo_height()
                if width < 10 or height < 10:
                    return  # Avoid too small sizes
                dpi = figure.get_dpi()
                figure.set_size_inches(width / dpi, height / dpi)
                figure.tight_layout()
                canvas.draw()

            # Resize each figure
            resize_figure(self.canvas1, self.fig1)
            resize_figure(self.canvas2, self.fig2)
            resize_figure(self.canvas3, self.fig3)

        finally:
            self.resizing = False

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
        array1=np.flip(np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data1, axes=(1,2)), axes=(1,2)), (1,0,2)))), axis=0)
        array2=np.flip(np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data2, axes=(1,2)), axes=(1,2)), (1,0,2)))), axis=0)
        # array3=np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data3, axes=(1,2)), axes=(1,2)), (1,0,2))))

        affine_matrix = np.eye(4)
        arrays=[array1, array2]
        filenames=['_' + self.BP_type + '_MNI.nii', '_R_I_MNI.nii']
        # np.save('hej.npy', self.data6_pat)
        array1_pat=np.rot90(np.flip(np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data5_pat, axes=(1,2)), axes=(1,2)), (2,0,1)))), axis=0))
        array2_pat=np.rot90(np.flip(np.rot90(np.rot90(np.transpose(np.rot90(np.rot90(self.data6_pat, axes=(1,2)), axes=(1,2)), (2,0,1)))), axis=0))

        
        arrays_pat=[array1_pat, array2_pat]
        filenames_pat=['_' + self.BP_type + '_PAT.nii', '_R_I_PAT.nii']
        save_as_nifti(self.directory, arrays, filenames, arrays_pat, filenames_pat, self.pixel_spacing, self.slice_thickness, self.image_positions)
        # if self.button_var:
        #     self.button_var.set("Left")
        # self.destroy()

    def middle_button_action(self):
        """Action for the middle button."""
        from Save_as_nifyt_dicom import save_as_dicom
        arrays=[self.data1, self.data2]
        output_dirs=['dicom ' + self.BP_type, 'dicom R_I']
        save_as_dicom(
                arrays=arrays,
                output_dirs=output_dirs,
                patient_id='Patient_001',
                # series_instance_uid=series_instance_uid,  # Optional
                # study_instance_uid='1.2.840.113619.2.55.3.604688123.12345.1607771234.467',  # Optional
                # sop_instance_uid_prefix='1.2.840.113619.2.55.3.604688123.12345.1607771234'
            )

    def right_button_action(self):
        """Action for the right button."""
        from PDF_park import PDF_park
        PDF_park(self.directory, self.data1, self.data2, self.data4, self.data7,  
                 self.regions_first_values_yz_neg_x0,self.regions_first_values_yz_pos_x1,
                 self.regions_first_values_yz_pos_x2,self.regions_first_values_yz_neg_x2,
                 self.regions_first_values_xz_pos_y2, self.regions_first_values_xz_neg_y2,
                 self.flow_region_last_figure_1, self.flow_region_last_figure_2,
                 self.flow_region_last_figure_3, self.flow_region_last_figure_4,
                 self.flow_region_last_figure_5, self.flow_region_last_figure_6,
                 self.z_scores, self.z_scores_flow, vmax1, vmax2, self.tracer, self.motion_parameters_array,
                 self.means_list_flow, self.mean_values_regions)


    def change_first_row(self, event):
        """Change the plot based on the selection from the first dropdown menu."""
        selection = self.plot_var1.get()
        space = self.space_var.get()
        print(f"First row selection changed to: {selection} in space {space}")
        if selection == self.BP_type:
            if space == 'MNI space':
                for i in range(3):
                    self.images1[i].set_data(self.channel_data1[i])
            elif space == 'PAT space':
                for i in range(3):
                    self.images1[i].set_data(self.channel_data5[i])

        self.canvas1.draw()

    def change_second_row(self, event):
        """Change the plot based on the selection from the second dropdown menu."""
        selection = self.plot_var2.get()
        space = self.space_var.get()
        print(f"Second row selection changed to: {selection} in space {space}")
        if selection == 'R_I':
            if space == 'MNI space':
                for i in range(3):
                    self.images2[i].set_data(self.channel_data2[i])
            elif space == 'PAT space':
                for i in range(3):
                    self.images2[i].set_data(self.channel_data6[i])
        self.canvas2.draw()

    def change_third_row(self, event):
        """Change the plot based on the selection from the third dropdown menu."""
        selection = self.plot_var3.get()
        print(f"Third row selection changed to: {selection}")

        self.fig3.clear()  # Clear the figure
            
        if selection == 'Z-score (neurodegenerativ regioner)':
            # Create the axes
            gs3 = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.02], figure=self.fig3)
            self.ax3 = [self.fig3.add_subplot(gs3[0, i]) for i in range(3)]
            self.cbar_ax3 = self.fig3.add_subplot(gs3[0, 3])

            # Initialize normalization
            self.norm3 = Normalize(vmin=-5, vmax=5)
            self.cmap_BWR = custom_cmap

            # Initialize images
            self.images3 = [
                self.ax3[i].imshow(self.channel_data4[i], interpolation='nearest', norm=self.norm3, cmap=self.cmap_BWR)
                for i in range(3)
            ]

            # Remove axis from each plot and set facecolor
            for ax in self.ax3:
                ax.set_facecolor('black')
                ax.tick_params(axis='both', colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.axis('off')

            # Create colorbar
            self.cbar3 = self.fig3.colorbar(self.images3[-1], cax=self.cbar_ax3)
            self.cbar3.set_label('Value', rotation=270, labelpad=8, color='white', fontsize=8)
            self.cbar3.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
            self.cbar3.outline.set_edgecolor('white')
            
        elif selection == 'Z-score (flödes regioner)':
            # Create the axes
            gs3 = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.02], figure=self.fig3)
            self.ax3 = [self.fig3.add_subplot(gs3[0, i]) for i in range(3)]
            self.cbar_ax3 = self.fig3.add_subplot(gs3[0, 3])

            # Initialize normalization
            self.norm3 = Normalize(vmin=-5, vmax=5)
            self.cmap_BWR = custom_cmap

            # Initialize images
            self.images3 = [
                self.ax3[i].imshow(self.channel_data7[i], interpolation='nearest', norm=self.norm3, cmap=self.cmap_BWR)
                for i in range(3)
            ]

            # Remove axis from each plot and set facecolor
            for ax in self.ax3:
                ax.set_facecolor('black')
                ax.tick_params(axis='both', colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.axis('off')

            # Create colorbar
            self.cbar3 = self.fig3.colorbar(self.images3[-1], cax=self.cbar_ax3)
            self.cbar3.set_label('Value', rotation=270, labelpad=8, color='white', fontsize=8)
            self.cbar3.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
            self.cbar3.outline.set_edgecolor('white')

        elif selection == 'SSP (neurodegenerativ regioner)':
            self.norm3 = Normalize(vmin=-5, vmax=5)  # Adjust normalization as needed

            # Call create_custom_figure to populate self.fig3 and get axes
            self.ax3, self.cbar_ax3 = create_custom_figure(
                self.regions_first_values_yz_neg_x0,
                self.regions_first_values_yz_pos_x1,
                self.regions_first_values_yz_pos_x2,
                self.regions_first_values_yz_neg_x2,
                self.regions_first_values_xz_pos_y2,
                self.regions_first_values_xz_neg_y2,
                data_set='SSP Z-score',
                norm=self.norm3,
                fig=self.fig3
            )

            # Create a ScalarMappable to get a colorbar and store it
            self.cbar3_mappable = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
            self.cbar3 = self.fig3.colorbar(self.cbar3_mappable, cax=self.cbar_ax3)

            # Set colorbar properties to match the theme
            self.cbar3.set_label('Z-score', rotation=270, labelpad=15, color='white', fontsize=8)
            self.cbar3.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
            self.cbar3.outline.set_edgecolor('white')
            
        elif selection == 'SSP (flödes regioner)':
            self.norm3 = Normalize(vmin=-5, vmax=5)  # Adjust normalization as needed

            # Call create_custom_figure to populate self.fig3 and get axes
            self.ax3, self.cbar_ax3 = create_custom_figure(
                self.flow_region_last_figure_1,
                self.flow_region_last_figure_2,
                self.flow_region_last_figure_3,
                self.flow_region_last_figure_4,
                self.flow_region_last_figure_5,
                self.flow_region_last_figure_6,
                data_set='SSP Z-score',
                norm=self.norm3,
                fig=self.fig3
            )

            # Create a ScalarMappable to get a colorbar and store it
            self.cbar3_mappable = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
            self.cbar3 = self.fig3.colorbar(self.cbar3_mappable, cax=self.cbar_ax3)

            # Set colorbar properties to match the theme
            self.cbar3.set_label('Z-score', rotation=270, labelpad=15, color='white', fontsize=8)
            self.cbar3.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
            self.cbar3.outline.set_edgecolor('white')

        self.canvas3.draw()
        
        

    def change_space(self, event):
        """Update the plot options based on the selected space (MNI or PAT)."""
        space = self.space_var.get()
        print(f"Space selection changed to: {space}")

        # Reset the plot selection variables
        self.plot_var1.set(self.BP_type)
        self.plot_var2.set('R_I')

        # Update the plots to reflect the new space selection
        self.change_first_row(None)
        self.change_second_row(None)

    def show_large_image(self, axes, colorbar=None, sm=None):
        """Open a new window to display a larger version of the images in the selected row with a colorbar (if provided)."""
        print("Opening large image window...")
        # Create a new window for displaying the large images
        large_image_window = tk.Toplevel(self)
        large_image_window.title("Large Images")
        large_image_window.configure(bg="black")
        fig = Figure(facecolor='black')

        # Adjust GridSpec to make colorbar thicker and shorter
        # Increase the width ratio for the colorbar to make it thicker
        # Use 'shrink' to make it shorter
        # 'aspect' controls the aspect ratio; higher values make the colorbar thicker
        if colorbar:
            gs = GridSpec(1, len(axes) + 1, width_ratios=[1] * len(axes) + [0.1], figure=fig)  # Increased ratio for colorbar
        else:
            gs = GridSpec(1, len(axes), width_ratios=[1] * len(axes), figure=fig)

        # Plot each image in the new figure
        large_axes = []
        for i, axis in enumerate(axes):
            # Create a subplot for each image
            ax = fig.add_subplot(gs[0, i], facecolor='black')

            # Extract the data and properties from the corresponding smaller plot
            if axis.images:
                for img in axis.images:
                    img_data = img.get_array()
                    norm = img.norm
                    cmap = img.get_cmap()
                    alpha = img.get_alpha()
                    im = ax.imshow(img_data, cmap=cmap, norm=norm, alpha=alpha, interpolation='bicubic')
                ax.axis('off')
                large_axes.append(im)
            else:
                continue  # Skip if no images are found
        # Add the colorbar to the figure if provided
        if colorbar:
            cbar_ax = fig.add_subplot(gs[0, len(axes)])
            if sm:
                # Customize the colorbar size using shrink and aspect
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', shrink=0.6, aspect=15)
            elif large_axes:
                # Customize the colorbar size using shrink and aspect
                cbar = fig.colorbar(large_axes[-1], cax=cbar_ax, orientation='vertical', shrink=0.6, aspect=15)
            else:
                pass  # Handle the case when there are no images to create a colorbar

            # Set colorbar label
            if colorbar.ax.get_ylabel():
                cbar.set_label(colorbar.ax.get_ylabel(), rotation=270, labelpad=15, color='white', fontsize=8)

            # Set colorbar properties to match the theme
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=12)
            cbar.outline.set_edgecolor('white')

        # Create a canvas to hold the figure and embed it in the new window
        canvas = FigureCanvasTkAgg(fig, master=large_image_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Adjust the layout and draw the canvas
        fig.tight_layout()
        canvas.draw()



def open_app_window_park(initial_window, button_var,
                          BP_MNI, R_I_MNI, BP_pat, R_I_pat, 
                          regions_first_values_yz_neg_x0, regions_first_values_yz_pos_x1, 
                          regions_first_values_yz_pos_x2, regions_first_values_yz_neg_x2, 
                          regions_first_values_xz_pos_y2, regions_first_values_xz_neg_y2,
                          z_scores, z_brain_regions, Z_brain_flow, z_scores_flow, 
                          flow_region_last_figure_1, flow_region_last_figure_2, 
                          flow_region_last_figure_3, flow_region_last_figure_4, 
                          flow_region_last_figure_5, flow_region_last_figure_6,
                          AIF_time, tracer, motion_parameters_array, means_list_flow, mean_values_regions,
                          directory, BP_reshape_list_pat_space, R_I_reshape_list_pat_space,
                          pixel_spacing, slice_thickness, image_positions):
    if AIF_time[-1]>=45:
        BP_type="BP"
    else:
        BP_type="SBR"
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
    app = App(BP_MNI, R_I_MNI, BP_pat, R_I_pat, regions_first_values_yz_neg_x0, 
              regions_first_values_yz_pos_x1, regions_first_values_yz_pos_x2, 
              regions_first_values_yz_neg_x2, regions_first_values_xz_pos_y2, 
              regions_first_values_xz_neg_y2, z_scores, z_brain_regions, Z_brain_flow, z_scores_flow,
              flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, 
              flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6,
              BP_type, tracer, motion_parameters_array, means_list_flow, mean_values_regions,
              directory, BP_reshape_list_pat_space, R_I_reshape_list_pat_space, 
              pixel_spacing, slice_thickness, image_positions,
              master=initial_window, button_var=button_var)
    print("App window created.")

    return app
