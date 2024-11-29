# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:58:08 2024

@author: jacke
"""

import matplotlib.pyplot as plt
import numpy as np
import ants
import time
import matplotlib
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from tkinter import ttk



import os
import sys
import numpy as np

# Function to determine the correct file path
def get_file_path(filename):
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, look in the temporary directory
        return os.path.join(sys._MEIPASS, filename)
    else:
        # Otherwise, look in the current working directory
        return os.path.join(os.getcwd(), filename)

plt.style.use('dark_background')
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

def apply_colormap(image_np, vmin, vmax, cmap=my_cmap):
    """
    Apply a colormap to a 2D numpy image using global vmin and vmax.
    Args:
        image_np: The numpy array of the image.
        vmin: Global minimum value for normalization.
        vmax: Global maximum value for normalization.
        cmap: The colormap to apply.
    Returns:
        Colormapped image as a numpy array.
    """
    # Normalize the image to [0, 1] range using global vmin and vmax
    normalized_img = (image_np - vmin) / (vmax - vmin)
    # Handle any values outside [0, 1]
    normalized_img = np.clip(normalized_img, 0, 1)
    # Apply the colormap
    colormap = plt.cm.get_cmap(cmap)
    colored_img = colormap(normalized_img)  # This returns an RGBA image
    # Convert RGBA to RGB (drop the alpha channel)
    return colored_img[:, :, :3]

def blend_images(fixed_image_np, transformed_image_colored_np, alpha=0.5):
    """
    Alpha blend the fixed grayscale image and the colored registered image.
    Args:
        fixed_image_np: The numpy array of the fixed (template) grayscale image.
        transformed_image_colored_np: The numpy array of the colored registered image (RGB format).
        alpha: The blending factor. 0.5 means equal blending of both images.
    Returns:
        A blended numpy array of the two images.
    """
    # Normalize the fixed grayscale image to [0, 1]
    fixed_image_norm = (fixed_image_np - fixed_image_np.min()) / (fixed_image_np.max() - fixed_image_np.min())
    # Handle any values outside [0, 1]
    fixed_image_norm = np.clip(fixed_image_norm, 0, 1)

    # Convert fixed grayscale image to RGB
    fixed_image_rgb = np.stack([fixed_image_norm]*3, axis=-1)

    # Blend the images using the formula: blended = alpha * fixed + (1 - alpha) * transformed
    blended_image = alpha * fixed_image_rgb + (1 - alpha) * transformed_image_colored_np
    return blended_image

def update_slice(slice_index, fixed_image_np, transformed_image_np, initial_window, panel_blended, fig_blended, ax_blended, global_vmin, global_vmax):
    """
    Update the displayed image when the slice slider is moved.
    """
    alpha_value = 0.5  # Fixed alpha value
    fixed_slice = fixed_image_np[:, :, slice_index]
    transformed_slice = transformed_image_np[:, :, slice_index]
    
    # Apply colormap to the transformed image using global vmin and vmax
    transformed_colored = apply_colormap(transformed_slice, vmin=global_vmin, vmax=global_vmax, cmap=my_cmap)
    
    # Blend the images with a fixed alpha value
    blended_image_np = blend_images(fixed_slice, transformed_colored, alpha_value)

    # Clear the previous image from the axes
    ax_blended.clear()

    # Remove axis labels and ticks
    ax_blended.axis('off')

    # Display the blended image with a black background
    ax_blended.imshow(np.rot90(blended_image_np))
    fig_blended.patch.set_facecolor('black')  # Set background color to black
    fig_blended.canvas.draw()

    # Convert the blended image from Matplotlib to PIL for Tkinter
    blended_image_pil = Image.frombytes('RGB', fig_blended.canvas.get_width_height(), fig_blended.canvas.tostring_rgb())
    blended_image_tk = ImageTk.PhotoImage(blended_image_pil)

    # Update the image in the existing Label and remove white background
    panel_blended.config(image=blended_image_tk, bg="black")
    panel_blended.image = blended_image_tk  # Keep a reference to avoid garbage collection

def on_slice_change(val, fixed_image_np, transformed_image_np, initial_window, panel_blended, fig_blended, ax_blended, global_vmin, global_vmax):
    """
    Callback function for slice slider movement.
    """
    slice_index = int(float(val))  # Ensure the value is an integer
    update_slice(slice_index, fixed_image_np, transformed_image_np, initial_window, panel_blended, fig_blended, ax_blended, global_vmin, global_vmax)

def Registrering(data_4d, initial_window, loading_label, original_screen_width, seed):
    # Apply the custom style
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TButton", background="grey", foreground="white")
    style.configure("TFrame", background="black")
    style.configure("TLabel", background="black", foreground="white")
    style.configure("TCombobox", background="grey", foreground="white", fieldbackground="grey")
    style.configure("TScale", background="black", troughcolor="grey", sliderlength=30)

    image_3d_downsampled = np.load(get_file_path('small_templat_corrected.npy'))
    print(image_3d_downsampled.shape)
    
    volume_index = 10  # Extracting the 11th volume
    img_3d = ants.from_numpy(data_4d[:, :, :, volume_index])

    template_file_path = get_file_path('caa_t1_in_mni_template_smooth.nii')
    template_3d_flip = ants.image_read(template_file_path)
    template_3d = ants.from_numpy(
        image_3d_downsampled,
        origin=template_3d_flip.origin,
        spacing=template_3d_flip.spacing,
        direction=template_3d_flip.direction
    )
    
    # Track if "Ja" or "Nej" is pressed
    button_pressed = tk.StringVar(value="")  # Will store "Ja" or "Nej"
    
    # Function to handle Ja button press
    def on_ja_button_click():
        button_pressed.set("Ja")
    
    # Function to handle Nej button press
    def on_nej_button_click():
        button_pressed.set("Nej")
    
    
    
    start_tid_1 = time.time()
    
    random_seed = 42
    np.random.seed(random_seed)
    registration_1 = ants.registration(
    fixed=template_3d,
    moving=img_3d,
    type_of_transform='Affine',
    random_seed=seed,
    reg_iterations=[500, 250, 100, 50],  # Setting iterations at each resolution level
    # aff_sampling=1.0
    )
    
    

    
    transformed_img_3d_1 = ants.apply_transforms(
        fixed=template_3d,
        moving=img_3d,
        transformlist=registration_1['fwdtransforms']
    )
    
    random_seed = 42
    np.random.seed(random_seed)
    registration_2 = ants.registration(
        fixed=template_3d,
        moving=transformed_img_3d_1,
        type_of_transform='SyN',
        random_seed=seed,
        number_of_iterations=[100, 50, 25],
        syn_sampling_rate=1.0
    )
    

    transformed_img_3d = ants.apply_transforms(
        fixed=template_3d,
        moving=transformed_img_3d_1,
        transformlist=registration_2['fwdtransforms']
    )
    


    slut_tid_1 = time.time()
    reg_tid = slut_tid_1 - start_tid_1
    print("Registrering tid:", reg_tid)

    # Get numpy arrays from the fixed and transformed ANTs images for display
    fixed_image_np = template_3d.numpy()
    transformed_image_np = transformed_img_3d.numpy()

    # Calculate global vmin and vmax for the transformed image
    global_vmin = transformed_image_np.min()
    global_vmax = transformed_image_np.max()

    # Display the initial slice (middle slice)
    slice_index = fixed_image_np.shape[2] // 2
    fixed_slice = fixed_image_np[:, :, slice_index]
    transformed_slice = transformed_image_np[:, :, slice_index]

    # Apply a colormap to the transformed (registered) image using global vmin and vmax
    transformed_colored = apply_colormap(
        transformed_slice,
        vmin=global_vmin,
        vmax=global_vmax,
        cmap=my_cmap
    )

    # Perform alpha blending of the fixed (grayscale) and transformed (colored) images
    blended_image_np = blend_images(fixed_slice, transformed_colored)

    # Create the panel for the blended image and prepare Matplotlib figure
    fig_blended, ax_blended = plt.subplots(figsize=(8, 8), dpi=50)  # Increased size and set DPI
    ax_blended.axis('off')  # Turn off axes
    ax_blended.imshow(blended_image_np)
    fig_blended.patch.set_facecolor('black')  # Set background color to black
    fig_blended.canvas.draw()
    blended_image_pil = Image.frombytes(
        'RGB',
        fig_blended.canvas.get_width_height(),
        fig_blended.canvas.tostring_rgb()
    )
    blended_image_tk = ImageTk.PhotoImage(blended_image_pil)

    # Display the blended image in a black-background Label widget
    panel_blended = tk.Label(initial_window, image=blended_image_tk, bg="black")
    panel_blended.image = blended_image_tk  # Keep a reference to avoid garbage collection
    panel_blended.pack()

    # Add a slider to go through the slices
    slice_slider = ttk.Scale(
        initial_window,
        from_=0,
        to=fixed_image_np.shape[2]-1,
        orient="horizontal",
        length=400,
        style="TScale",
        command=lambda val: on_slice_change(
            val,
            fixed_image_np,
            transformed_image_np,
            initial_window,
            panel_blended,
            fig_blended,
            ax_blended,
            global_vmin,
            global_vmax
        )
    )
    slice_slider.set(slice_index)  # Set slider to the middle slice
    slice_slider.pack(side="bottom", pady=10)  # Positioned at the bottom

    # Add "Ja" and "Nej" buttons below the slice slider
    button_frame = ttk.Frame(initial_window, style="TFrame")
    button_frame.pack(side="bottom", pady=10)  # Positioned directly below the slice slider
    
    ja_button = ttk.Button(
        button_frame,
        text="Ja",
        command=on_ja_button_click,
        style="TButton"
    )
    ja_button.pack(side="left", padx=5)

    nej_button = ttk.Button(
        button_frame,
        text="Nej",
        command=on_nej_button_click,
        style="TButton"
    )
    nej_button.pack(side="left", padx=5)
    
    loading_label.config(text="Nöjd med registreringen?")
    loading_label.pack(expand=True)
    # Wait until either "Ja" or "Nej" is pressed
    initial_window.wait_variable(button_pressed)

    # Now check the button_pressed variable to handle the response
    if button_pressed.get() == "Ja":
        print("Ja button pressed, exiting loop.")
        initial_window.quit()
    elif button_pressed.get() == "Nej":
        print("Nej button pressed, re-running registration.")
        for widget in initial_window.winfo_children():
            widget.pack_forget()
        loading_label.config(text="Registrerar igen")
        loading_label.pack(expand=True)
        initial_window.update_idletasks()  # Force update of the label text
        initial_window.update()
        seed=seed+2
        registration_1,registration_2, template_3d = Registrering(data_4d, initial_window, loading_label, original_screen_width, seed)

    return registration_1, registration_2, template_3d
