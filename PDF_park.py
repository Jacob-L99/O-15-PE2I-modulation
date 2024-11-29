import os
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Table, TableStyle, Spacer, PageBreak, Paragraph, Spacer
from reportlab.lib.pagesizes import landscape, letter
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import platform
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
from matplotlib.cm import ScalarMappable
from io import BytesIO

# Define the custom colormap
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

def nii_gz_to_numpy(file_path):
    """
    Loads a .nii.gz file and converts it to a NumPy array.

    Parameters:
    - file_path (str): The path to the .nii.gz file.

    Returns:
    - data (np.ndarray): The image data as a NumPy array.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        nii_img = nib.load(file_path)
    except Exception as e:
        raise IOError(f"An error occurred while loading the NIfTI file: {e}")

    data = nii_img.get_fdata()
    return data

# Define the list of mask names in the order they appear in the NIfTI file
mask_names = [
    "R_Cingulate_Ant", "L_Cingulate_Ant", "R_Cingulate_Post", "L_Cingulate_Post",
    "R_Insula", "L_Insula", "R_Brainstem", "L_Brainstem", "R_Thalamus", "L_Thalamus",
    "R_Caudate", "L_Caudate", "R_Putamen", "L_Putamen", "R_Pallidum", "L_Pallidum",
    "R_Substantia_nigra", "L_Substantia_nigra", "R_Frontal_Lat", "L_Frontal_Lat",
    "R_Orbital", "L_Orbital", "R_Frontal_Med_Sup", "L_Frontal_Med_Sup",
    "R_Precentral", "L_Precentral", "R_Parietal_Inf", "L_Parietal_Inf",
    "R_Postcentral", "L_Postcentral", "R_Precuneus", "L_Precuneus",
    "R_Parietal_Sup", "L_Parietal_Sup", "R_Temporal_Mesial", "L_Temporal_Mesial",
    "R_Temporal_Basal", "L_Temporal_Basal", "R_Temporal_Lat_Ant", "L_Temporal_Lat_Ant",
    "R_Occipital_Med", "L_Occipital_Med", "R_Occipital_Lat", "L_Occipital_Lat",
    "R_Cerebellum", "L_Cerebellum", "R_Vermis", "L_Vermis"
]




import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

def create_custom_figure(last_figures, data_set):
    """
    Creates a custom figure by combining multiple pre-generated figures.

    Parameters:
    - last_figures (list): List of pre-generated matplotlib figures.
    - data_set (str): The dataset identifier.
    - custom_cmap (matplotlib.colors.Colormap): The colormap to use for the colorbar.

    Returns:
    - fig (matplotlib.figure.Figure): The combined custom figure.
    """
    fig = Figure(facecolor='black')


    # Create a figure with 1 row and 7 columns (6 plots + colorbar)
    gs = GridSpec(1, 7, width_ratios=[1]*6 + [0.05], figure=fig)
    axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
    cbar_ax = fig.add_subplot(gs[0, 6])

    for i, figure in enumerate(last_figures):
        ax_new = axs[i]
        for ax_old in figure.axes:
            for img in ax_old.images:
                blended_image = img.get_array()
                im = ax_new.imshow(blended_image, cmap=img.get_cmap(), alpha=img.get_alpha())
        ax_new.axis('off')
        ax_new.set_aspect('equal')

    # Adjust the colorbar axis to make it shorter
    # Get the current position of the colorbar axis
    pos = cbar_ax.get_position()
    # Define the new height (e.g., 60% of the original height)
    new_height = pos.height * 0.01
    # Calculate the new y position to center the colorbar vertically
    new_y = pos.y0 + (pos.height - new_height) / 5
    # Update the position of the colorbar axis
    cbar_ax.set_position([pos.x0, new_y, pos.width, new_height])

    
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-5, vmax=5))
    sm.set_array([])  # Necessary for ScalarMappable
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Z-score', rotation=270, labelpad=15, fontsize=8, color='white')
    cbar.ax.tick_params(labelsize=6, colors='white')
    cbar.outline.set_edgecolor('white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    fig.suptitle(f"{data_set}")
    fig.tight_layout()
    return fig

        
def create_custom_figure_SSP(first_values_yz_neg_x0, first_values_yz_pos_x1,
first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
first_values_xz_neg_y2, data_set):
    fig = Figure(facecolor='black')
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
    norm=None
    if norm is None:
        norm = Normalize(vmin=0, vmax=1.2)

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

    fig.suptitle(f"{data_set}")
    # Adjust layout
    fig.tight_layout()

    return fig # Return images for updating norm


def PDF_park(directory, BP_reshape_list, R_I_reshape_list, z_brain_regions, z_brain_flow, 
            last_figure_1, last_figure_2, last_figure_3, 
            last_figure_4, last_figure_5, last_figure_6, 
            last_figure_1_flow, last_figure_2_flow, last_figure_3_flow, 
            last_figure_4_flow, last_figure_5_flow, last_figure_6_flow,
            z_scores, z_scores_flow, vmax1, vmax2, tracer, motion_parameters_array,
            means_list_flow, mean_values_regions):
    print('vmax:', vmax1, vmax2)
    """
    Generates a PDF report with brain slice images and a table of z-scores.

    Parameters:
    - BP_reshape_list (np.ndarray): Reshaped BP data.
    - R_I_reshape_list (np.ndarray): Reshaped R_I data.
    - BP_reshape_list_pat (np.ndarray): Reshaped BP patient data.
    - R_I_reshape_list_pat (np.ndarray): Reshaped R_I patient data.
    - z_score_R_I (list): List of z-scores for regions.
    - z_min, z_med, z_max: Additional parameters (usage depends on implementation).
    - last_figure_1 to last_figure_6 (matplotlib.figure.Figure): Pre-generated figures.
    """
    # Define the output PDF file
    pdf_file = "output_with_individual_3d_slices.pdf"
    pdf = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()

    # List to hold elements (figures and table) for the PDF
    elements = []
    
    pat_dir = directory

    # Extract only the last part of the path
    pat_dir_name = os.path.basename(pat_dir)
    # Add a title
    title = Paragraph(f"{tracer}-PE2I parametriska bilder", styles['Title'])
    elements.append(title)
    
    directory_name = Paragraph(f"{pat_dir_name}", styles['Title'])
    elements.append(directory_name)
    
    # Add a spacer
    elements.append(Spacer(1, 12))
    
    # Add a spacer to create space before the footer
    elements.append(Spacer(1, 500))  # Adjust the height as needed to push content to the bottom
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add the date and time as a footer at the bottom of the page
    footer_text = Paragraph(f"Genererad: {current_time}", styles['BodyText'])
    elements.append(footer_text)
    
    elements.append(PageBreak())

    # List of volume arrays and their titles
    volume_data_MNI = [
        (BP_reshape_list, "BP (MNI)"),
        (R_I_reshape_list, "R_I (MNI)")
    ]


    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        slices = [60, 80, 100]
        
        print(title)
        if title == "BP Reshape":
            vmaxx=vmax1
        else:
            vmaxx=vmax2

        norm = Normalize(vmin=0, vmax=vmaxx)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=vmaxx)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
    
    volume_data_MNI = [
        (z_brain_regions, "Z-score (neurodegenerativ regioner)"),
        (z_brain_flow, "Z-score (flödes regioner)")
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        slices = [60, 80, 100]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)


    # Create custom figure
    last_figures = [last_figure_1, last_figure_2, last_figure_3, 
                   last_figure_4, last_figure_5, last_figure_6]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score (regions)")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [last_figure_1_flow, last_figure_2_flow, last_figure_3_flow, 
                   last_figure_4_flow, last_figure_5_flow, last_figure_6_flow]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score (flow)")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Compute z-scores (assuming z_score_R_I is already the list of z-scores)
    means = z_scores


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Medel', "Region:", "Z-score:", 'Medel']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_regions[mask_names[i]] 
                mean_2 = mean_values_regions[mask_names[i+1]]
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    

    
    mask_names_flow=['R Cerebellum', 'L Cerebellum', 'R Posterior', 'L Posterior', 'R Middle',
                     'L Middle', 'R Anterior', 'L Anterior']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Medel', "Region", "Z-score", 'Medel']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_flow[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores_flow[i+1]
            
            mean_1 = means_list_flow[i] 
            mean_2 = means_list_flow[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_flow[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    time_points = motion_parameters_array[:, 0]  # Time points
    translations = motion_parameters_array[:, 1:4]  # Tx, Ty, Tz
    
    # Create the plot
    plt.figure(figsize=(8, 4), facecolor='white')
    plt.plot(time_points, translations[:, 0], label='Tx (mm)', marker='o')
    plt.plot(time_points, translations[:, 1], label='Ty (mm)', marker='o')
    plt.plot(time_points, translations[:, 2], label='Tz (mm)', marker='o')
    plt.title('Rörelse över tid')
    plt.xlabel('Frames')
    plt.ylabel('Translation (mm)')
    plt.ylim(-10, 10)  # Set y-axis range for translations
    plt.legend()
    plt.tight_layout()
    
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=300)
    plt.close()
    img_data.seek(0)
    
    # Add the image to the PDF using the in-memory BytesIO object
    image = Image(img_data, width=400, height=200)
    elements.append(image)

    # Build the PDF with the elements
    pdf.build(elements)

    # Open the PDF file after creation
    if platform.system() == 'Darwin':       # macOS
        os.system(f'open "{pdf_file}"')
    elif platform.system() == 'Windows':    # Windows
        os.startfile(pdf_file)
    else:                                   # Linux variants
        os.system(f'xdg-open "{pdf_file}"')








       
def PDF_water(directory, transformed_K_1_1, transformed_K_2_1, K1_k2_MNI, Z_brain, Z_brain_regions, first_values_yz_neg_x0, first_values_yz_pos_x1,
          first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
          first_values_xz_neg_y2,
          last_figure_1, last_figure_2, last_figure_3,
          last_figure_4, last_figure_5, last_figure_6,
          region_last_figure_1, region_last_figure_2, region_last_figure_3, 
          region_last_figure_4, region_last_figure_5, region_last_figure_6, 
          flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, 
          flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6,
          z_scores, z_scores_flow, vmax, means_list_k1 ,mean_values_k1,
          transformed_K_1_1_ref, transformed_K_2_1_ref, K1_k2_MNI_ref, Z_brain_ref, Z_brain_regions_ref, first_values_yz_neg_x0_ref, first_values_yz_pos_x1_ref,
          first_values_yz_pos_x2_ref, first_values_yz_neg_x2_ref, first_values_xz_pos_y2_ref,
          first_values_xz_neg_y2_ref,
          last_figure_1_ref, last_figure_2_ref, last_figure_3_ref,
          last_figure_4_ref, last_figure_5_ref, last_figure_6_ref,
          region_last_figure_1_ref, region_last_figure_2_ref, region_last_figure_3_ref, 
          region_last_figure_4_ref, region_last_figure_5_ref, region_last_figure_6_ref, 
          flow_region_last_figure_1_ref, flow_region_last_figure_2_ref, flow_region_last_figure_3_ref, 
          flow_region_last_figure_4_ref, flow_region_last_figure_5_ref, flow_region_last_figure_6_ref,
          z_scores_ref, z_scores_flow_ref, means_list_k1_ref ,mean_values_k1_ref,
          motion_parameters_array_1):
    

    vmax1=vmax
    

    """
    Generates a PDF report with brain slice images and a table of z-scores.

    Parameters:
    - BP_reshape_list (np.ndarray): Reshaped BP data.
    - R_I_reshape_list (np.ndarray): Reshaped R_I data.
    - BP_reshape_list_pat (np.ndarray): Reshaped BP patient data.
    - R_I_reshape_list_pat (np.ndarray): Reshaped R_I patient data.
    - z_score_R_I (list): List of z-scores for regions.
    - z_min, z_med, z_max: Additional parameters (usage depends on implementation).
    - last_figure_1 to last_figure_6 (matplotlib.figure.Figure): Pre-generated figures.
    """
    # Define the output PDF file
    pdf_file = "output_with_individual_3d_slices_flow.pdf"
    pdf = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    # List to hold elements (figures and table) for the PDF
    elements = []
    
    pat_dir = directory

    # Extract only the last part of the path
    pat_dir_name = os.path.basename(pat_dir)
    # Add a title
    title = Paragraph("O-15-vatten parametriska bilder", styles['Title'])
    elements.append(title)
    
    directory_name = Paragraph(f"{pat_dir_name}", styles['Title'])
    elements.append(directory_name)
    
    # Add a spacer
    elements.append(Spacer(1, 12))
    
    # Add a spacer to create space before the footer
    elements.append(Spacer(1, 500))  # Adjust the height as needed to push content to the bottom
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add the date and time as a footer at the bottom of the page
    footer_text = Paragraph(f"Genererad: {current_time}", styles['BodyText'])
    elements.append(footer_text)
    
    elements.append(PageBreak())

    # List of volume arrays and their titles
    volume_data_MNI = [
        (transformed_K_1_1, "Perfusion (MNI)"),
        (transformed_K_2_1, "Flow-out rate (MNI)"),
        (K1_k2_MNI, "K_2/K_1 (MNI)"),
    ]

    
    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]

        norm = Normalize(vmin=0, vmax=vmax1)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=vmax1)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain, "z-score (flödes regioner)"),
        (Z_brain_regions, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0, first_values_yz_pos_x1,
    first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
    first_values_xz_neg_y2, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1, last_figure_2, last_figure_3, 
                   last_figure_4, last_figure_5, last_figure_6]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1, region_last_figure_2, region_last_figure_3, 
                    region_last_figure_4, region_last_figure_5, region_last_figure_6]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, 
                    flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Medel', "Region:", "Z-score:", 'Medel']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1[mask_names[i]] 
                mean_2 = mean_values_k1[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    means=z_scores_flow
    
    means = z_scores_flow
    
    mask_names_flow=['R Cerebellum', 'L Cerebellum', 'R Posterior', 'L Posterior', 'R Middle',
                     'L Middle', 'R Anterior', 'L Anterior']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Medel', "Region", "Z-score", 'Medel']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores[i+1]
            
            mean_1 = means_list_k1[i] 
            mean_2 = means_list_k1[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    #------------
    #ref
    #------------
    
    elements.append(PageBreak())
    
    # Add a title
    title = Paragraph("Reference", styles['Title'])
    elements.append(title)
    
    
    volume_data_MNI = [
        (transformed_K_1_1_ref, "Perfusion (MNI)"),
        (transformed_K_2_1_ref, "Flow-out rate (MNI)"),
        (K1_k2_MNI_ref, "K_2/K_1 (MNI)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=0, vmax=2)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])


        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=2)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain_ref, "z-score (flödes regioner)"),
        (Z_brain_regions_ref, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0_ref, first_values_yz_pos_x1_ref,
    first_values_yz_pos_x2_ref, first_values_yz_neg_x2_ref, first_values_xz_pos_y2_ref,
    first_values_xz_neg_y2_ref, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1_ref, last_figure_2_ref, last_figure_3_ref, 
                   last_figure_4_ref, last_figure_5_ref, last_figure_6_ref]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1_ref, region_last_figure_2_ref, region_last_figure_3_ref, 
                    region_last_figure_4_ref, region_last_figure_5_ref, region_last_figure_6_ref]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1_ref, flow_region_last_figure_2_ref, flow_region_last_figure_3_ref, 
                    flow_region_last_figure_4_ref, flow_region_last_figure_5_ref, flow_region_last_figure_6_ref]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Mean', "Region:", "Z-score:", 'Mean']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1_ref[mask_names[i]] 
                mean_2 = mean_values_k1_ref[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores_ref)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    
    means = z_scores_flow_ref
    
    mask_names_flow=['Cerebellum_dex', 'Cerebellum sin', 'Posterior dex', 'Posterior sin', 'Middle dex',
                     'Middle sin', 'Anterior dex', 'Anterior sin']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Mean', "Region", "Z-score", 'Mean']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores_ref[i+1]
            
            mean_1 = means_list_k1_ref[i] 
            mean_2 = means_list_k1_ref[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    time_points = motion_parameters_array_1[:, 0]  # Time points
    translations = motion_parameters_array_1[:, 1:4]  # Tx, Ty, Tz
    
    # Create the plot
    plt.figure(figsize=(8, 4), facecolor='white')
    plt.plot(time_points, translations[:, 0], label='Tx (mm)', marker='o')
    plt.plot(time_points, translations[:, 1], label='Ty (mm)', marker='o')
    plt.plot(time_points, translations[:, 2], label='Tz (mm)', marker='o')
    plt.title('Rörelse över tid')
    plt.xlabel('Frames')
    plt.ylabel('Translation (mm)')
    plt.ylim(-10, 10)  # Set y-axis range for translations
    plt.legend()
    plt.tight_layout()
    
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=300)
    plt.close()
    img_data.seek(0)
    
    # Add the image to the PDF using the in-memory BytesIO object
    image = Image(img_data, width=400, height=200)
    elements.append(image)


    # Build the PDF with the elements
    pdf.build(elements)

    # Open the PDF file after creation
    if platform.system() == 'Darwin':       # macOS
        os.system(f'open "{pdf_file}"')
    elif platform.system() == 'Windows':    # Windows
        os.startfile(pdf_file)
    else:                                   # Linux variants
        os.system(f'xdg-open "{pdf_file}"')




def PDF_water_wat1_2(directory, transformed_K_1_1, transformed_K_2_1, K1_k2_MNI, Z_brain, Z_brain_regions, first_values_yz_neg_x0, first_values_yz_pos_x1,
            first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
            first_values_xz_neg_y2,
            last_figure_1, last_figure_2, last_figure_3,
            last_figure_4, last_figure_5, last_figure_6,
            region_last_figure_1, region_last_figure_2, region_last_figure_3, 
            region_last_figure_4, region_last_figure_5, region_last_figure_6, 
            flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, 
            flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6,
            z_scores, z_scores_flow, vmax, means_list_k1 ,mean_values_k1,
            transformed_K_1_1_ref, transformed_K_2_1_ref, K1_k2_MNI_ref, Z_brain_ref, Z_brain_regions_ref, first_values_yz_neg_x0_ref, first_values_yz_pos_x1_ref,
            first_values_yz_pos_x2_ref, first_values_yz_neg_x2_ref, first_values_xz_pos_y2_ref,
            first_values_xz_neg_y2_ref,
            last_figure_1_ref, last_figure_2_ref, last_figure_3_ref,
            last_figure_4_ref, last_figure_5_ref, last_figure_6_ref,
            region_last_figure_1_ref, region_last_figure_2_ref, region_last_figure_3_ref, 
            region_last_figure_4_ref, region_last_figure_5_ref, region_last_figure_6_ref, 
            flow_region_last_figure_1_ref, flow_region_last_figure_2_ref, flow_region_last_figure_3_ref, 
            flow_region_last_figure_4_ref, flow_region_last_figure_5_ref, flow_region_last_figure_6_ref,
            z_scores_ref, z_scores_flow_ref, means_list_k1_ref ,mean_values_k1_ref,
            motion_parameters_array_1,
          
            transformed_K_1_2, transformed_K_2_2, K1_k2_MNI_2, Z_brain_2, Z_brain_regions_2, first_values_yz_neg_x0_2, first_values_yz_pos_x1_2,
            first_values_yz_pos_x2_2, first_values_yz_neg_x2_2, first_values_xz_pos_y2_2,
            first_values_xz_neg_y2_2,
            last_figure_1_2, last_figure_2_2, last_figure_3_2,
            last_figure_4_2, last_figure_5_2, last_figure_6_2,
            region_last_figure_1_2, region_last_figure_2_2, region_last_figure_3_2, 
            region_last_figure_4_2, region_last_figure_5_2, region_last_figure_6_2, 
            flow_region_last_figure_1_2, flow_region_last_figure_2_2, flow_region_last_figure_3_2, 
            flow_region_last_figure_4_2, flow_region_last_figure_5_2, flow_region_last_figure_6_2,
            z_scores_2, z_scores_flow_2, means_list_k1_2 ,mean_values_k1_2,
            transformed_K_1_1_ref_2, transformed_K_2_1_ref_2, K1_k2_MNI_ref_2, Z_brain_ref_2, Z_brain_regions_ref_2, first_values_yz_neg_x0_ref_2, first_values_yz_pos_x1_ref_2,
            first_values_yz_pos_x2_ref_2, first_values_yz_neg_x2_ref_2, first_values_xz_pos_y2_ref_2,
            first_values_xz_neg_y2_ref_2,
            last_figure_1_ref_2, last_figure_2_ref_2, last_figure_3_ref_2,
            last_figure_4_ref_2, last_figure_5_ref_2, last_figure_6_ref_2,
            region_last_figure_1_ref_2, region_last_figure_2_ref_2, region_last_figure_3_ref_2, 
            region_last_figure_4_ref_2, region_last_figure_5_ref_2, region_last_figure_6_ref_2, 
            flow_region_last_figure_1_ref_2, flow_region_last_figure_2_ref_2, flow_region_last_figure_3_ref_2, 
            flow_region_last_figure_4_ref_2, flow_region_last_figure_5_ref_2, flow_region_last_figure_6_ref_2,
            z_scores_ref_2, z_scores_flow_ref_2, means_list_k1_ref_2 ,mean_values_k1_ref_2,
            motion_parameters_array_2,
  
            transformed_K_1_3, transformed_K_2_3, K1_k2_MNI_3, Z_brain_3, Z_brain_regions_3, first_values_yz_neg_x0_3, first_values_yz_pos_x1_3,
            first_values_yz_pos_x2_3, first_values_yz_neg_x2_3, first_values_xz_pos_y2_3,
            first_values_xz_neg_y2_3,
            last_figure_1_3, last_figure_2_3, last_figure_3_3,
            last_figure_4_3, last_figure_5_3, last_figure_6_3,
            region_last_figure_1_3, region_last_figure_2_3, region_last_figure_3_3, 
            region_last_figure_4_3, region_last_figure_5_3, region_last_figure_6_3, 
            flow_region_last_figure_1_3, flow_region_last_figure_2_3, flow_region_last_figure_3_3, 
            flow_region_last_figure_4_3, flow_region_last_figure_5_3, flow_region_last_figure_6_3,
            z_scores_3, z_scores_flow_3, means_list_k1_3 ,mean_values_k1_3,
            transformed_K_1_1_ref_3, transformed_K_3_1_ref_3, K1_k2_MNI_ref_3, Z_brain_ref_3, Z_brain_regions_ref_3, first_values_yz_neg_x0_ref_3, first_values_yz_pos_x1_ref_3,
            first_values_yz_pos_x2_ref_3, first_values_yz_neg_x2_ref_3, first_values_xz_pos_y2_ref_3,
            first_values_xz_neg_y2_ref_3,
            last_figure_1_ref_3, last_figure_2_ref_3, last_figure_3_ref_3,
            last_figure_4_ref_3, last_figure_5_ref_3, last_figure_6_ref_3,
            region_last_figure_1_ref_3, region_last_figure_2_ref_3, region_last_figure_3_ref_3, 
            region_last_figure_4_ref_3, region_last_figure_5_ref_3, region_last_figure_6_ref_3, 
            flow_region_last_figure_1_ref_3, flow_region_last_figure_2_ref_3, flow_region_last_figure_3_ref_3, 
            flow_region_last_figure_4_ref_3, flow_region_last_figure_5_ref_3, flow_region_last_figure_6_ref_3,
            z_scores_ref_3, z_scores_flow_ref_3, means_list_k1_ref_3 ,mean_values_k1_ref_3
          ):
    # np.savez('mean_values_k1.npz', mean_values_k1)
    vmax1=vmax
    # print(type(mean_values_k1))
    mean_values_k1 = mean_values_k1[0]
    # print(type(mean_values_k1))
    # print(type(mean_values_k1_ref))
    
    # print(type(mean_values_k1_2))
    # print(type(mean_values_k1_ref_2))
    
    # print(type(mean_values_k1_3))
    # print(type(mean_values_k1_ref_3))
    # mean_values_k1_ref = mean_values_k1_ref[0]
    
    # mean_values_k1_2 = mean_values_k1_2[0]
    # mean_values_k1_ref_2 = mean_values_k1_ref_2[0]
    
    # mean_values_k1_3 = mean_values_k1_3[0]
    # mean_values_k1_ref_3 = mean_values_k1_ref_3[0]
    


    

    """
    Generates a PDF report with brain slice images and a table of z-scores.

    Parameters:
    - BP_reshape_list (np.ndarray): Reshaped BP data.
    - R_I_reshape_list (np.ndarray): Reshaped R_I data.
    - BP_reshape_list_pat (np.ndarray): Reshaped BP patient data.
    - R_I_reshape_list_pat (np.ndarray): Reshaped R_I patient data.
    - z_score_R_I (list): List of z-scores for regions.
    - z_min, z_med, z_max: Additional parameters (usage depends on implementation).
    - last_figure_1 to last_figure_6 (matplotlib.figure.Figure): Pre-generated figures.
    """
    # Define the output PDF file
    pdf_file = "output_with_individual_3d_slices_flow.pdf"
    pdf = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    # List to hold elements (figures and table) for the PDF
    elements = []
    
    pat_dir = directory

    # Extract only the last part of the path
    pat_dir_name = os.path.basename(pat_dir)
    # Add a title
    title = Paragraph("O-15-vatten parametriska bilder", styles['Title'])
    elements.append(title)
    
    directory_name = Paragraph(f"{pat_dir_name}", styles['Title'])
    elements.append(directory_name)
    
    # Add a spacer
    elements.append(Spacer(1, 12))
    
    # Add a spacer to create space before the footer
    elements.append(Spacer(1, 500))  # Adjust the height as needed to push content to the bottom
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add the date and time as a footer at the bottom of the page
    footer_text = Paragraph(f"Genererad: {current_time}", styles['BodyText'])
    elements.append(footer_text)
    
    elements.append(PageBreak())

    # List of volume arrays and their titles
    volume_data_MNI = [
        (transformed_K_1_1, "Perfusion (MNI)"),
        (transformed_K_2_1, "Flow-out rate (MNI)"),
        (K1_k2_MNI, "Volume of Distribution (MNI)"),
    ]

    
    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]

        norm = Normalize(vmin=0, vmax=vmax1)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=vmax1)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain, "z-score (flödes regioner)"),
        (Z_brain_regions, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0, first_values_yz_pos_x1,
    first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2,
    first_values_xz_neg_y2, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1, last_figure_2, last_figure_3, 
                   last_figure_4, last_figure_5, last_figure_6]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1, region_last_figure_2, region_last_figure_3, 
                    region_last_figure_4, region_last_figure_5, region_last_figure_6]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1, flow_region_last_figure_2, flow_region_last_figure_3, 
                    flow_region_last_figure_4, flow_region_last_figure_5, flow_region_last_figure_6]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Mean', "Region:", "Z-score:", 'Mean']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1[mask_names[i]] 
                mean_2 = mean_values_k1[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    means=z_scores_flow
    
    means = z_scores_flow
    
    mask_names_flow=['Cerebellum_dex', 'Cerebellum sin', 'Posterior dex', 'Posterior sin', 'Middle dex',
                     'Middle sin', 'Anterior dex', 'Anterior sin']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Mean', "Region", "Z-score", 'Mean']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores[i+1]
            
            mean_1 = means_list_k1[i] 
            mean_2 = means_list_k1[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    #------------
    #ref
    #------------
    
    elements.append(PageBreak())
    
    # Add a title
    title = Paragraph("Reference", styles['Title'])
    elements.append(title)
    
    
    volume_data_MNI = [
        (transformed_K_1_1_ref, "Perfusion (MNI)"),
        (transformed_K_2_1_ref, "Flow-out rate (MNI)"),
        (K1_k2_MNI_ref, "Volume of Distribution (MNI)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=0, vmax=2)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])


        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=2)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain_ref, "z-score (flödes regioner)"),
        (Z_brain_regions_ref, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0_ref, first_values_yz_pos_x1_ref,
    first_values_yz_pos_x2_ref, first_values_yz_neg_x2_ref, first_values_xz_pos_y2_ref,
    first_values_xz_neg_y2_ref, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1_ref, last_figure_2_ref, last_figure_3_ref, 
                   last_figure_4_ref, last_figure_5_ref, last_figure_6_ref]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1_ref, region_last_figure_2_ref, region_last_figure_3_ref, 
                    region_last_figure_4_ref, region_last_figure_5_ref, region_last_figure_6_ref]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1_ref, flow_region_last_figure_2_ref, flow_region_last_figure_3_ref, 
                    flow_region_last_figure_4_ref, flow_region_last_figure_5_ref, flow_region_last_figure_6_ref]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Mean', "Region:", "Z-score:", 'Mean']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1_ref[mask_names[i]] 
                mean_2 = mean_values_k1_ref[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores_ref)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    
    means = z_scores_flow_ref
    
    mask_names_flow=['Cerebellum_dex', 'Cerebellum sin', 'Posterior dex', 'Posterior sin', 'Middle dex',
                     'Middle sin', 'Anterior dex', 'Anterior sin']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Mean', "Region", "Z-score", 'Mean']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores_ref[i+1]
            
            mean_1 = means_list_k1_ref[i] 
            mean_2 = means_list_k1_ref[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    time_points = motion_parameters_array_1[:, 0]  # Time points
    translations = motion_parameters_array_1[:, 1:4]  # Tx, Ty, Tz
    
    # Create the plot
    plt.figure(figsize=(8, 4), facecolor='white')
    plt.plot(time_points, translations[:, 0], label='Tx (mm)', marker='o')
    plt.plot(time_points, translations[:, 1], label='Ty (mm)', marker='o')
    plt.plot(time_points, translations[:, 2], label='Tz (mm)', marker='o')
    plt.title('Rörelse över tid')
    plt.xlabel('Frames')
    plt.ylabel('Translation (mm)')
    plt.ylim(-10, 10)  # Set y-axis range for translations
    plt.legend()
    plt.tight_layout()
    
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=300)
    plt.close()
    img_data.seek(0)
    
    # Add the image to the PDF using the in-memory BytesIO object
    image = Image(img_data, width=400, height=200)
    elements.append(image)
    
    
    #------------wat2---------------------------
    vmax1=vmax
    

    """
    Generates a PDF report with brain slice images and a table of z-scores.

    Parameters:
    - BP_reshape_list (np.ndarray): Reshaped BP data.
    - R_I_reshape_list (np.ndarray): Reshaped R_I data.
    - BP_reshape_list_pat (np.ndarray): Reshaped BP patient data.
    - R_I_reshape_list_pat (np.ndarray): Reshaped R_I patient data.
    - z_score_R_I (list): List of z-scores for regions.
    - z_min, z_med, z_max: Additional parameters (usage depends on implementation).
    - last_figure_1 to last_figure_6 (matplotlib.figure.Figure): Pre-generated figures.
    """

    elements.append(PageBreak())


    # Add a title
    title = Paragraph("Stress", styles['Title'])
    elements.append(title)
    

    # List of volume arrays and their titles
    volume_data_MNI = [
        (transformed_K_1_2, "Perfusion (MNI)"),
        (transformed_K_2_2, "Flow-out rate (MNI)"),
        (K1_k2_MNI_2, "Volume of Distribution (MNI)"),
    ]

    
    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]

        norm = Normalize(vmin=0, vmax=vmax1)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=vmax1)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain_2, "z-score (flödes regioner)"),
        (Z_brain_regions_2, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0_2, first_values_yz_pos_x1_2,
    first_values_yz_pos_x2_2, first_values_yz_neg_x2_2, first_values_xz_pos_y2_2,
    first_values_xz_neg_y2_2, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1_2, last_figure_2_2, last_figure_3_2, 
                   last_figure_4_2, last_figure_5_2, last_figure_6_2]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1_2, region_last_figure_2_2, region_last_figure_3_2, 
                    region_last_figure_4_2, region_last_figure_5_2, region_last_figure_6_2]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1_2, flow_region_last_figure_2_2, flow_region_last_figure_3_2, 
                    flow_region_last_figure_4_2, flow_region_last_figure_5_2, flow_region_last_figure_6_2]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores_2


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Mean', "Region:", "Z-score:", 'Mean']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1_2[mask_names[i]] 
                mean_2 = mean_values_k1_2[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores_2[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores_2)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    
    means = z_scores_flow_2
    
    mask_names_flow=['Cerebellum_dex', 'Cerebellum sin', 'Posterior dex', 'Posterior sin', 'Middle dex',
                     'Middle sin', 'Anterior dex', 'Anterior sin']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Mean', "Region", "Z-score", 'Mean']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_2[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores_2[i+1]
            
            mean_1 = means_list_k1_2[i] 
            mean_2 = means_list_k1_2[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_2[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    #------------
    #ref
    #------------
    
    elements.append(PageBreak())
    
    # Add a title
    title = Paragraph("Stress Reference", styles['Title'])
    elements.append(title)
    
    
    volume_data_MNI = [
        (transformed_K_1_1_ref_2, "Perfusion (MNI)"),
        (transformed_K_2_1_ref_2, "Flow-out rate (MNI)"),
        (K1_k2_MNI_ref_2, "Volume of Distribution (MNI)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=0, vmax=2)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])


        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=2)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain_ref_2, "z-score (flödes regioner)"),
        (Z_brain_regions_ref_2, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0_ref_2, first_values_yz_pos_x1_ref_2,
    first_values_yz_pos_x2_ref_2, first_values_yz_neg_x2_ref_2, first_values_xz_pos_y2_ref_2,
    first_values_xz_neg_y2_ref_2, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1_ref_2, last_figure_2_ref_2, last_figure_3_ref_2, 
                   last_figure_4_ref_2, last_figure_5_ref_2, last_figure_6_ref_2]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1_ref_2, region_last_figure_2_ref_2, region_last_figure_3_ref_2, 
                    region_last_figure_4_ref_2, region_last_figure_5_ref_2, region_last_figure_6_ref_2]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1_ref_2, flow_region_last_figure_2_ref_2, flow_region_last_figure_3_ref_2, 
                    flow_region_last_figure_4_ref_2, flow_region_last_figure_5_ref_2, flow_region_last_figure_6_ref_2]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores_2


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Mean', "Region:", "Z-score:", 'Mean']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1_ref_2[mask_names[i]] 
                mean_2 = mean_values_k1_ref_2[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores_2[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores_ref_2)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    
    means = z_scores_flow_ref_2
    
    mask_names_flow=['Cerebellum_dex', 'Cerebellum sin', 'Posterior dex', 'Posterior sin', 'Middle dex',
                     'Middle sin', 'Anterior dex', 'Anterior sin']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Mean', "Region", "Z-score", 'Mean']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref_2[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores_ref_2[i+1]
            
            mean_1 = means_list_k1_ref_2[i] 
            mean_2 = means_list_k1_ref_2[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref_2[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    time_points = motion_parameters_array_2[:, 0]  # Time points
    translations = motion_parameters_array_2[:, 1:4]  # Tx, Ty, Tz
    
    # Create the plot
    plt.figure(figsize=(8, 4), facecolor='white')
    plt.plot(time_points, translations[:, 0], label='Tx (mm)', marker='o')
    plt.plot(time_points, translations[:, 1], label='Ty (mm)', marker='o')
    plt.plot(time_points, translations[:, 2], label='Tz (mm)', marker='o')
    plt.title('Rörelse över tid')
    plt.xlabel('Frames')
    plt.ylabel('Translation (mm)')
    plt.ylim(-10, 10)  # Set y-axis range for translations
    plt.legend()
    plt.tight_layout()
    
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=300)
    plt.close()
    img_data.seek(0)
    
    # Add the image to the PDF using the in-memory BytesIO object
    image = Image(img_data, width=400, height=200)
    elements.append(image)
    
    #----------------wat2/wat1-------------------------------
    vmax1=vmax
    

    """
    Generates a PDF report with brain slice images and a table of z-scores.

    Parameters:
    - BP_reshape_list (np.ndarray): Reshaped BP data.
    - R_I_reshape_list (np.ndarray): Reshaped R_I data.
    - BP_reshape_list_pat (np.ndarray): Reshaped BP patient data.
    - R_I_reshape_list_pat (np.ndarray): Reshaped R_I patient data.
    - z_score_R_I (list): List of z-scores for regions.
    - z_min, z_med, z_max: Additional parameters (usage depends on implementation).
    - last_figure_1 to last_figure_6 (matplotlib.figure.Figure): Pre-generated figures.
    """

    elements.append(PageBreak())

    # Add a title
    title = Paragraph("Stress/baseline", styles['Title'])
    elements.append(title)
    


    # List of volume arrays and their titles
    volume_data_MNI = [
        (transformed_K_1_3, "Perfusion (MNI)"),
        (transformed_K_2_3, "Flow-out rate (MNI)"),
        (K1_k2_MNI_3, "Volume of Distribution (MNI)"),
    ]

    
    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]

        norm = Normalize(vmin=0, vmax=vmax1)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=vmax1)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain_3, "z-score (flödes regioner)"),
        (Z_brain_regions_3, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0_3, first_values_yz_pos_x1_3,
    first_values_yz_pos_x2_3, first_values_yz_neg_x2_3, first_values_xz_pos_y2_3,
    first_values_xz_neg_y2_3, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1_3, last_figure_2_3, last_figure_3_3, 
                   last_figure_4_3, last_figure_5_3, last_figure_6_3]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1_3, region_last_figure_2_3, region_last_figure_3_3, 
                    region_last_figure_4_3, region_last_figure_5_3, region_last_figure_6_3]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1_3, flow_region_last_figure_2_3, flow_region_last_figure_3_3, 
                    flow_region_last_figure_4_3, flow_region_last_figure_5_3, flow_region_last_figure_6_3]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores_3


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Mean', "Region:", "Z-score:", 'Mean']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1_3[mask_names[i]] 
                mean_2 = mean_values_k1_3[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores_3[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores_3)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    
    means = z_scores_flow_3
    
    mask_names_flow=['Cerebellum_dex', 'Cerebellum sin', 'Posterior dex', 'Posterior sin', 'Middle dex',
                     'Middle sin', 'Anterior dex', 'Anterior sin']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Mean', "Region", "Z-score", 'Mean']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_3[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores_3[i+1]
            
            mean_1 = means_list_k1_3[i] 
            mean_2 = means_list_k1_3[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_3[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
    #------------
    #ref
    #------------
    
    elements.append(PageBreak())
    
    # Add a title
    title = Paragraph("Stress/Baseline Reference", styles['Title'])
    elements.append(title)
    
    
    volume_data_MNI = [
        (transformed_K_1_1_ref_3, "Perfusion (MNI)"),
        (transformed_K_3_1_ref_3, "Flow-out rate (MNI)"),
        (K1_k2_MNI_ref_3, "Volume of Distribution (MNI)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=0, vmax=2)
        sm = ScalarMappable(cmap=my_cmap, norm=norm)
        sm.set_array([])


        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=my_cmap, vmin=0, vmax=2)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)
        
        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)
        
   
        
   
    
    volume_data_MNI = [
        (Z_brain_ref_3, "z-score (flödes regioner)"),
        (Z_brain_regions_ref_3, "z-score (neurodegenerativ regioner)"),
    ]

    # Generate separate figures with slices 60, 80, 100
    for volume, title in volume_data_MNI:
        fig, axs = plt.subplots(1, 6, figsize=(12, 4))
        slices = [28, 42, 56, 70, 84, 98]
        
        norm = Normalize(vmin=-5, vmax=5)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        for idx, slice_index in enumerate(slices):
            if slice_index >= volume.shape[2]:
                print(f"Slice index {slice_index} is out of bounds for volume with shape {volume.shape}")
                axs[idx].set_visible(False)
                continue
            axs[idx].imshow(volume[:, :, slice_index], cmap=custom_cmap, vmin=-5, vmax=5)
            axs[idx].axis("off")
            
        cbar_ax = fig.add_axes([0.92, 0.2, 0.005, 0.6])  # Adjust the position as needed
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{title}")
        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        plt.close(fig)

        image = Image(img_data, width=500, height=180)
        elements.append(image)

    custom_fig = create_custom_figure_SSP(first_values_yz_neg_x0_ref_3, first_values_yz_pos_x1_ref_3,
    first_values_yz_pos_x2_ref_3, first_values_yz_neg_x2_ref_3, first_values_xz_pos_y2_ref_3,
    first_values_xz_neg_y2_ref_3, data_set="SSP")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)

    # Create custom figure
    last_figures = [last_figure_1_ref_3, last_figure_2_ref_3, last_figure_3_ref_3, 
                   last_figure_4_ref_3, last_figure_5_ref_3, last_figure_6_ref_3]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [region_last_figure_1_ref_3, region_last_figure_2_ref_3, region_last_figure_3_ref_3, 
                    region_last_figure_4_ref_3, region_last_figure_5_ref_3, region_last_figure_6_ref_3]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    # Create custom figure
    last_figures = [flow_region_last_figure_1_ref_3, flow_region_last_figure_2_ref_3, flow_region_last_figure_3_ref_3, 
                    flow_region_last_figure_4_ref_3, flow_region_last_figure_5_ref_3, flow_region_last_figure_6_ref_3]
    custom_fig = create_custom_figure(last_figures, data_set="SSP Z-score")

    img_data = BytesIO()
    custom_fig.savefig(img_data, format="png")
    img_data.seek(0)
    plt.close(custom_fig)

    image = Image(img_data, width=500, height=300)
    elements.append(image)
    
    
    means = z_scores_3


    
    def prepare_paired_table_data(mask_names, z_scores):
        table_data = [["Region:", "Z-score:", 'Mean', "Region:", "Z-score:", 'Mean']]
        print(f'z-score: {z_scores[2]}')
        # Pair the regions and z-scores
        for i in range(0, len(mask_names), 2):
            if i+1 < len(mask_names):
                # Handle pairs
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                mask_name_2 = mask_names[i+1].replace("_", " ")
                z_score_2 = z_scores[i+1]
                
                mean_1 = mean_values_k1_ref_3[mask_names[i]] 
                mean_2 = mean_values_k1_ref_3[mask_names[i+1]]
                
                # # Ensure z_scores are scalars
                # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
                # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
                
                # Format z-scores
                z_str_1 = f"{z_score_1:.2f}" 
                z_str_2 = f"{z_score_2:.2f}" 
                
                mean_str_1 = f'{mean_1:.2f}'
                mean_str_2 = f'{mean_2:.2f}'
                
                
                table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
            else:
                # Handle the last unpaired region
                mask_name_1 = mask_names[i].replace("_", " ")
                z_score_1 = z_scores[i]
                z_str_1 = f"{z_score_1:.2f}" 
                table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region
        
        return table_data

    # Prepare the paired table data
    print(f'z-score: {z_scores_3[2]}')
    paired_data = prepare_paired_table_data(mask_names, z_scores_ref_3)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    paired_table = Table(paired_data, colWidths=col_widths, repeatRows=1)

    # Define the table style
    paired_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),  # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),  # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    # Create the PDF document with landscape orientation and adjusted margins
    doc = SimpleDocTemplate(
        "regions_zscores.pdf",
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    elements.append(PageBreak())
    # Add the table to the elements
    elements.append(paired_table)
    
    
    means = z_scores_flow_ref_3
    
    mask_names_flow=['Cerebellum_dex', 'Cerebellum sin', 'Posterior dex', 'Posterior sin', 'Middle dex',
                     'Middle sin', 'Anterior dex', 'Anterior sin']

    # Prepare table data
    table_data = [["Region", "Z-score", 'Mean', "Region", "Z-score", 'Mean']]
    for i in range(0, len(mask_names_flow), 2):
        if i+1 < len(mask_names_flow):
            # Handle pairs
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref_3[i]
            mask_name_2 = mask_names_flow[i+1].replace("_", " ")
            z_score_2 = z_scores_ref_3[i+1]
            
            mean_1 = means_list_k1_ref_3[i] 
            mean_2 = means_list_k1_ref_3[i+1]
            
            # # Ensure z_scores are scalars
            # z_score_1 = z_score_1.item() if isinstance(z_score_1, np.ndarray) and z_score_1.size == 1 else np.nan
            # z_score_2 = z_score_2.item() if isinstance(z_score_2, np.ndarray) and z_score_2.size == 1 else np.nan
            
            # Format z-scores
            z_str_1 = f"{z_score_1:.2f}" 
            z_str_2 = f"{z_score_2:.2f}" 
            
            mean_str_1 = f'{mean_1:.2f}'
            mean_str_2 = f'{mean_2:.2f}'
            
            
            table_data.append([mask_name_2, z_str_1, mean_str_1, mask_name_1, z_str_2, mean_str_2])
        else:
            # Handle the last unpaired region
            mask_name_1 = mask_names_flow[i].replace("_", " ")
            z_score_1 = z_scores_ref_3[i]
            z_str_1 = f"{z_score_1:.2f}" 
            table_data.append([mask_name_1, z_str_1, "", ""])  # Empty cells for the unpaired region

    # paired_data = prepare_paired_table_data(mask_names, z_scores)

    # Define column widths (adjust as needed to fit your layout)
    col_widths = [140, 70, 70, 140, 70, 70]  # Total width: 560 points

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style
    # table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.black),
    ]))

    elements.append(table)
    
  

    # Build the PDF with the elements
    pdf.build(elements)

    # Open the PDF file after creation
    if platform.system() == 'Darwin':       # macOS
        os.system(f'open "{pdf_file}"')
    elif platform.system() == 'Windows':    # Windows
        os.startfile(pdf_file)
    else:                                   # Linux variants
        os.system(f'xdg-open "{pdf_file}"')
