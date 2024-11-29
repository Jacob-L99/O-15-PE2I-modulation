# import numpy as np
# from scipy.spatial.transform import Rotation as R_sc
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from scipy.linalg import orthogonal_procrustes
# import logging

# # Configure logging for detailed debugging
# logging.basicConfig(filename='processing.log', level=logging.DEBUG, 
#                     format='%(asctime)s %(levelname)s:%(message)s')

# def create_seven_dots(array_shape, distance=10):
#     """
#     Create a seven-dot configuration in the center of a 3D array.
    
#     Parameters:
#     array_shape (tuple): Shape of the array (x, y, z)
#     distance (int): Distance from the center to peripheral dots
    
#     Returns:
#     numpy.ndarray: 3D array with seven dots in the center
#     """
#     A = np.zeros(array_shape, dtype=int)
#     x, y, z = array_shape
#     center = (x // 2, y // 2, z // 2)
    
#     # Center dot
#     A[center[0], center[1], center[2]] = 1
    
#     # Peripheral dots in a hexagonal pattern around the center
#     peripheral_points = calculate_peripheral_dots(center, distance, num_dots=6)
#     for point in peripheral_points:
#         A[point] = 1
    
#     return A

# def calculate_peripheral_dots(center, distance, num_dots=6):
#     """
#     Calculate the coordinates of peripheral dots arranged in a hexagonal pattern.
    
#     Parameters:
#     center (tuple): Coordinates of the center dot (x, y, z)
#     distance (int): Distance from the center to each peripheral dot
#     num_dots (int): Number of peripheral dots (default is 6)
    
#     Returns:
#     list of tuples: Coordinates of peripheral dots
#     """
#     peripheral_points = []
#     for i in range(num_dots):
#         angle_deg = (360 / num_dots) * i
#         angle_rad = np.radians(angle_deg)
#         x = center[0] + distance * np.cos(angle_rad)
#         y = center[1] + distance * np.sin(angle_rad)
#         z = center[2]  # Assuming all dots lie in the same Z-plane
#         peripheral_points.append((int(round(x)), int(round(y)), int(round(z))))
#     return peripheral_points

# def extract_and_cluster_points(A, threshold=0, eps=1.5, min_samples=1, pixel_spacing=1.9, slice_thickness=2.8):
#     """
#     Extract the coordinates of points in a 3D array that are greater than a threshold,
#     cluster them to handle multiple pixels per dot, and scale to real-world units.
    
#     Parameters:
#     A (numpy.ndarray): 3D array
#     threshold (int or float): Threshold value to consider a point as part of the configuration
#     eps (float): Maximum distance between two samples for them to be considered in the same neighborhood
#     min_samples (int): Minimum number of samples in a neighborhood for a point to be considered a core point
#     pixel_spacing (float): Distance between pixels in X and Y dimensions
#     slice_thickness (float): Distance between slices in Z dimension
    
#     Returns:
#     numpy.ndarray: Array of cluster centroids with shape (N, 3) in real-world units
#     """
#     points = np.argwhere(A > threshold)
#     if points.size == 0:
#         logging.warning("No points found above the threshold.")
#         return np.array([])  # No points above threshold
    
#     # Perform DBSCAN clustering
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
#     labels = clustering.labels_
    
#     unique_labels = set(labels)
#     centroids = []
#     for label in unique_labels:
#         if label == -1:
#             # Noise points (if any)
#             logging.debug("Noise points detected and ignored.")
#             continue
#         cluster_points = points[labels == label]
#         centroid = cluster_points.mean(axis=0)
#         # Scale to real-world units
#         centroid_real = np.array([
#             centroid[0] * pixel_spacing,    # X-axis
#             centroid[1] * pixel_spacing,    # Y-axis
#             centroid[2] * slice_thickness  # Z-axis
#         ])
#         centroids.append(centroid_real)
#         logging.debug(f"Cluster {label}: Centroid (mm) = {centroid_real}")
    
#     return np.array(centroids)

# def compute_pca(points):
#     """
#     Compute PCA for a set of points.
    
#     Parameters:
#     points (numpy.ndarray): Array of points, shape (N, 3)
    
#     Returns:
#     centroid (numpy.ndarray): Centroid of the points, shape (3,)
#     components (numpy.ndarray): Principal components, shape (3, 3)
#     eigenvalues (numpy.ndarray): Eigenvalues corresponding to principal components, shape (3,)
#     """
#     centroid = np.mean(points, axis=0)
#     centered = points - centroid
#     covariance_matrix = np.cov(centered, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
#     # Sort eigenvectors by descending eigenvalues
#     idx = np.argsort(eigenvalues)[::-1]
#     eigenvectors = eigenvectors[:, idx]
#     eigenvalues = eigenvalues[idx]
#     return centroid, eigenvectors, eigenvalues

# def compute_rotation_matrix_procrustes(source_points, target_points):
#     """
#     Compute the rotation matrix using Orthogonal Procrustes Analysis.
    
#     Parameters:
#     source_points (numpy.ndarray): Source points, shape (N, 3)
#     target_points (numpy.ndarray): Target points, shape (N, 3)
    
#     Returns:
#     numpy.ndarray: Rotation matrix, shape (3, 3)
#     """
#     # Center the points
#     source_centroid = np.mean(source_points, axis=0)
#     target_centroid = np.mean(target_points, axis=0)
#     source_centered = source_points - source_centroid
#     target_centered = target_points - target_centroid

#     # Perform Orthogonal Procrustes
#     R, scale = orthogonal_procrustes(source_centered, target_centered)
    
#     # Ensure R is a proper rotation matrix
#     if np.linalg.det(R) < 0:
#         logging.debug("Rotation matrix determinant is negative. Adjusting to ensure proper rotation.")
#         R[:, -1] *= -1
    
#     return R

# def rotation_matrix_to_euler_angles(R, convention='zyx'):
#     """
#     Convert a rotation matrix to Euler angles based on the specified convention.
    
#     Parameters:
#     R (numpy.ndarray): Rotation matrix, shape (3, 3)
#     convention (str): Euler angle convention (default is 'zyx')
    
#     Returns:
#     tuple: (roll, pitch, yaw) in degrees
#     """
#     r = R_sc.from_matrix(R)
#     euler = r.as_euler(convention, degrees=True)
#     yaw, pitch, roll = euler
#     return roll, pitch, yaw

# def compute_angle_deviation(angle, tolerance=1.0):
#     """
#     Compute the deviation of an angle from the nearest multiple of 90°, 180°, or 270°.
    
#     If the angle is within a small tolerance of the nearest multiple, the deviation is set to 0°.
#     Otherwise, it is the difference from the nearest multiple, with sign indicating direction.
    
#     Parameters:
#     angle (float): The angle in degrees to adjust.
#     tolerance (float): The allowable deviation from multiples to set deviation to 0°.
    
#     Returns:
#     float: Adjusted angle representing the deviation from the nearest multiple of 90°, 180°, or 270°.
#     """
#     multiples = [0, 90, 180, 270, 360]
#     angle_mod = angle % 360

#     # Find the nearest multiple
#     nearest_multiple = min(multiples, key=lambda x: abs(x - angle_mod))

#     # Compute deviation
#     deviation = angle_mod - nearest_multiple

#     # Adjust deviation to be within -180° to +180°
#     if deviation > 180:
#         deviation -= 360
#     elif deviation < -180:
#         deviation += 360

#     # Set deviation to 0° if within tolerance
#     if np.isclose(deviation, 0, atol=tolerance):
#         return 0.0
#     else:
#         return deviation

# def visualize_clusters(A_points, B_points, frame_num):
#     """
#     Visualize the original and transformed configurations with clusters for a specific frame.
    
#     Parameters:
#     A_points (numpy.ndarray): Original points, shape (N, 3)
#     B_points (numpy.ndarray): Transformed points (cluster centroids), shape (N, 3)
#     frame_num (int): Frame number being visualized
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Original points
#     ax.scatter(A_points[:,0], A_points[:,1], A_points[:,2], c='blue', marker='o', label='A (Original)', alpha=0.6)

#     # Transformed points (cluster centroids)
#     ax.scatter(B_points[:,0], B_points[:,1], B_points[:,2], c='red', marker='^', label='B (Transformed)', alpha=0.6)

#     ax.set_xlabel('X (mm)')
#     ax.set_ylabel('Y (mm)')
#     ax.set_zlabel('Z (mm)')
#     ax.legend()
#     ax.set_title(f'Frame {frame_num}: Original vs Transformed Dots with Clusters')

#     plt.show()

# def main():
#     # ==================== User Section: Define Your Original Array Shape and Dot Configuration Here ====================

#     # Define the shape of the original 3D array (x, y, z)
#     array_shape = (128, 128, 89)  # Modify as needed

#     # Define the distance from the center to peripheral dots (in pixels)
#     distance_pixels = 10  # Adjust based on your specific requirements

#     # Define pixel spacing and slice thickness (in mm)
#     pixel_spacing = 1.9  # mm
#     slice_thickness = 2.8  # mm

#     # ===================== End of User Section: Define Your Original Array Shape and Dot Configuration =====================

#     # Step 1: Create the original array with seven dots
#     A = create_seven_dots(array_shape, distance=distance_pixels)
#     logging.info("Original seven-dot configuration created.")

#     # Step 2: Extract and cluster points from the original array (scaled to mm)
#     A_points = extract_and_cluster_points(
#         A, 
#         threshold=0, 
#         eps=1.5, 
#         min_samples=1, 
#         pixel_spacing=pixel_spacing, 
#         slice_thickness=slice_thickness
#     )
#     logging.info(f"Number of clusters (dots) in original configuration (A): {A_points.shape[0]}")
#     print(f"Number of clusters (dots) in original configuration (A): {A_points.shape[0]}")

#     if A_points.shape[0] != 7:
#         logging.error(f"Original configuration has {A_points.shape[0]} clusters instead of 7.")
#         print(f"Error: Original configuration has {A_points.shape[0]} clusters instead of 7.")
#         return

#     # Step 3: Compute PCA for A
#     centroid_A, components_A, eigenvalues_A = compute_pca(A_points)
#     logging.info(f"PCA for original configuration (A) computed. Centroid: {centroid_A}, Eigenvalues: {eigenvalues_A}")
#     print(f"PCA for original configuration (A) computed.")

#     # Step 4: Load the transformed 4D array B.npy
#     try:
#         B = np.load('B.npy')  # Ensure B.npy is in the same directory or provide the full path
#         logging.info("Loaded 'B.npy' successfully.")
#     except FileNotFoundError:
#         logging.error("'B.npy' not found.")
#         print("Error: 'B.npy' not found. Please ensure the file exists in the current directory or provide the correct path.")
#         return
#     except Exception as e:
#         logging.error(f"Error loading 'B.npy': {e}")
#         print(f"Error loading 'B.npy': {e}")
#         return

#     # Step 5: Validate the shape of B
#     expected_shape = (128, 128, 89, 23)  # Modify if your array shape differs
#     if B.shape != expected_shape:
#         logging.error(f"'B.npy' is expected to have shape {expected_shape}, but has shape {B.shape}.")
#         print(f"Error: 'B.npy' is expected to have shape {expected_shape}, but has shape {B.shape}.")
#         return

#     num_frames = B.shape[3]
#     logging.info(f"Number of frames in B.npy: {num_frames}")
#     print(f"Number of frames in B.npy: {num_frames}")

#     # Step 6: Provide known translations (optional)
#     # Populate the known_translations list with your provided data
#     known_translations = [
#         (0.000, 0.000, 0.000),    # Frame 0
#         (0.000, 0.000, 0.000),    # Frame 1
#         (0.000, 0.000, 0.000),    # Frame 2
#         (0.000, 0.000, 0.000),    # Frame 3
#         (-0.036, -0.009, 0.296),  # Frame 4
#         (-0.016, -0.009, 0.029),  # Frame 5
#         (0.001, -0.005, 0.045),   # Frame 6
#         (-0.004, -0.003, -0.009), # Frame 7
#         (-0.001, -0.002, 0.024),  # Frame 8
#         (-0.005, -0.001, 0.013),  # Frame 9
#         (0.000, 0.000, 0.000),    # Frame 10
#         (-0.004, -0.001, -0.000), # Frame 11
#         (-0.004, 0.000, 0.003),   # Frame 12
#         (-0.004, -0.006, 0.106),  # Frame 13
#         (-0.002, -0.001, -0.003), # Frame 14
#         (-0.015, -0.004, 0.000),  # Frame 15
#         (0.000, -0.001, 0.003),    # Frame 16
#         (0.002, -0.003, -0.005),   # Frame 17
#         (-0.001, -0.003, 0.074),   # Frame 18
#         (-0.012, -0.005, 0.065),   # Frame 19
#         (-0.001, -0.011, -0.018),  # Frame 20
#         (0.002, -0.020, 0.109),    # Frame 21
#         (-0.000, -0.009, 0.081),   # Frame 22
#     ]

#     # Validate the number of known translations matches the number of frames
#     if len(known_translations) != num_frames:
#         logging.error(f"Number of known translations ({len(known_translations)}) does not match number of frames ({num_frames}).")
#         print(f"Error: Number of known translations ({len(known_translations)}) does not match number of frames ({num_frames}).")
#         return

#     # Step 7: Prepare a list to store transformation parameters for each frame
#     transformation_parameters = []

#     # Define the threshold based on how the dots are represented in B.npy
#     # If the dots are represented by 1s, use threshold=0
#     # If by higher values, adjust accordingly (e.g., 254 for 255-valued dots)
#     threshold = 0  # Modify if necessary

#     for frame_idx in range(num_frames):
#         print(f"\nProcessing frame {frame_idx + 1}/{num_frames}...")
#         logging.info(f"Processing frame {frame_idx + 1}/{num_frames}.")

#         # Extract the 3D array for the current frame (assuming frames are along the last axis)
#         B_frame = B[:, :, :, frame_idx]

#         # Extract and cluster points (scaled to mm)
#         B_points = extract_and_cluster_points(
#             B_frame, 
#             threshold=threshold, 
#             eps=1.5, 
#             min_samples=1, 
#             pixel_spacing=pixel_spacing, 
#             slice_thickness=slice_thickness
#         )

#         # Debugging: Check the number of clusters (dots)
#         num_B_points = B_points.shape[0]
#         print(f"Number of clusters (dots) in frame {frame_idx + 1}: {num_B_points}")
#         logging.info(f"Number of clusters (dots) in frame {frame_idx + 1}: {num_B_points}")

#         if num_B_points != 7:
#             print(f"Warning: Frame {frame_idx + 1} has {num_B_points} clusters instead of 7. Skipping this frame.")
#             logging.warning(f"Frame {frame_idx + 1} has {num_B_points} clusters instead of 7. Skipping this frame.")
#             continue

#         # Compute PCA for B
#         centroid_B, components_B, eigenvalues_B = compute_pca(B_points)
#         logging.info(f"PCA for Frame {frame_idx + 1} computed. Centroid: {centroid_B}, Eigenvalues: {eigenvalues_B}")

#         # Retrieve known translation for the current frame (scaled to mm)
#         t_pixels = np.array(known_translations[frame_idx])  # (Tx, Ty, Tz) in pixels
#         t_mm = np.array([
#             t_pixels[0] * pixel_spacing,    # Tx in mm
#             t_pixels[1] * pixel_spacing,    # Ty in mm
#             t_pixels[2] * slice_thickness   # Tz in mm
#         ])
#         logging.debug(f"Known Translation (Frame {frame_idx + 1}): {t_mm} mm")

#         # Compute rotation matrix using Orthogonal Procrustes
#         R = compute_rotation_matrix_procrustes(A_points, B_points)
#         logging.debug(f"Rotation Matrix (Frame {frame_idx + 1}):\n{R}")

#         # Convert rotation matrix to Euler angles
#         # Adjust the 'convention' parameter if needed (e.g., 'zyx' or 'xyz')
#         roll, pitch, yaw = rotation_matrix_to_euler_angles(R, convention='zyx')
#         logging.debug(f"Euler Angles (Frame {frame_idx + 1}): Roll={roll}, Pitch={pitch}, Yaw={yaw}")

#         # Adjust angles to reflect deviation from nearest multiple of 90°, 180°, 270°
#         roll_deviation = compute_angle_deviation(roll)
#         pitch_deviation = compute_angle_deviation(pitch)
#         yaw_deviation = compute_angle_deviation(yaw)
#         logging.debug(f"Rotation Deviations (Frame {frame_idx + 1}): Roll={roll_deviation}, Pitch={pitch_deviation}, Yaw={yaw_deviation}")

#         # Append the parameters to the list
#         transformation_parameters.append([
#             frame_idx + 1,    # Frame number (1-based indexing)
#             t_mm[0], t_mm[1], t_mm[2],  # Translation (Tx, Ty, Tz) in mm
#             roll_deviation, pitch_deviation, yaw_deviation   # Rotation deviations (Rx_r, Ry_r, Rz_r) in degrees
#         ])

#         # Display the adjusted transformation
#         print(f"Estimated Transformation for Frame {frame_idx + 1}:")
#         print(f"Translation (Tx, Ty, Tz): ({t_mm[0]:.3f} mm, {t_mm[1]:.3f} mm, {t_mm[2]:.3f} mm)")
#         print(f"Rotation Deviation (Rx_r, Ry_r, Rz_r): ({roll_deviation:.3f}°, {pitch_deviation:.3f}°, {yaw_deviation:.3f}°)")
#         logging.info(f"Frame {frame_idx + 1} Transformation: Tx={t_mm[0]:.3f}, Ty={t_mm[1]:.3f}, Tz={t_mm[2]:.3f}, Rx_r={roll_deviation:.3f}, Ry_r={pitch_deviation:.3f}, Rz_r={yaw_deviation:.3f}")

#         # Optional: Visualize the alignment for selected frames
#         # For example, visualize every 5th frame and the last frame
#         if (frame_idx + 1) % 5 == 0 or (frame_idx + 1) == num_frames:
#             visualize_clusters(A_points, B_points, frame_idx + 1)
#             logging.info(f"Visualization for Frame {frame_idx + 1} displayed.")

#     # Step 8: Save the transformation parameters to a .txt file
#     if transformation_parameters:
#         output_filename = 'transformation_parameters.txt'
#         try:
#             with open(output_filename, 'w') as f:
#                 # Write header
#                 f.write("Frame\tTranslation_X(mm)\tTranslation_Y(mm)\tTranslation_Z(mm)\tRotation_X_Deviation(gr)\tRotation_Y_Deviation(gr)\tRotation_Z_Deviation(gr)\n")
#                 # Write data
#                 for params in transformation_parameters:
#                     frame_num, tx_mm, ty_mm, tz_mm, rx_dev, ry_dev, rz_dev = params
#                     f.write(f"{frame_num}\t{tx_mm:.3f}\t{ty_mm:.3f}\t{tz_mm:.3f}\t{rx_dev:.3f}\t{ry_dev:.3f}\t{rz_dev:.3f}\n")
#             logging.info(f"Transformation parameters saved to '{output_filename}'.")
#             print(f"\nTransformation parameters for {len(transformation_parameters)} frames have been saved to '{output_filename}'.")
#         except Exception as e:
#             logging.error(f"Error writing to '{output_filename}': {e}")
#             print(f"Error writing to '{output_filename}': {e}")
#     else:
#         logging.warning("No transformation parameters were saved because no frames met the criteria.")
#         print("\nNo transformation parameters were saved because no frames met the criteria.")

# if __name__ == "__main__":
#     main()

import numpy as np
from scipy.spatial.transform import Rotation as R_sc
from sklearn.cluster import DBSCAN
from scipy.linalg import orthogonal_procrustes


def extract_and_cluster_points(A, threshold=0, eps=1.5, min_samples=1, pixel_spacing=1.9, slice_thickness=2.8):
    points = np.argwhere(A > threshold)
    if points.size == 0:
        return np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        centroid = cluster_points.mean(axis=0)
        centroid_real = np.array([
            centroid[0] * pixel_spacing,
            centroid[1] * pixel_spacing,
            centroid[2] * slice_thickness
        ])
        centroids.append(centroid_real)

    return np.array(centroids)


def compute_rotation_matrix_procrustes(source_points, target_points):
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    R, scale = orthogonal_procrustes(source_centered, target_centered)

    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    return R


def rotation_matrix_to_euler_angles(R, convention='zyx'):
    r = R_sc.from_matrix(R)
    euler = r.as_euler(convention, degrees=True)
    return euler[2], euler[1], euler[0]


def compute_angle_deviation(angle, tolerance=1.0):
    multiples = [0, 90, 180, 270, 360]
    angle_mod = angle % 360

    nearest_multiple = min(multiples, key=lambda x: abs(x - angle_mod))
    deviation = angle_mod - nearest_multiple

    if deviation > 180:
        deviation -= 360
    elif deviation < -180:
        deviation += 360

    if np.isclose(deviation, 0, atol=tolerance):
        return 0.0
    return deviation


def process_transformation(A, B, known_translations, output_filename, pixel_spacing=1.9, slice_thickness=2.8, threshold=0, eps=1.5, min_samples=1):
    A_points = extract_and_cluster_points(
        A,
        threshold=threshold,
        eps=eps,
        min_samples=min_samples,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness
    )

    if A_points.shape[0] != 7:
        print("Error: Original configuration has incorrect number of clusters.")
        return

    num_frames = B.shape[3]
    if len(known_translations) != num_frames:
        print("Error: Known translations do not match the number of frames.")
        return

    transformation_parameters = []

    for frame_idx in range(num_frames):
        B_frame = B[:, :, :, frame_idx]

        B_points = extract_and_cluster_points(
            B_frame,
            threshold=threshold,
            eps=eps,
            min_samples=min_samples,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )

        if B_points.shape[0] != 7:
            continue

        t_pixels = np.array(known_translations[frame_idx])
        t_mm = np.array([
            t_pixels[0] * pixel_spacing,
            t_pixels[1] * pixel_spacing,
            t_pixels[2] * slice_thickness
        ])

        R = compute_rotation_matrix_procrustes(A_points, B_points)

        roll, pitch, yaw = rotation_matrix_to_euler_angles(R, convention='zyx')

        roll_dev = compute_angle_deviation(roll)
        pitch_dev = compute_angle_deviation(pitch)
        yaw_dev = compute_angle_deviation(yaw)

        transformation_parameters.append({
            "frame": frame_idx + 1,
            "translation_mm": t_mm.tolist(),
            "rotation_deviation_deg": [roll_dev, pitch_dev, yaw_dev]
        })

    # Save transformation parameters to a .txt file
    with open(output_filename, 'w') as file:
        file.write("Frame\tTranslation_X(mm)\tTranslation_Y(mm)\tTranslation_Z(mm)\tRotation_X_Deviation(deg)\tRotation_Y_Deviation(deg)\tRotation_Z_Deviation(deg)\n")
        for params in transformation_parameters:
            frame = params["frame"]
            t_mm = params["translation_mm"]
            r_dev = params["rotation_deviation_deg"]
            file.write(f"{frame}\t{t_mm[0]:.3f}\t{t_mm[1]:.3f}\t{t_mm[2]:.3f}\t{r_dev[0]:.3f}\t{r_dev[1]:.3f}\t{r_dev[2]:.3f}\n")

    print(f"Transformation parameters saved to {output_filename}")


# Example Usage
# Assuming `A`, `B`, and `known_translations` are already defined and loaded.
# A: Original 3D array
# B: 4D array (e.g., `np.load("B.npy")`)
# known_translations: List of tuples representing known translations

# Example:
# A = create_seven_dots((128, 128, 89), distance=10)  # Your 3D array creation logic
# B = np.load("B.npy")  # Ensure the correct path
# known_translations = [(0.0, 0.0, 0.0), ...]  # Define your translations
# output_filename = "transformation_param
