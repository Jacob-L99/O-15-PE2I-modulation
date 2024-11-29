import ants
import numpy as np

def register_wat(data_4d_1, data_4d_2):
    """
    Perform registration between frame 10 of two 4D numpy datasets using ANTsPy.
    
    Parameters:
        data_4d_1: np.ndarray
            The first 4D numpy dataset (shape: [x, y, z, t]).
        data_4d_2: np.ndarray
            The second 4D numpy dataset (shape: [x, y, z, t]).
    
    Returns:
        list
            Forward transforms from the registration process.
    """
    # Validate input shapes
    if data_4d_1.ndim != 4 or data_4d_2.ndim != 4:
        raise ValueError("Both inputs must be 4D numpy arrays with shape [x, y, z, t].")
    if data_4d_1.shape[:3] != data_4d_2.shape[:3]:
        raise ValueError("Spatial dimensions of the two datasets must match.")
    if data_4d_1.shape[3] <= 10 or data_4d_2.shape[3] <= 10:
        raise ValueError("Both datasets must have at least 11 frames.")

    # Extract the 10th frame (index 9, 0-based indexing)
    frame_10_1 = data_4d_1[..., 10]
    frame_10_2 = data_4d_2[..., 10]

    # Convert numpy arrays to ANTs images
    ants_frame_10_1 = ants.from_numpy(frame_10_1)
    ants_frame_10_2 = ants.from_numpy(frame_10_2)

    # Perform quick registration
    registration = ants.registration(
        fixed=ants_frame_10_1,
        moving=ants_frame_10_2,
        type_of_transform="QuickRigid"
    )

    # Return the forward transforms
    return registration['fwdtransforms']
