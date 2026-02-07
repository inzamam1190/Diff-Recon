import h5py
import matplotlib.pyplot as plt
import torch
import fastmri
from fastmri.data import transforms as T


def load_and_visualize(file_path, mask_outer=False):
    """
    Load MRI k-space data from an H5 file, optionally apply frequency masking,
    and visualize both the k-space (frequency domain) and reconstructed image (spatial domain).

    This function performs the following operations:
    1. Reads k-space data from an HDF5 file containing MRI measurements
    2. Extracts the middle slice from a 3D volume for analysis
    3. Converts the k-space data to a PyTorch tensor
    4. Optionally applies a mask to zero out outer 50% of k-space frequencies
    5. Applies inverse 2D FFT to transform k-space to image space
    6. Computes the magnitude of complex-valued image data
    7. Displays side-by-side visualization of log-scaled k-space and reconstructed image

    Args:
        file_path (str): Path to the H5 file containing k-space MRI data with key "kspace"
        mask_outer (bool, optional): If True, zeros out outer 50% of k-space frequencies
                                     (central 50% region retained). Defaults to False.

    Returns:
        None: Displays matplotlib figure with k-space and reconstructed image visualizations

    Raises:
        FileNotFoundError: If the specified H5 file does not exist
        KeyError: If the H5 file does not contain a "kspace" dataset
    """

    # 1. Open the H5 file
    hf = h5py.File(file_path, "r")

    # 2. Extract raw k-space data
    # Shape: (Slices, Height, Width) for singlecoil
    kspace = hf["kspace"][()]

    # Pick the middle slice
    slice_idx = kspace.shape[0] // 2
    slice_kspace = kspace[slice_idx]

    # 3. Transform to Tensor
    slice_kspace_tensor = T.to_tensor(slice_kspace)

    # --- Optional Masking to Zero Out Outer 50% of K-Space Frequencies ---
    if mask_outer:
        h, w, _ = slice_kspace_tensor.shape
        # Create a mask that zeros out the outer 50%
        mask = torch.zeros((h, w, 1))
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :] = 1
        slice_kspace_tensor = slice_kspace_tensor * mask
    # -------------------------------------------

    # 4. Apply Inverse FFT
    sampled_image = fastmri.ifft2c(slice_kspace_tensor)
    image_abs = fastmri.complex_abs(sampled_image)

    # 5. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    kspace_mag = torch.log(fastmri.complex_abs(slice_kspace_tensor) + 1e-9)
    ax[0].imshow(kspace_mag, cmap="gray")
    ax[0].set_title("Log K-Space (Frequency Domain)")

    ax[1].imshow(image_abs, cmap="gray")
    ax[1].set_title("Reconstructed Knee")

    plt.show()


if __name__ == "__main__":
    # Update this path to your specific filename!
    DATA_PATH = "data/knee_singlecoil_test/file1000000.h5"
    load_and_visualize(DATA_PATH, mask_outer=False)
