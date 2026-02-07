import h5py
import matplotlib.pyplot as plt
import torch
import fastmri
from fastmri.data import transforms as T


def load_and_visualize(file_path, mask_outer=False):
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

    # --- Optional Masking (The Physics Quiz) ---
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
