from pathlib import Path

import fastmri
import matplotlib.pyplot as plt
import torch

from loader import load_and_reconstruct


def create_comprehensive_plot(file_path):
    """
    Create a comprehensive visualization comparing k-space and reconstructed images
    across different filtering modes.

    This function generates a 2x3 subplot figure displaying k-space data (log scale)
    in the top row and corresponding reconstructed images in the bottom row for three
    different mask types: None (unfiltered), low_pass, and high_pass.

    Args:
        file_path (str or Path): Path to the MRI data file (typically .h5 format)
            containing k-space data to be loaded and reconstructed.

    Returns:
        None. Displays the plot and saves it as 'kspace_vs_image_comparison.png'
        in the outputs directory.

    Notes:
        - The k-space magnitude is displayed using a logarithmic scale for better
          visualization of dynamic range.
        - The output directory is determined relative to the script location
          (parent.parent / "outputs").
        - Requires the load_and_reconstruct function to be available.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        Exception: If load_and_reconstruct fails to process the file.
    """
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "outputs"

    modes = [None, "low_pass", "high_pass"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, mode in enumerate(modes):
        img_abs, kspace_tensor = load_and_reconstruct(file_path, mask_type=mode)

        # --- Top Row: K-Space (Log Scale) ---
        k_mag = torch.log(
            fastmri.complex_abs(kspace_tensor) + 1e-9
        ).numpy()  # Add small value to avoid log(0)
        axes[0, i].imshow(k_mag, cmap="gray")
        axes[0, i].set_title(f"K-Space: {mode}")
        axes[0, i].axis("off")
        axes[1, i].imshow(img_abs, cmap="gray")
        axes[1, i].set_title(f"Image: {mode}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "kspace_vs_image_comparison.png")
    plt.show()


if __name__ == "__main__":
    DATA_PATH = "/Volumes/T7 Shield/Diff-Recon/data/knee_singlecoil_val/file1000538.h5"  # Update this
    create_comprehensive_plot(DATA_PATH)
