import h5py
import torch
import fastmri
from fastmri.data import transforms as T


def load_and_reconstruct(file_path, mask_type=None):
    """Load and reconstruct an MRI slice from k-space stored in an HDF5 file.

    The function loads a k-space dataset named "kspace" from the given HDF5
    file, selects the middle slice of the volume, converts it to a PyTorch
    tensor, optionally applies a frequency-domain mask, performs an inverse
    2D FFT to obtain the image, and computes the magnitude.

    Mask behavior (controlled by ``mask_type``):
        - ``None`` (default): no masking, use full k-space.
        - ``'low_pass'``: keep the central 10% of k-space (indices
          ``h//2-16:h//2+16`` and ``w//2-16:w//2+16``) and zero the rest.
        - ``'high_pass'``: zero the central third of k-space and keep the edges.

    Args:
        file_path (str): Path to the HDF5 file containing k-space data under the
            key "kspace". Expected shape is (slices, height, width) for single-coil.
        mask_type (str or None, optional): Type of mask to apply; one of
            {None, 'low_pass', 'high_pass'}. Defaults to ``None``.

    Returns:
        tuple: A tuple containing:
            - image_abs (torch.Tensor): Magnitude of the reconstructed image.
            - masked_kspace (torch.Tensor): The masked k-space tensor.

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
        KeyError: If the HDF5 file does not contain a "kspace" dataset.
    """

    hf = h5py.File(file_path, "r")
    kspace = hf["kspace"][()]
    slice_kspace = kspace[kspace.shape[0] // 2]
    slice_kspace_tensor = T.to_tensor(slice_kspace)

    h, w, _ = slice_kspace_tensor.shape

    # Create the mask
    if mask_type == "low_pass":
        mask = torch.zeros((h, w, 1))
        # Keep only a tiny 10% center to really see the blur
        mask[h // 2 - 16 : h // 2 + 16, w // 2 - 16 : w // 2 + 16, :] = 1
    elif mask_type == "high_pass":
        mask = torch.ones((h, w, 1))
        mask[h // 2 - 32 : h // 2 + 32, w // 2 - 32 : w // 2 + 32, :] = 0
    else:
        mask = torch.ones((h, w, 1))

    # Apply mask and IFFT
    masked_kspace = slice_kspace_tensor * mask
    image = fastmri.ifft2c(masked_kspace)
    image_abs = fastmri.complex_abs(image)

    return image_abs, masked_kspace
