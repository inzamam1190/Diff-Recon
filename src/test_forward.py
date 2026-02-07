import torch
import matplotlib.pyplot as plt
from loader import load_and_reconstruct
from diffusion_utils import DiffusionScheduler


def test_diffusion_on_knee(file_path):
    # 1. Load a real clean slice (x_0)
    # Ensure you use a validation file for a sharp baseline
    image_abs, _ = load_and_reconstruct(file_path, mask_type=None)

    # Standardize: Diffusion models work best with normalized data
    # Let's scale it to roughly [-1, 1] or [0, 1]
    x_0 = image_abs / torch.max(image_abs)
    x_0 = x_0.unsqueeze(0)  # Add batch dimension [1, H, W]

    # 2. Initialize Scheduler
    scheduler = DiffusionScheduler(timesteps=1000)

    # 3. Sample at different stages of "destruction"
    test_steps = [0, 50, 150, 400, 999]
    fig, axes = plt.subplots(1, len(test_steps), figsize=(20, 5))

    for i, t_val in enumerate(test_steps):
        t = torch.tensor([t_val])
        x_t, _ = scheduler.add_noise(x_0, t)

        # Display
        axes[i].imshow(x_t.squeeze().numpy(), cmap="gray")
        axes[i].set_title(f"Step t={t_val}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use one of your .h5 files from the SSD
    DATA_PATH = "/Volumes/T7 Shield/Diff-Recon/data/knee_singlecoil_val/file1000628.h5"
    test_diffusion_on_knee(DATA_PATH)
