"""
Diffusion model utilities for the forward diffusion processes.

This module implements the diffusion scheduler that manages the noise schedule
and handles adding noise to clean images during the forward diffusion process.
"""

import torch


class DiffusionScheduler:
    """
    Manages the noise schedule for the diffusion process.

    This scheduler implements a linear beta schedule that defines how much noise is added
    at each timestep during the forward diffusion process. It pre-computes all necessary
    coefficients for efficient noise injection using the reparameterization trick.

    Attributes:
        timesteps (int): Total number of diffusion steps (T).
        betas (torch.Tensor): Noise schedule values β_t for each timestep.
        alphas (torch.Tensor): Values 1 - β_t for each timestep.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas (ᾱ_t).
        sqrt_alphas_cumprod (torch.Tensor): √(ᾱ_t) for reparameterization.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): √(1 - ᾱ_t) for reparameterization.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        # 1. Define the linear beta schedule (the amount of noise added at each step)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

        # 2. Precalculate alphas and their cumulative products
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # 3. Precalculate sqrt values for the reparameterization trick:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, t):
        """
        Forward process: Adds Gaussian noise to a clean image at a specific timestep.

        Implements the forward diffusion process using the reparameterization trick:
            x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

        where:
            - x_0 is the clean image
            - x_t is the noised image at timestep t
            - ᾱ_t is the cumulative product of alphas at timestep t
            - ε is random Gaussian noise

        Args:
            x_start (torch.Tensor): Clean image tensor (x_0). Shape: (batch_size, channels, height, width).
            t (torch.Tensor): Timestep indices (0 to T-1). Shape: (batch_size,).

        Returns:
            tuple:
                - x_noisy (torch.Tensor): Noised image at timestep t. Same shape as x_start.
                - noise (torch.Tensor): The Gaussian noise that was added. Same shape as x_start.
        """
        noise = torch.randn_like(x_start)

        # Extract and reshape noise schedule coefficients for broadcasting
        # View as (batch_size, 1, 1) to broadcast across spatial dimensions
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1
        )

        # Apply reparameterization: mix original signal with noise
        x_noisy = (
            sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        )

        return x_noisy, noise
