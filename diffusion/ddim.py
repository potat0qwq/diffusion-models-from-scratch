import numpy as np
import torch
import torch.nn as nn


class DDIM(nn.Module):
    def __init__(
        self,
        n_steps: int,
        beta_min: float = 1e-5,
        beta_max: float = 5e-3,
    ):
        super().__init__()

        assert 0 < beta_min < beta_max <= 1
        assert n_steps > 0

        self.n_steps = n_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Linear beta schedule
        self.register_buffer(
            "beta",
            torch.linspace(beta_min, beta_max, n_steps),
        )

        self.register_buffer(
            "alpha",
            1.0 - self.beta,
        )

        self.register_buffer(
            "alpha_bar",
            self.alpha.cumprod(dim=0),
        )

    def diffusion_loss(
        self,
        model: nn.Module,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the standard diffusion noise-prediction loss.

        A random timestep t is sampled for each input point.
        Gaussian noise is added according to the forward diffusion process,
        and the model learns to predict the added noise.
        """

        device = inp.device
        batch_size = inp.shape[0]

        # Sample Gaussian noise
        eps = torch.randn_like(inp, device=device)

        # Sample a random diffusion timestep for each data point
        t = torch.randint(
            0,
            self.n_steps,
            (batch_size,),
            device=device,
        )

        # Closed-form forward diffusion:
        #
        # x_t = sqrt(alpha_bar_t) * x_0
        #       + sqrt(1 - alpha_bar_t) * epsilon
        alpha_bar_t = self.alpha_bar[t][:, None]

        x_noisy = (
            torch.sqrt(alpha_bar_t) * inp
            + torch.sqrt(1.0 - alpha_bar_t) * eps
        )

        # Predict the noise added to x_0
        eps_pred = model(x_noisy, t)

        # Standard noise-prediction objective
        return nn.MSELoss()(eps_pred, eps)

    def ddpm_sample(
        self,
        model: nn.Module,
        n_samples: int = 128,
        return_trajectory: bool = False,
    ):
        """
        Generate samples using DDPM reverse diffusion.

        Args:
            model:
                Trained noise-prediction model.

            n_samples:
                Number of samples to generate.

            return_trajectory:
                If True, return the complete reverse diffusion trajectory.
                Otherwise, return only the final generated samples.
        """

        with torch.no_grad():
            device = next(model.parameters()).device

            # Start from Gaussian noise x_T
            x = torch.randn(
                n_samples,
                2,
                device=device,
            )

            trajectory = []

            if return_trajectory:
                trajectory.append(
                    x.detach().cpu().clone()
                )

            # Reverse diffusion:
            # x_T -> x_{T-1} -> ... -> x_0
            for t in reversed(range(self.n_steps)):
                alpha_t = self.alpha[t].repeat(
                    n_samples
                )[:, None]

                alpha_bar_t = self.alpha_bar[t].repeat(
                    n_samples
                )[:, None]

                t_tensor = torch.full(
                    (n_samples,),
                    t,
                    device=device,
                    dtype=torch.long,
                )

                # Predict epsilon_theta(x_t, t)
                eps_theta = model(
                    x,
                    t_tensor,
                )

                # Mean of the DDPM reverse transition
                x_mean = (
                    x
                    - eps_theta
                    * (1.0 - alpha_t)
                    / torch.sqrt(1.0 - alpha_bar_t)
                ) / torch.sqrt(alpha_t)

                # Add stochastic noise except at the final step
                if t > 0:
                    sigma_t = torch.sqrt(
                        1.0 - self.alpha[t]
                    )

                    z = torch.randn_like(x)

                    x = x_mean + sigma_t * z
                else:
                    x = x_mean

                if return_trajectory:
                    trajectory.append(
                        x.detach().cpu().clone()
                    )

            if return_trajectory:
                return trajectory

            return x

    def ddim_sample(
        self,
        model: nn.Module,
        n_samples: int = 128,
        return_trajectory: bool = False,
    ):
        """
        Generate samples using deterministic DDIM sampling.

        The current implementation uses at most 20 sampling timesteps
        and corresponds to eta = 0.
        """

        with torch.no_grad():
            device = next(model.parameters()).device

            # Start from Gaussian noise x_T
            x = torch.randn(
                n_samples,
                2,
                device=device,
            )

            trajectory = []

            if return_trajectory:
                trajectory.append(
                    x.detach().cpu().clone()
                )

            # Use fewer reverse steps than DDPM
            ddim_steps = min(
                20,
                self.n_steps,
            )

            timesteps = np.linspace(
                0,
                self.n_steps - 1,
                ddim_steps,
                endpoint=True,
            )

            timesteps = np.round(
                timesteps
            ).astype(np.int64)

            timesteps = np.unique(
                timesteps
            )[::-1]

            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                s = timesteps[i + 1]

                alpha_bar_t = self.alpha_bar[t]
                alpha_bar_s = self.alpha_bar[s]

                t_tensor = torch.full(
                    (n_samples,),
                    t,
                    device=device,
                    dtype=torch.long,
                )

                # Predict epsilon_theta(x_t, t)
                eps_theta = model(
                    x,
                    t_tensor,
                )

                sqrt_alpha_bar_t = torch.sqrt(
                    alpha_bar_t
                )

                sqrt_alpha_bar_s = torch.sqrt(
                    alpha_bar_s
                )

                sqrt_one_minus_alpha_bar_t = torch.sqrt(
                    1.0 - alpha_bar_t
                )

                sqrt_one_minus_alpha_bar_s = torch.sqrt(
                    1.0 - alpha_bar_s
                )

                # Estimate x_0 from x_t
                x0_pred = (
                    x
                    - sqrt_one_minus_alpha_bar_t
                    * eps_theta
                ) / sqrt_alpha_bar_t

                # Deterministic DDIM update
                x = (
                    sqrt_alpha_bar_s * x0_pred
                    + sqrt_one_minus_alpha_bar_s
                    * eps_theta
                )

                if return_trajectory:
                    trajectory.append(
                        x.detach().cpu().clone()
                    )

            if return_trajectory:
                return trajectory

            return x