import torch
import torch.nn as nn
import numpy as np

class DDIM(nn.Module):

    def __init__(self, n_steps: int, minval: float = 1e-5, maxval: float = 5e-3):
        super().__init__()
        assert 0 < minval < maxval <= 1
        assert n_steps > 0
        self.n_steps = n_steps
        self.minval = minval
        self.maxval = maxval
        self.register_buffer("beta", torch.linspace(minval, maxval, n_steps))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(0))

    def diffusion_loss(self, model: nn.Module, inp: torch.Tensor) -> torch.Tensor:
        device = inp.device
        batch_size = inp.shape[0]

        # create the noise perturbation
        eps = torch.randn_like(inp, device=device)

        # convert discrete time into a positional encoding embedding
        t = torch.randint(0, self.n_steps, (batch_size,), device=device)

        # compute the closed form sample x_noisy after t time steps
        a_t = self.alpha_bar[t][:, None]
        x_noisy = torch.sqrt(a_t) * inp + torch.sqrt(1 - a_t) * eps

        # predict the noise added given time t
        eps_pred = model(x_noisy, t)

        # Gaussian posterior, i.e. learn the Gaussian kernel.
        return nn.MSELoss()(eps_pred, eps)

    def ddpm_sample(self, model: nn.Module, n_samples: int = 128, return_trajectory: bool = False,):
        with torch.no_grad():
            device = next(model.parameters()).device

            # start off with an intial random ensemble of particles
            x = torch.randn(n_samples, 2, device=device)
            trajectory = []

            if return_trajectory:
                trajectory.append(x.detach().cpu().clone())

            # the number of steps is fixed before beginning training. unfortunately.
            for t in reversed(range(self.n_steps)):
                # apply the same variance to all particles in the ensemble equally.
                a = self.alpha[t].repeat(n_samples)[:, None]
                abar = self.alpha_bar[t].repeat(n_samples)[:, None]

                # deterministic trajectory. eps_theta is similar to the Force on the particle
                eps_theta = model(x, torch.tensor([t] * n_samples, dtype=torch.long))
                x_mean = (x - eps_theta * (1 - a) / torch.sqrt(1 - abar)) / torch.sqrt(
                    a
                )
                sigma_t = torch.sqrt(1 - self.alpha[t])

                # sample a different realization of noise for each particle and propagate
                z = torch.randn_like(x)
                x = x_mean + sigma_t * z
                if return_trajectory:
                    trajectory.append(x_mean.detach().cpu().clone())

            if return_trajectory:
                return trajectory

            return x_mean  # clever way to skip the last noise addition


    def ddim_sample(self, model: nn.Module, n_samples: int = 128, return_trajectory: bool = False,):

      with torch.no_grad():
        device = next(model.parameters()).device
        x = torch.randn(n_samples, 2, device=device)
        trajectory = []
        if return_trajectory:
            trajectory.append(x.detach().cpu().clone())

        # number of DDIM steps (about 20)
        ddim_steps = min(20, self.n_steps)
        # create a uniform sequence of timesteps from T-1 down to 0
        timesteps = np.linspace(0, self.n_steps - 1, ddim_steps, endpoint=True)
        timesteps = np.round(timesteps).astype(np.int64)
        timesteps = np.unique(timesteps)[::-1]   # descending, unique, ensures 0 is included

        # iterate over the chosen timesteps
        for i in range(len(timesteps) - 1):
            t = timesteps[i]          # current time step (larger)
            s = timesteps[i + 1]      # next time step (smaller)

            # get alpha_bar for t and s
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_s = self.alpha_bar[s]

            # predicted noise at time t
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eps_theta = model(x, t_tensor)

            # DDIM update rule (eta = 0, deterministic)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_alpha_bar_s = torch.sqrt(alpha_bar_s)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            sqrt_one_minus_alpha_bar_s = torch.sqrt(1 - alpha_bar_s)

            # predict x0 from x_t
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
            # compute x_s
            x = sqrt_alpha_bar_s * x0_pred + sqrt_one_minus_alpha_bar_s * eps_theta
            if return_trajectory:
                trajectory.append(x.detach().cpu().clone())

        if return_trajectory:
            return trajectory
        return x