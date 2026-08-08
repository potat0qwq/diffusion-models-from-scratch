import os

import torch

from models.unet import BasicDiscreteTimeModel
from diffusion.ddim import DDIM
from utils.visualization import save_sampling_gif

# ======================
# Hyperparameters
# ======================

n_steps = 100
d_model = 128
n_layers = 2
n_samples = 512

# ======================
# Build model
# ======================

model = BasicDiscreteTimeModel(
    d_model=d_model,
    n_layers=n_layers,
)

diffuser = DDIM(
    n_steps=n_steps,
)

# ======================
# Load checkpoint
# ======================

checkpoint_path = "checkpoints/model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
    )
)

model.to(device)
diffuser.to(device)
model.eval()

print("Checkpoint loaded.")

# ======================
# DDPM Sampling
# ======================

ddpm_trajectory = diffuser.ddpm_sample(
    model,
    n_samples=n_samples,
    return_trajectory=True,
)

save_sampling_gif(
    ddpm_trajectory,
    filename="ddpm_reverse.gif",
)

print("DDPM animation saved.")

# ======================
# DDIM Sampling
# ======================

ddim_trajectory = diffuser.ddim_sample(
    model,
    n_samples=n_samples,
    return_trajectory=True,
)

save_sampling_gif(
    ddim_trajectory,
    filename="ddim_reverse.gif",
)

print("DDIM animation saved.")

print("Sampling finished!")