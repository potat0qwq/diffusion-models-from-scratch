import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import BasicDiscreteTimeModel
from diffusion.ddim import DDIM
from utils.trainer import train
from utils.visualization import save_all_results

# ======================
# Hyperparameters
# ======================

n_steps = 100
d_model = 128
n_layers = 2

batch_size = 128
n_epochs = 401
sample_size = 512
seed = 42

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
# Train
# ======================

losses, ddpm_samples, ddim_samples = train(
    model=model,
    diffuser=diffuser,
    batch_size=batch_size,
    n_epochs=n_epochs,
    sample_size=sample_size,
    seed=seed,
)

# ======================
# Save checkpoint
# ======================

os.makedirs("checkpoints", exist_ok=True)

checkpoint_path = "checkpoints/model.pth"

torch.save(model.state_dict(), checkpoint_path)

# Save figures
save_all_results(
    losses,
    ddpm_samples,
    ddim_samples,
)

print("\nTraining finished!")
print(f"Model saved to: {checkpoint_path}")