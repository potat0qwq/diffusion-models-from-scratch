import os
import numpy as np
import torch

from models.unet import BasicDiscreteTimeModel
from diffusion.ddim import DDIM
from utils.trainer import train
from utils.visualization import save_loss_curve


# =========================
# Configuration
# =========================

n_steps = 100
d_model = 128
n_layers = 2

batch_size = 128
n_epochs = 400

seed = 42

# =========================
# Reproducibility
# =========================

np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =========================
# Model
# =========================

model = BasicDiscreteTimeModel(
    d_model=d_model,
    n_layers=n_layers,
).to(device)


# =========================
# Diffusion Process
# =========================

diffuser = DDIM(
    n_steps=n_steps,
).to(device)


# =========================
# Training
# =========================

losses = train(
    model=model,
    diffuser=diffuser,
    batch_size=batch_size,
    n_epochs=n_epochs,
    seed=seed,
)


# =========================
# Save Checkpoint
# =========================

os.makedirs(
    "checkpoints",
    exist_ok=True,
)

checkpoint_path = "checkpoints/model.pth"

torch.save(
    model.state_dict(),
    checkpoint_path,
)


# =========================
# Save Results
# =========================

save_loss_curve(losses)


print("\nTraining finished!")
print(f"Model saved to: {checkpoint_path}")