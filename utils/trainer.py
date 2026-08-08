import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import make_swiss_roll
from tqdm import tqdm

def train(
    model: nn.Module,
    diffuser,
    batch_size: int = 128,
    n_epochs: int = 400,
    sample_size: int = 512,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset
    N = 1024
    X = make_swiss_roll(
        n_samples=N,
        noise=1e-1
    )[0][:, [0, 2]] / 10.0

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    losses = []
    ddpm_samples = []
    ddim_samples = []

    with tqdm(total=n_epochs) as pbar:

        for epoch in range(n_epochs):

            ids = np.random.choice(
                N,
                N,
                replace=False,
            )

            loss_epoch = []

            for i in range(0, N, batch_size):

                x = torch.tensor(
                    X[ids[i:i + batch_size]],
                    dtype=torch.float32,
                )

                optimizer.zero_grad()

                loss = diffuser.diffusion_loss(
                    model,
                    x,
                )

                loss.backward()

                optimizer.step()

                loss_epoch.append(loss.item())

            avg_loss = np.mean(loss_epoch)

            losses.append(avg_loss)

            ddpm_samples.append(
                diffuser.ddpm_sample(
                    model,
                    n_samples=sample_size,
                )
            )

            ddim_samples.append(
                diffuser.ddim_sample(
                    model,
                    n_samples=sample_size,
                )
            )

            pbar.update(1)
            pbar.set_description(
                f"Epoch {epoch} | Loss {avg_loss:.4f}"
            )

    return losses, ddpm_samples, ddim_samples