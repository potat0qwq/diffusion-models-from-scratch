import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import make_swiss_roll
from tqdm import tqdm


def train(
    model: nn.Module,
    diffuser: nn.Module,
    batch_size: int = 128,
    n_epochs: int = 400,
    seed: int = 42,
):
    """
    Train the diffusion noise-prediction model.

    The training device is inferred from the model parameters,
    allowing the same code to run on either CPU or CUDA.
    """

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Infer device from model
    device = next(model.parameters()).device

    print(f"Training on device: {device}")

    # Generate 2D Swiss Roll dataset
    N = 1024

    X = make_swiss_roll(
        n_samples=N,
        noise=1e-1,
        random_state=seed,
    )[0][:, [0, 2]] / 10.0

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    losses = []

    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):

            # Shuffle training data
            ids = np.random.choice(
                N,
                N,
                replace=False,
            )

            epoch_losses = []

            for i in range(0, len(ids), batch_size):

                # Move training batch directly to CPU / CUDA
                x = torch.tensor(
                    X[ids[i:i + batch_size]],
                    dtype=torch.float32,
                    device=device,
                )

                optimizer.zero_grad()

                loss = diffuser.diffusion_loss(
                    model,
                    x,
                )

                loss.backward()

                optimizer.step()

                epoch_losses.append(
                    loss.item()
                )

            avg_loss = np.mean(
                epoch_losses
            )

            losses.append(
                avg_loss
            )

            pbar.update(1)

            pbar.set_description(
                f"Epoch {epoch + 1}/{n_epochs} "
                f"| Loss {avg_loss:.4f}"
            )

    return losses