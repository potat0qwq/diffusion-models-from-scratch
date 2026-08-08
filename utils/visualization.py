import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def save_loss_curve(losses, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    np.save(
        os.path.join(save_dir, "loss.npy"),
        np.array(losses),
    )

    plt.figure(figsize=(6, 4))

    plt.plot(
        losses,
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "loss.png"),
        dpi=300,
    )

    plt.close()

    print(f"Loss curve saved to {save_dir}/loss.png")


def save_sampling_gif(
    samples,
    filename,
    save_dir="results",
    interval=50,
):

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame):

        ax.clear()

        x = samples[frame]

        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()

        ax.scatter(
            x[:, 0],
            x[:, 1],
            s=8,
        )

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

        ax.set_title(f"Step {frame}")

        ax.set_aspect("equal")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(samples),
        interval=interval,
    )

    save_path = os.path.join(
        save_dir,
        filename,
    )

    ani.save(
        save_path,
        writer="pillow",
    )

    plt.close()

    print(f"Animation saved to {save_path}")


def save_all_results(
    losses,
    ddpm_samples,
    ddim_samples,
    save_dir="results",
):

    save_loss_curve(
        losses,
        save_dir,
    )

    save_sampling_gif(
        ddpm_samples,
        "ddpm.gif",
        save_dir,
    )

    save_sampling_gif(
        ddim_samples,
        "ddim.gif",
        save_dir,
    )