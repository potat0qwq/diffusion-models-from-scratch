# Diffusion Models from Scratch

A PyTorch implementation of diffusion models from scratch, focusing on the core mechanics of **DDPM** and **DDIM** on a 2D Swiss Roll dataset.

The project implements the forward diffusion process, noise prediction model, DDPM sampling, DDIM sampling, training pipeline, checkpointing, and reverse diffusion visualization.


## Overview

Diffusion models learn to generate data by gradually removing noise from a noisy sample.

This project provides a minimal and modular implementation of the diffusion pipeline:

```text
Data
  │
  ▼
Swiss Roll
  │
  ▼
Forward Diffusion
x₀ → xₜ
  │
  ▼
Noise Prediction Model
εθ(xₜ, t)
  │
  ▼
Reverse Diffusion
  │
  ├── DDPM Sampling
  │
  └── DDIM Sampling
  │
  ▼
Generated Samples
```

## Features

- PyTorch implementation of a discrete-time diffusion model
- Forward diffusion with a linear beta schedule
- Sinusoidal timestep positional encoding
- Residual noise prediction network
- DDPM reverse sampling
- DDIM reverse sampling
- Configurable diffusion steps
- Training loss visualization
- Reverse diffusion trajectory visualization
- Model checkpoint saving and loading
- CPU / CUDA support
- Separate training and inference scripts


## Dataset

The model is trained on a 2D Swiss Roll dataset generated using `scikit-learn`.

The original 3D Swiss Roll is projected onto two dimensions and normalized before training.

```python
X = make_swiss_roll(
    n_samples=N,
    noise=1e-1
)[0][:, [0, 2]] / 10.0
```
The low-dimensional dataset makes it possible to directly visualize the diffusion and generation processes.

## Model Architecture

The noise prediction model is a lightweight MLP with timestep conditioning.

```text
xₜ ∈ R²
 │
 ▼
Linear(2 → d_model)
 │
 ▼
Residual Block
 │
 ├── Timestep Positional Encoding
 ├── Linear
 ├── GELU
 ├── Linear
 └── LayerNorm + Residual Connection
 │
 ▼
Residual Block
 │
 ▼
Linear(d_model → 2)
 │
 ▼
Predicted Noise εθ(xₜ, t)
```
### Default Configuration
```python
d_model  = 128
n_layers = 2
n_steps  = 100
```
The timestep `t` is represented using sinusoidal positional encoding and injected into each residual block.

## Diffusion Process

The forward diffusion process gradually adds Gaussian noise to the original data.

For timestep `t`:

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

where

$$
\epsilon \sim \mathcal{N}(0,I)
$$

and

$$
\bar{\alpha}_t =
\prod_{s=1}^{t}\alpha_s
$$

The model learns to predict the noise added to the sample:

$$
\epsilon_\theta(x_t,t)
$$

The training objective is the standard noise prediction objective:

```math
\mathcal{L}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon - \epsilon_\theta(x_t,t)
\right\|^2
\right]
```

---
## DDPM Sampling

The project implements the iterative DDPM reverse process.

Starting from Gaussian noise:

```text
x_T
 │
 ▼
x_{T-1}
 │
 ▼
x_{T-2}
 │
 ▼
...
 │
 ▼
x_1
 │
 ▼
x_0
```
At each timestep, the model predicts the noise component and updates the current sample.

The implementation can optionally record the entire reverse diffusion trajectory for visualization.

## DDIM Sampling

The project also implements DDIM sampling.

Instead of using every diffusion timestep, DDIM selects a smaller set of timesteps for the reverse process.

The current implementation uses up to 20 sampling steps:

```python
ddim_steps = min(20, self.n_steps)
```
The sampler uses the deterministic setting:
```python
η = 0
```
This allows the reverse process to be visualized with substantially fewer sampling steps than the full DDPM trajectory.

## DDPM vs DDIM

DDPM and DDIM use the same trained noise prediction model but differ in their reverse sampling procedures.

| Method | Steps | Stochastic | Behavior |
|---|---:|---|---|
| DDPM | 100 | Yes | Full reverse diffusion |
| DDIM | ≤ 20 | No (`η = 0`) | Deterministic |

The project implements both methods using the same trained noise prediction model, allowing their reverse diffusion trajectories to be compared directly.

## Results

### Training Loss

The training loss is saved to:

`results/loss.png`

![Training Loss](results/loss.png)

### DDPM Reverse Diffusion

The following animation shows the DDPM reverse diffusion process, starting from Gaussian noise and progressively generating samples from the learned distribution.

![DDPM Reverse Diffusion](results/ddpm_reverse.gif)

### DDIM Reverse Diffusion

The following animation shows the deterministic DDIM reverse diffusion process using a reduced number of sampling steps.

![DDIM Reverse Diffusion](results/ddim_reverse.gif)

---
## Project Structure

```text
diffusion-models-from-scratch/
│
├── checkpoints/
│   └── model.pth
│
├── diffusion/
│   └── ddim.py
│
├── models/
│   ├── embedding.py
│   └── unet.py
│
├── utils/
│   ├── trainer.py
│   └── visualization.py
│
├── results/
│   ├── loss.png
│   ├── ddpm_reverse.gif
│   └── ddim_reverse.gif
│
├── train.py
├── sample.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Components

| File | Description |
|---|---|
| `models/embedding.py` | Sinusoidal timestep positional encoding |
| `models/unet.py` | Noise prediction network and residual blocks |
| `diffusion/ddim.py` | Diffusion process, DDPM and DDIM sampling |
| `utils/trainer.py` | Dataset preparation and training loop |
| `utils/visualization.py` | Loss curves and sampling animations |
| `train.py` | Training entry point and checkpoint saving |
| `sample.py` | Checkpoint loading and reverse diffusion sampling |

---
## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd diffusion-models-from-scratch
```
### 2. Create a Virtual Environment
Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
```
Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
---
## Training

Run:

```bash
python train.py
```
The training script:
1. Generates the Swiss Roll dataset.
2. Initializes the noise prediction model.
3. Initializes the diffusion process.
4. Trains the model using the noise prediction objective.
5. Records the training loss.
6. Saves the trained model checkpoint.
The trained model is saved to:
```text
checkpoints/model.pth
```
The loss curve is saved to:
```text
results/loss.png
```
---
## Sampling

After training, run:

```bash
python sample.py
```
The script:
1. Builds the same model architecture.
2. Loads the trained checkpoint.
3. Runs DDPM reverse sampling.
4. Runs DDIM reverse sampling.
5. Records the reverse diffusion trajectories.
6. Generates GIF visualizations.

The generated animations are saved to:
```text
results/ddpm_reverse.gif
results/ddim_reverse.gif
```
## Configuration

The main hyperparameters can be modified in `train.py` and `sample.py`.

Default training configuration:

```python
n_steps = 100
d_model = 128
n_layers = 2
batch_size = 128
n_epochs = 400
sample_size = 512
seed = 42
```
### Diffusion Schedule

The current implementation uses a linear beta schedule:

```text
beta_min = 1e-5
beta_max = 5e-3
```
The number of diffusion steps can be changed through:
```python
n_steps = 100
```
## Reproducibility

The training pipeline sets random seeds for NumPy and PyTorch:

```python
np.random.seed(seed)
torch.manual_seed(seed)
```
This improves reproducibility across runs.

Exact numerical results may still vary depending on the hardware and execution environment.
## Design Choices

### Why Use a 2D Dataset?

Diffusion models are typically demonstrated on high-dimensional image or text data, but the underlying reverse diffusion process can be difficult to inspect directly.

A 2D Swiss Roll dataset makes the process observable:

```text
Gaussian Noise
      ↓
Noisy Distribution
      ↓
Progressive Denoising
      ↓
Learned Data Distribution
```
This provides an intuitive way to study how the reverse diffusion process evolves.
### Why Implement Both DDPM and DDIM?

DDPM and DDIM share the same learned noise prediction model but use different reverse sampling procedures.

Implementing both methods makes it possible to directly compare:

- Number of sampling steps
- Stochastic vs deterministic generation
- Reverse diffusion trajectories
- Sampling behavior

---

## Future Work

- [ ] Experiment with different beta schedules
- [ ] Add configurable DDIM `η`
- [ ] Compare different DDIM sampling step counts
- [ ] Add quantitative evaluation metrics
- [ ] Extend the model to image datasets
- [ ] Implement classifier-free guidance
- [ ] Add experiment configuration files
- [ ] Add automated evaluation

---
## References

1. Ho, J., Jain, A., & Abbeel, P.  
   *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020.

2. Song, J., Meng, C., & Ermon, S.  
   *Denoising Diffusion Implicit Models.* ICLR, 2021.

---
## License

This project is released under the MIT License.