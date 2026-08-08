import math
import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        maxlen: int = 1024,
        min_freq: float = 1e-4,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        pos_enc = self._get_pos_enc(d_model=d_model, maxlen=maxlen, min_freq=min_freq)
        self.register_buffer(
            "pos_enc", torch.tensor(pos_enc, dtype=dtype, device=device)
        )

    def _get_pos_enc(self, d_model: int, maxlen: int, min_freq: float):
        position = np.arange(maxlen)
        freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)
        pos_enc = position[:, None] * freqs[None]
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

    def forward(self, x):
        return self.pos_enc[x]