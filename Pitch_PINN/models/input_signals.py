from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureNormalizer":
        array = np.asarray(values, dtype=float)
        mean = np.mean(array, axis=0)
        scale = np.std(array, axis=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        return cls(mean, scale)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return (values - mean) / scale


class MultiSignalInterpolator(nn.Module):
    def __init__(self, time_s: np.ndarray, values: np.ndarray) -> None:
        super().__init__()
        time = np.asarray(time_s, dtype=float).reshape(-1)
        signals = np.asarray(values, dtype=float)
        if signals.ndim == 1:
            signals = signals[:, None]
        if len(time) != len(signals) or len(time) < 2 or not np.all(np.diff(time) > 0):
            raise ValueError("interpolation data need equal lengths and strictly increasing time")
        self.register_buffer("time_s", torch.as_tensor(time))
        self.register_buffer("values", torch.as_tensor(signals))

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        flat = time_s.reshape(-1).clamp(self.time_s[0], self.time_s[-1])
        index = (torch.searchsorted(self.time_s, flat, right=True) - 1).clamp(0, self.time_s.numel() - 2)
        t0, t1 = self.time_s[index], self.time_s[index + 1]
        fraction = ((flat - t0) / (t1 - t0)).unsqueeze(1)
        return self.values[index] + fraction * (self.values[index + 1] - self.values[index])
