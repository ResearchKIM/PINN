from __future__ import annotations

import sys

import torch
from torch import nn


class PitchStateNetwork(nn.Module):
    """Approximate relative body pitch angle as a smooth function of time."""

    def __init__(
        self,
        t_min_s: float,
        t_max_s: float,
        *,
        hidden_width: int = 48,
        hidden_layers: int = 3,
        angle_scale_rad: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or hidden_layers <= 0 or angle_scale_rad <= 0.0:
            raise ValueError("network dimensions and angle scale must be positive")
        self.register_buffer("t_min_s", torch.tensor(float(t_min_s)))
        duration = max(float(t_max_s) - float(t_min_s), sys.float_info.epsilon)
        self.register_buffer("duration_s", torch.tensor(duration))
        self.angle_scale_rad = float(angle_scale_rad)

        layers: list[nn.Module] = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (time_s - self.t_min_s) / self.duration_s - 1.0
        return self.angle_scale_rad * self.network(normalized)


class FullCarStateNetwork(nn.Module):
    """Predict relative sprung heave, roll and pitch from six trajectory features."""

    def __init__(self, *, hidden_width: int = 48, hidden_layers: int = 3,
                 heave_scale_m: float = 0.05, roll_scale_rad: float = 0.01,
                 pitch_scale_rad: float = 0.01) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(6, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers.append(nn.Linear(hidden_width, 3))
        self.network = nn.Sequential(*layers)
        self.register_buffer("output_scale", torch.tensor([heave_scale_m, roll_scale_rad, pitch_scale_rad]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features) * self.output_scale
