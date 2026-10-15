from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .state_network import PitchStateNetwork
from .trainable_parameters import BoundedPitchParameters
from .input_signals import FeatureNormalizer, MultiSignalInterpolator
from .state_network import FullCarStateNetwork
from .trainable_parameters import BoundedSuspensionRates


def _derivative(output: torch.Tensor, input_value: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        input_value,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


@dataclass(frozen=True)
class PitchPrediction:
    time_s: torch.Tensor
    theta_rad: torch.Tensor
    pitch_rate_radps: torch.Tensor
    pitch_acceleration_radps2: torch.Tensor


class AxInterpolator(nn.Module):
    """Piecewise-linear, time-differentiable longitudinal acceleration input."""

    def __init__(self, time_s: np.ndarray, ax_mps2: np.ndarray) -> None:
        super().__init__()
        time = np.asarray(time_s, dtype=float).reshape(-1)
        ax = np.asarray(ax_mps2, dtype=float).reshape(-1)
        if time.size != ax.size or time.size < 2:
            raise ValueError("time and Ax need equal lengths of at least two")
        if not np.all(np.diff(time) > 0.0):
            raise ValueError("interpolation time must be strictly increasing")
        self.register_buffer("source_time_s", torch.as_tensor(time))
        self.register_buffer("source_ax_mps2", torch.as_tensor(ax))

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        flat = time_s.reshape(-1)
        index = torch.searchsorted(self.source_time_s, flat, right=True) - 1
        index = index.clamp(0, self.source_time_s.numel() - 2)
        t0 = self.source_time_s[index]
        t1 = self.source_time_s[index + 1]
        a0 = self.source_ax_mps2[index]
        a1 = self.source_ax_mps2[index + 1]
        fraction = (flat - t0) / (t1 - t0)
        return (a0 + fraction * (a1 - a0)).reshape_as(time_s)


class PitchPINN(nn.Module):
    def __init__(
        self,
        state_network: PitchStateNetwork,
        physical_parameters: BoundedPitchParameters,
    ) -> None:
        super().__init__()
        self.state_network = state_network
        self.physical_parameters = physical_parameters

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        return self.state_network(time_s)

    def predict_with_derivatives(self, time_s: torch.Tensor) -> PitchPrediction:
        differentiable_time = time_s
        if not differentiable_time.requires_grad:
            differentiable_time = time_s.detach().clone().requires_grad_(True)
        theta = self(differentiable_time)
        pitch_rate = _derivative(theta, differentiable_time)
        pitch_acceleration = _derivative(pitch_rate, differentiable_time)
        return PitchPrediction(
            differentiable_time, theta, pitch_rate, pitch_acceleration
        )


@dataclass(frozen=True)
class FullCarPrediction:
    time_s: torch.Tensor
    features: torch.Tensor
    state: torch.Tensor
    state_rate: torch.Tensor
    state_acceleration: torch.Tensor
    wheel_displacement: torch.Tensor
    wheel_velocity: torch.Tensor

    @property
    def pitch_rate_radps(self) -> torch.Tensor:
        return self.state_rate[:, 2:3]


class FullCarPitchPINN(nn.Module):
    def __init__(self, network: FullCarStateNetwork, physical_parameters: BoundedSuspensionRates,
                 time_s: np.ndarray, ax_mps2: np.ndarray, wheel_velocity_mps: np.ndarray,
                 wheel_acceleration_mps2: np.ndarray) -> None:
        super().__init__()
        self.state_network, self.physical_parameters = network.double(), physical_parameters.double()
        exogenous = np.column_stack([ax_mps2, wheel_acceleration_mps2])
        self.signal_interpolator = MultiSignalInterpolator(time_s, exogenous)
        normalizer = FeatureNormalizer.fit(np.column_stack([time_s, exogenous]))
        self.register_buffer("feature_mean", torch.as_tensor(normalizer.mean))
        self.register_buffer("feature_scale", torch.as_tensor(normalizer.scale))
        self.wheel_velocity_interpolator = MultiSignalInterpolator(time_s, wheel_velocity_mps)
        dt = np.diff(time_s)[:, None]
        increments = 0.5 * (wheel_velocity_mps[:-1] + wheel_velocity_mps[1:]) * dt
        wheel_z = np.vstack([np.zeros((1, 4)), np.cumsum(increments, axis=0)])
        self.wheel_displacement_interpolator = MultiSignalInterpolator(time_s, wheel_z)

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        signals = self.signal_interpolator(time_s)
        features = torch.cat([time_s, signals], dim=1)
        return self.state_network((features - self.feature_mean) / self.feature_scale)

    def predict_with_derivatives(self, time_s: torch.Tensor) -> FullCarPrediction:
        t = time_s if time_s.requires_grad else time_s.detach().clone().requires_grad_(True)
        signals = self.signal_interpolator(t)
        features = torch.cat([t, signals], dim=1)
        state = self.state_network((features - self.feature_mean) / self.feature_scale)
        rate = torch.cat([_derivative(state[:, i:i+1], t) for i in range(3)], dim=1)
        acceleration = torch.cat([_derivative(rate[:, i:i+1], t) for i in range(3)], dim=1)
        return FullCarPrediction(t, features, state, rate, acceleration,
                                 self.wheel_displacement_interpolator(t),
                                 self.wheel_velocity_interpolator(t))
