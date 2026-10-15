from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from Pitch_PINN.physics.half_car import EffectivePitchParameters
from Pitch_PINN.physics.full_car import SuspensionRates


@dataclass(frozen=True)
class PitchParameterBounds:
    damping_rate_s_inv: tuple[float, float]
    stiffness_rate_s2: tuple[float, float]
    ax_gain_rad_per_m: tuple[float, float]

    def __post_init__(self) -> None:
        for name, pair in (
            ("damping_rate_s_inv", self.damping_rate_s_inv),
            ("stiffness_rate_s2", self.stiffness_rate_s2),
            ("ax_gain_rad_per_m", self.ax_gain_rad_per_m),
        ):
            if len(pair) != 2 or pair[0] >= pair[1]:
                raise ValueError(f"{name} bounds must be an increasing pair")


def _inverse_sigmoid_fraction(value: float, bounds: tuple[float, float]) -> float:
    fraction = (value - bounds[0]) / (bounds[1] - bounds[0])
    epsilon = 1e-9
    fraction = min(max(fraction, epsilon), 1.0 - epsilon)
    return float(torch.logit(torch.tensor(fraction)).item())


def _bounded(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)


class BoundedPitchParameters(nn.Module):
    """Train ODE coefficients without leaving configured physical bounds."""

    def __init__(
        self,
        bounds: PitchParameterBounds,
        initial: EffectivePitchParameters,
        *,
        learn_ax_gain: bool = True,
    ) -> None:
        super().__init__()
        self.bounds = bounds
        self.initial = initial
        self.raw_damping = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid_fraction(
                    float(initial.damping_rate_s_inv), bounds.damping_rate_s_inv
                )
            )
        )
        self.raw_stiffness = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid_fraction(
                    float(initial.stiffness_rate_s2), bounds.stiffness_rate_s2
                )
            )
        )
        self.raw_ax_gain = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid_fraction(
                    float(initial.ax_gain_rad_per_m), bounds.ax_gain_rad_per_m
                )
            ),
            requires_grad=learn_ax_gain,
        )

    def values(self) -> EffectivePitchParameters:
        return EffectivePitchParameters(
            _bounded(self.raw_damping, self.bounds.damping_rate_s_inv),
            _bounded(self.raw_stiffness, self.bounds.stiffness_rate_s2),
            _bounded(self.raw_ax_gain, self.bounds.ax_gain_rad_per_m),
        )

    def regularization_loss(self) -> torch.Tensor:
        values = self.values()
        terms = []
        for value, initial, bounds in (
            (
                values.damping_rate_s_inv,
                float(self.initial.damping_rate_s_inv),
                self.bounds.damping_rate_s_inv,
            ),
            (
                values.stiffness_rate_s2,
                float(self.initial.stiffness_rate_s2),
                self.bounds.stiffness_rate_s2,
            ),
            (
                values.ax_gain_rad_per_m,
                float(self.initial.ax_gain_rad_per_m),
                self.bounds.ax_gain_rad_per_m,
            ),
        ):
            terms.append(((value - initial) / (bounds[1] - bounds[0])) ** 2)
        return torch.stack(terms).mean()


@dataclass(frozen=True)
class SuspensionRateBounds:
    k_front_npm: tuple[float, float]
    k_rear_npm: tuple[float, float]
    c_front_nspm: tuple[float, float]
    c_rear_nspm: tuple[float, float]


class BoundedSuspensionRates(nn.Module):
    def __init__(self, bounds: SuspensionRateBounds, initial: SuspensionRates) -> None:
        super().__init__()
        self.bounds, self.initial = bounds, initial
        for name in bounds.__dataclass_fields__:
            pair, value = getattr(bounds, name), float(getattr(initial, name))
            if pair[0] >= pair[1] or not pair[0] <= value <= pair[1]:
                raise ValueError(f"invalid {name} bounds or initial value")
            self.register_parameter("raw_" + name, nn.Parameter(torch.tensor(_inverse_sigmoid_fraction(value, pair))))

    def values(self) -> SuspensionRates:
        return SuspensionRates(**{name: _bounded(getattr(self, "raw_" + name), getattr(self.bounds, name)) for name in self.bounds.__dataclass_fields__})

    def regularization_loss(self) -> torch.Tensor:
        values = self.values()
        terms = [((getattr(values, n) - float(getattr(self.initial, n))) / (getattr(self.bounds, n)[1] - getattr(self.bounds, n)[0])) ** 2 for n in self.bounds.__dataclass_fields__]
        return torch.stack(terms).mean()
